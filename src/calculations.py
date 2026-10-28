import warnings

import numpy as np
from scipy.optimize import curve_fit


def rescale_spectrum(
    spectrum_data: np.ndarray,
    qpu_schedule: np.ndarray,
    reference_schedule: np.ndarray,
    energy_rescaling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale spectrum_data (computed under reference_schedule, row-aligned) to qpu_schedule.

    For each native (reference) row, find qpu_schedule's s where its B/A ratio matches the
    native row's B_ref/A_ref ratio, then scale by A_qpu(matched s) / A_ref(native row) --
    matching diagonalize_AK.py's en_rescaling direction exactly.
    """
    column_length = len(spectrum_data[:, 0])
    A_ref = reference_schedule[:column_length, 1]
    B_ref = reference_schedule[:column_length, 2]

    A = qpu_schedule[:column_length, 1]
    B = qpu_schedule[:column_length, 2] * energy_rescaling
    S_qpu = qpu_schedule[:column_length, 0]

    s_vals = np.interp(B_ref / A_ref, B / A, S_qpu)
    A_qpu_at_match = np.interp(s_vals, S_qpu, A)
    E_rescale = A_qpu_at_match / A_ref

    multiplier = np.zeros((column_length, column_length))
    # At qpu_schedule == reference_schedule this is the identity (E_rescale == 1).
    np.fill_diagonal(multiplier, E_rescale)

    new_spectrum_data = np.matmul(multiplier, spectrum_data)
    return s_vals, new_spectrum_data


def calculate_tau(spectra: np.ndarray, c: np.ndarray) -> float:
    """Calculate the Landau-Zener excitation form under a linearized approximation

    gap(s) ~ root(e(s)^2 + Delta(s)^2) = root(Dm^2 + (s-s*)^2 v^2)
    gap^2 = (Dm^2 + s*^2 v^2) - (2 s* v^2)s + v^2 s^2
    # gap2 is quadratic model in s, fit coefficients:
    Pex = exp(- tf /t_ad)
    t_ad = 2 hbar v / ( pi * Dm^2)

    equivalent statement: v^2 = 1/2 d^2/ds^2 gap^2(s) at s=s*

    Args:
        spectra: sequential eigenvalues as row vectors, for given c.
        c: c values at which spectra is evaluated

    Returns:
        * adiabatic time scale t_ad. P(diabatic) = exp(- tf /t_ad)
    """
    gap2 = (spectra[:, 4] - spectra[:, 3]) ** 2
    idx = int(np.argmin(gap2))
    if not (idx > 0 and idx < spectra.shape[0] - 1):
        warnings.warn("The gap is minimized at the boundary, fit aborted")
        return -1, -1, -1, -1
    p = np.polyfit(x=c[idx - 1 : idx + 2], y=gap2[idx - 1 : idx + 2], deg=2)
    v = np.sqrt(p[0])
    cstar = -p[1] / p[0] / 2
    gap = np.sqrt(p[2] - p[0] * cstar**2)
    t_ad = 2 * v / (np.pi * gap**2)  # gap and v have units h. Need to multiply by dt_by_ds
    return t_ad, gap, cstar


def _exponential_decay(anneal_times: np.ndarray, amplitude: float, tau: float) -> np.ndarray:
    """Excitation probability model f(t) = A * exp(-t / tau)."""
    return amplitude * np.exp(-anneal_times / tau)


def fit_exponential(
    anneal_times: np.ndarray, probs: np.ndarray, errs: np.ndarray
) -> tuple[float, float, float, float]:
    """Two-parameter weighted chi-square fit of A and tau for f(t) = A * exp(-t / tau).

    A stable initial tau is seeded from a linear fit to log(probs) with A fixed to 1, then
    both parameters are refined together by weighted non-linear least squares. Probabilities
    are assumed strictly within (0, 1), so the logarithm in the seeding stage is well defined.

    Args:
        anneal_times: Annealing times (ns) at which probs were measured.
        probs: Measured excited state probabilities, strictly within (0, 1).
        errs: Standard deviation of each measurement, used as chi-square weights.

    Returns:
        A tuple (amplitude, tau, amplitude_err, tau_err) of the best-fit parameters and their
        one-sigma standard errors (from the fit covariance).

    Raises:
        RuntimeError: If the non-linear least squares fit does not converge.
    """
    times = np.asarray(anneal_times, dtype=float)
    probs = np.asarray(probs, dtype=float)
    sigma = np.asarray(errs, dtype=float)
    sigma = np.where(sigma > 0, sigma, 1.0)  # unweighted fallback for zero-error points

    # Stage 1: linear fit of log(p) = -t / tau (A fixed at 1) for a stable initial tau.
    slope = np.sum(times * np.log(probs)) / np.sum(times**2)
    tau_init = -1.0 / slope if slope < 0 else float(times.max())

    # Stage 2: weighted non-linear least squares for A and tau jointly.
    popt, pcov = curve_fit(
        _exponential_decay,
        times,
        probs,
        p0=[1.0, tau_init],
        sigma=sigma,
        absolute_sigma=True,
    )
    amplitude, tau = popt
    amplitude_err, tau_err = np.sqrt(np.diag(pcov))
    return amplitude, tau, amplitude_err, tau_err
