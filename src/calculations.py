import numpy as np
import warnings

def rescale_spectrum(
        spectrum_data: np.ndarray,
        qpu_schedule: np.ndarray,
        reference_schedule: np.ndarray,  
        energy_rescaling: float = 1.0
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
    B = qpu_schedule[:column_length, 2]*energy_rescaling
    S_qpu = qpu_schedule[:column_length, 0]

    s_vals = np.interp(B_ref/A_ref, B/A, S_qpu)
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
    t_ad = 2*v / (np.pi * gap**2)  # gap and v have units h. Need to multiply by dt_by_ds
    return t_ad, gap, cstar


def fit_exponential_amplitude(
    anneal_times: np.ndarray, probs: np.ndarray, errs: np.ndarray, tau: float
) -> float:
    """Chi-square fit of amplitude A for f(t) = A * exp(-t / tau), tau held fixed.

    Args:
        anneal_times: Annealing times (ns) at which probs were measured.
        probs: Measured excited state probabilities.
        errs: Standard deviation of each measurement, used as chi-square weights.
        tau: Fixed decay time scale.

    Returns:
        Best-fit amplitude A.
    """
    basis = np.exp(-anneal_times / tau)
    sigma = np.where(errs > 0, errs, 1.0)  # unweighted fallback for zero-error points
    weights = 1 / sigma**2
    return np.sum(weights * probs * basis) / np.sum(weights * basis**2)
