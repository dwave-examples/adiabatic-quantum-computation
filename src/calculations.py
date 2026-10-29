# Copyright 2026 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import numpy as np
from scipy.optimize import curve_fit


def rescale_spectrum(
    spectrum_data: np.ndarray,
    qpu_schedule: np.ndarray,
    energy_rescaling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale a spectrum computed under ``reference_schedule`` to ``qpu_schedule``.

    Each row of ``spectrum_data`` was computed at some point on the reference schedule, identified
    by the row's B/A ratio (columns 1 and 2 of the file; the ratio is independent of the A/2, B/2
    convention). For each row, find the anneal fraction ``s`` of ``qpu_schedule`` with the same
    B/A ratio, then scale the row's energies by A_qpu(matched s) / A_ref(reference s). This
    matches the ``en_rescaling`` direction used in ``diagonalize_AK.py``. When ``qpu_schedule``
    is ``reference_schedule`` and ``energy_rescaling`` is 1, the energies are returned unchanged.

    Args:
        spectrum_data: Spectrum file contents with columns ``(s, A, B, E_0, E_1, ...)``, one row
            per point of the reference schedule. Columns 3 onward hold the ground-state and
            excited-state energies; only those columns are meaningful after rescaling.
        qpu_schedule: Annealing schedule of the target QPU as columns ``(s, A(s), B(s))``.
        energy_rescaling: Factor applied to B(s) of the QPU schedule, matching the demo's
            *Energy Rescaling* slider.

    Returns:
        A tuple containing:

        - np.ndarray: The anneal fraction of ``qpu_schedule`` matched to each row.
        - np.ndarray: The rescaled spectrum, with the same shape as ``spectrum_data``.
    """

    A_spec = np.trim_zeros(spectrum_data[:, 1], "b")
    B_spec = spectrum_data[:len(A_spec), 2]
    ratio_spec = B_spec / A_spec

    schdule_energy_rescale = 2 # A and B values in the spectrum file are multiplied by 2 relative to the schedule
    A_qpu = schdule_energy_rescale*np.trim_zeros(qpu_schedule[:, 1], "b")
    B_qpu = schdule_energy_rescale*qpu_schedule[:len(A_qpu), 2] * energy_rescaling
    ratio_qpu = B_qpu / A_qpu
    S_qpu = qpu_schedule[:len(A_qpu), 0]
    S_spec = np.interp(ratio_spec, ratio_qpu, S_qpu)

    A_qpu_at_match = np.interp(S_spec, S_qpu, A_qpu)
    E_rescale = A_qpu_at_match / A_spec

    new_spectrum_data = spectrum_data*E_rescale[:, np.newaxis]
    return S_spec, new_spectrum_data


def calculate_tau(spectra: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    """Calculate the Landau-Zener excitation timescale under a linearized approximation.

    Near the minimum gap the two-level gap is modeled as

        gap(s) ~ sqrt(e(s)^2 + Delta(s)^2) = sqrt(Dm^2 + (s - s*)^2 v^2)
        gap^2 = (Dm^2 + s*^2 v^2) - (2 s* v^2) s + v^2 s^2

    so ``gap^2`` is quadratic in ``s`` and its coefficients are fitted to the three points
    around the minimum. Equivalently, v^2 = 1/2 d^2/ds^2 gap^2(s) at s = s*. The excitation
    probability is then Pex = exp(-tf / t_ad) with t_ad = 2 hbar v / (pi Dm^2).

    Args:
        spectra: Eigenvalues as row vectors, one row per value of ``c``. Column 3 is the
            ground-state energy and column 4 the first excited-state energy.
        c: Values of the anneal parameter at which ``spectra`` is evaluated.

    Returns:
        A tuple containing:

        - float: The adiabatic timescale t_ad, where P(diabatic) = exp(-tf / t_ad).
        - float: The minimum gap Dm.
        - float: The anneal parameter s* at which the gap is minimized.

        All three are -1 if the gap is minimized at the boundary of ``c``, in which case a
        warning is issued and no fit is performed.
    """
    gap2 = (spectra[:, 4] - spectra[:, 3]) ** 2
    idx = int(np.argmin(gap2))

    if not (idx > 0 and idx < spectra.shape[0] - 1):
        warnings.warn("The gap is minimized at the boundary, fit aborted")
        return -1.0, -1.0, -1.0

    p = np.polyfit(x=c[idx - 1 : idx + 2], y=gap2[idx - 1 : idx + 2], deg=2)
    v = np.sqrt(p[0])
    cstar = -p[1] / p[0] / 2
    gap = np.sqrt(p[2] - p[0] * cstar**2)
    t_ad = 2 * v / (np.pi * gap**2)  # gap and v have units h. Need to multiply by dt_by_ds
    return t_ad, gap, cstar


def _exponential_decay(anneal_times: np.ndarray, amplitude: float, tau: float) -> np.ndarray:
    """Excitation probability model f(t) = A * exp(-t / tau).

    Args:
        anneal_times: Annealing times at which to evaluate the model.
        amplitude: The prefactor A.
        tau: The decay timescale, in the same units as ``anneal_times``.

    Returns:
        The model evaluated at each annealing time.
    """
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
        errs: Standard deviation of each measurement, used as chi-square weights. Points with
            a zero error are given unit weight.

    Returns:
        A tuple containing:

        - float: The best-fit amplitude A.
        - float: The best-fit timescale tau.
        - float: The one-sigma standard error of A, from the fit covariance.
        - float: The one-sigma standard error of tau, from the fit covariance.

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
