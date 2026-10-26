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

import numpy as np
import plotly.graph_objects as go

from demo_configs import (
    ENERGY_GAP_COLOR,
    EXCITED_STATE_COLORS,
    MODEL_COLORS,
    SHOW_PARALLEL_SAMPLES,
    RESULTS_FIG_Y_MINS,
)
from src.demo_enums import ModelType, ProblemType
from src.model_loading import ProblemModel
from src.qpu_resources import get_schedule_filepath
from src.values import (
    DEFAULT_PLOT_WIDTH,
    ENERGY_TIME_RESCALING,
    MAX_PLOT_WIDTH,
    SAMPLE_POINT_SIZE,
    Y_AXIS_RESCALING,
)

LINE_WIDTH = 2
MARKER_OUTLINE = dict(color="white", width=1)
ERROR_BAR = dict(thickness=3, width=8)  # Stem thickness and cap width in pixels.

PLOT_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=60, r=10, t=10, b=50),
    legend=dict(
        xanchor="left",
        x=0.01,
        yanchor="bottom",
        y=0.01,
        bgcolor="rgba(255, 255, 255, 0.7)",
    ),
)



def make_scatter(x, y, type, id, other_args=None):

    if type == "Theory":
        tau = other_args["tau"]
        scatter = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"{id.label} Reference [\u03c4(Landau-Zener)={tau:.3g}]",
            line=dict(color=MODEL_COLORS[id], width=LINE_WIDTH),
        )

    elif type == "Sample": #TODO finish and apply later
        model = other_args["model"]
        y_errs = other_args["err"]
        sample_point_size = other_args["point_size"]
        point_label = other_args["label"]
        go.Scatter(
                    x=x,
                    y=y,
                    error_y=dict(type="data", array=y_errs, visible=True, **ERROR_BAR),
                    mode="markers",
                    marker=dict(
                        symbol="diamond-wide",
                        size=sample_point_size,
                        color=MODEL_COLORS[model],
                        line=MARKER_OUTLINE,
                    ),
                    name=f"{model.label} {point_label}",
                )
    elif type == "Parallel":
        pass
    elif type == "Spectrum":
        series_labels = ["1st", "2nd", "3rd", "4th", "5th"]
        scatter = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"{series_labels[id]} Excited State",
        )
    return scatter
    

def get_schedule(qpu_name: str="Advantage2_system1") -> np.ndarray:
    """Load an annealing schedule, in the convention H/h = -A * sum_i X_i + sum_{i<j} Jij Z_i Z_j.

    Schedule files store A(s), B(s) assuming the convention
    H/h = -(A/2) * sum_i X_i + (B/2) * sum_{i<j} Jij Z_i Z_j, so the loaded columns
    are doubled here to match the factor-of-2-free convention used throughout this module.
    """
    schedule_filepath = get_schedule_filepath(qpu_name)
    with open(schedule_filepath, "r") as f:
        schedule_data = np.loadtxt(f, delimiter=",", skiprows=1)
    schedule_data[:, 1:3] *= 2
    return schedule_data

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
    t_ad = 2*v / (
        np.pi * gap**2
    )  # gap and v have units h. Need to multiply by dt_by_ds
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


def generate_plots(
    job_results: dict|None,
    qpu_name: str,
    selected_model: ModelType,
    energy_rescaling: float = 1.0,
    problem_type: ProblemType = ProblemType.DEFAULT,
    show_parallel_samples: bool = SHOW_PARALLEL_SAMPLES,
) -> tuple[go.Figure, go.Figure]:
    """
    """

    qpu_schedule = get_schedule(qpu_name)
    spectrum_ref_schedule = get_schedule("Advantage2_system1")
    tau_ref_schedule = get_schedule("Advantage_system6")
    
    results_fig = go.Figure()
    spectrum_fig = go.Figure()

    spectrum_to_plot = None
    spectrum_s_vals = None
    tau_qpu_by_model = {}

    for model in ModelType:
        theory_xs, theory_ys = ProblemModel.load_theory_data_from_file(problem_type, model)
        base_spectrum = ProblemModel.load_spectrum_scatter_from_file(model, problem_type)
        s_rescaled, spectrum_rescaled = rescale_spectrum(base_spectrum, qpu_schedule, spectrum_ref_schedule, energy_rescaling=energy_rescaling)

        # The reference is Advantage_system6 at full bqm sscaling
        # Advantage_system6 has schedule closest 
        tau_qpu, gap, c_gap = calculate_tau(spectrum_rescaled, s_rescaled)
        tau_qpu_by_model[model] = tau_qpu

        if model == selected_model:
            spectrum_to_plot = spectrum_rescaled
            spectrum_s_vals = s_rescaled
            spectral_gap = gap
            spectral_gap_c_val = c_gap

        # tau_ref is always computed at full (unscaled) bqm strength, regardless of energy_rescaling
        ref_s_rescaled, ref_spectrum_rescaled = rescale_spectrum(base_spectrum, tau_ref_schedule, spectrum_ref_schedule, energy_rescaling=1.0)
        tau_ref, _, __ = calculate_tau(ref_spectrum_rescaled, ref_s_rescaled)
        x_multiplier =  tau_ref / tau_qpu

        print(f"Model: {model.label}, QPU{qpu_name}, tau_qpu: {tau_qpu}, tau_ref: {tau_ref}, mult: {x_multiplier}")

        theory_xs = [x / x_multiplier for x in theory_xs[:-1]]
        theory_ys = theory_ys[:-1]
        results_fig.add_trace(
            make_scatter(theory_xs, theory_ys, "Theory", model, other_args={"tau": tau_qpu})
        )

    egs_column = spectrum_to_plot[:, 3]
    excited_columns = [spectrum_to_plot[:, i] for i in range(4,9)]

    for idx, column in enumerate(excited_columns):
        spectrum_fig.add_trace(
            make_scatter(spectrum_s_vals, column - egs_column, "Spectrum", idx)
        )

    spectrum_fig.add_trace(
        go.Scatter(
            x=[spectral_gap_c_val, spectral_gap_c_val],
            y = [0, spectral_gap],
            mode="lines",
            name=f"Energy Gap {round(spectral_gap, 2)} GHz",
            line=dict(color=ENERGY_GAP_COLOR, width=LINE_WIDTH, dash="dash")
        )
    )


    spectrum_fig.update_yaxes(type="log", range=[-1, 2])
    spectrum_fig.update_xaxes(range=[float(np.min(spectrum_s_vals)), float(np.max(spectrum_s_vals))])

    spectrum_fig.update_layout(
        xaxis_title="Anneal Progress", yaxis_title="Excitation Energy (GHz)", **PLOT_LAYOUT
    )

    x_range = DEFAULT_PLOT_WIDTH
    min_y_val = 1

    for model in ModelType:
        if len(job_results[model.name]) > 0:
            anneal_times = []
            egs_probs = []
            y_errs = []
            parallel_times = []
            parallel_probs = []
            sample_point_size = SAMPLE_POINT_SIZE
            sample_point_label = "Samples"

            for ta, stat_set in job_results[model.name].items():
                anneal_times.append(ta)
                egs_probs.append(np.mean(stat_set))
                y_errs.append(np.std(stat_set))

                if show_parallel_samples:
                    parallel_times.extend([ta] * len(stat_set))
                    parallel_probs.extend(stat_set)

                set_min = min(stat_set)
                if set_min < min_y_val:
                    min_y_val = set_min

            fit_errs = list(y_errs)

            if show_parallel_samples:
                sample_point_size += 4
                y_errs = []
                results_fig.add_trace(
                    go.Scatter(
                        x=parallel_times,
                        y=parallel_probs,
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=max(sample_point_size - 7, 1),
                            color=MODEL_COLORS[model],
                            line=MARKER_OUTLINE,
                        ),
                        name=f"{model.label} {sample_point_label}",
                    )
                )
                sample_point_label = "Sample Mean"

            results_fig.add_trace(
                go.Scatter(
                    x=anneal_times,
                    y=egs_probs,
                    error_y=dict(type="data", array=y_errs, visible=True, **ERROR_BAR),
                    mode="markers",
                    marker=dict(
                        symbol="diamond-wide",
                        size=sample_point_size,
                        color=MODEL_COLORS[model],
                        line=MARKER_OUTLINE,
                    ),
                    name=f"{model.label} {sample_point_label}",
                )
            )

            if max(anneal_times) > DEFAULT_PLOT_WIDTH - 1:
                x_range = MAX_PLOT_WIDTH

            if len(anneal_times) >= 2:
                fit_amplitude = fit_exponential_amplitude(
                    np.array(anneal_times), np.array(egs_probs), np.array(fit_errs), tau_qpu_by_model[model]
                )
                fit_times = np.linspace(0, max(anneal_times), 100)
                fit_probs = fit_amplitude * np.exp(-fit_times / tau_qpu_by_model[model])
                results_fig.add_trace(
                    go.Scatter(
                        x=fit_times,
                        y=fit_probs,
                        mode="lines",
                        line=dict(color=MODEL_COLORS[model], width=LINE_WIDTH, dash="dash"),
                        name=f"Landau-Zener Theory fit y=A exp(-t/{tau_qpu_by_model[model]:.3g})",
                    )
                )

    

    results_fig.update_layout(
        xaxis_title="Annealing Time (ns)", yaxis_title="Excited State Probability", **PLOT_LAYOUT
    )

    y_axis_min = RESULTS_FIG_Y_MINS[0]
    # pick the largest y_axis_min that still shows our smallest data point from among preset vals
    for y_min in RESULTS_FIG_Y_MINS: 
        if min_y_val < 10**y_axis_min: 
            y_axis_min = y_min

    results_fig.update_xaxes(range=[0, x_range])
    results_fig.update_yaxes(type="log", range=[y_axis_min, 0])

    return results_fig, spectrum_fig
