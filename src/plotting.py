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
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go

from demo_configs import (
    ENERGY_GAP_COLOR,
    EXCITED_STATE_COLORS,
    MODEL_COLORS,
    REFERENCE_QPU,
    SHOW_PARALLEL_SAMPLES,
)
from src.calculations import calculate_tau, fit_exponential, rescale_spectrum
from src.demo_enums import ModelType, ProblemType, ScatterType, TheoryCurveMode
from src.model_loading import ProblemModel
from src.qpu_resources import get_schedule_filepath
from src.values import (
    DEFAULT_PLOT_WIDTH,
    MAX_PLOT_WIDTH,
    RESULTS_FIG_Y_MINS,
    SAMPLE_POINT_SIZE,
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


def make_scatter(
    x_vals: list[float],
    y_vals: list[float],
    scatter_type: ScatterType,
    other_params: dict,
    other_data: list | None = None,
) -> go.Scatter:
    """Encapsulates the specific plotting and formatting logic for each of the various
    types of scatter plots created by generate_plots. This keeps that function cleaner
    and allows the plot details to be checked and edited in one place, rather than
    being scattered throughout a function that does many other things.

    Args:
        x_vals: The x coordinates for the scatter plot.
        y_vals: The y coordinates for the scatter plot.
        scatter_type: The type of scatter plot to make. Each has different formatting,
            and may expect different inputs in the last two arguments.
        other_params: Extra parameters needed for the selected plot type.
        other_data: Extra data, like error bars.

    Returns:
        The scatter plot object.
    """

    if scatter_type is ScatterType.THEORY_REF:
        model = other_params["model"]
        scatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=f"{model.label} {TheoryCurveMode.REFERENCE.label}",
            line=dict(color=MODEL_COLORS[model], width=LINE_WIDTH),
        )
    elif scatter_type is ScatterType.THEORY_LZ:
        model = other_params["model"]
        amplitude = other_params["amplitude"]
        tau = other_params["tau"]
        amplitude_err = other_params.get("amplitude_err")
        tau_err = other_params.get("tau_err")
        if amplitude_err is not None and tau_err is not None:
            formula = (
                f"({round(amplitude, 2)}±{round(amplitude_err, 2)})"
                f"exp(-t/({round(tau, 2)}±{round(tau_err, 2)}))"
            )
        else:
            formula = f"{round(amplitude, 2)}exp(-t/{round(tau, 2)})"
        scatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(color=MODEL_COLORS[model], width=LINE_WIDTH, dash="dash"),
            name=f"{model.label}: {formula}",
        )

    elif scatter_type is ScatterType.SAMPLE:
        model = other_params["model"]
        y_errs = other_data
        is_parallel = other_params["parallel"]
        if is_parallel:
            sample_point_size = SAMPLE_POINT_SIZE + 4
            point_label = "Sample Mean"
        else:
            sample_point_size = SAMPLE_POINT_SIZE
            point_label = "Sample"
        scatter = go.Scatter(
            x=x_vals,
            y=y_vals,
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
    elif scatter_type is ScatterType.PARALLEL_SAMPLES:
        model = other_params["model"]
        point_label = "Samples"
        scatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                symbol="circle",
                size=max(SAMPLE_POINT_SIZE - 3, 1),
                color=MODEL_COLORS[model],
                line=MARKER_OUTLINE,
            ),
            name=f"{model.label} {point_label}",
        )
    elif scatter_type is ScatterType.SPECTRUM:
        series_labels = ["1st", "2nd", "3rd", "4th", "5th"]
        column_idx = other_params["col_idx"]
        scatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(color=EXCITED_STATE_COLORS[column_idx], width=LINE_WIDTH),
            name=f"{series_labels[column_idx]} Excited State",
        )
    elif scatter_type is ScatterType.ENERGY_GAP:
        scatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=f"Energy Gap {round(y_vals[1], 2)} GHz",
            line=dict(color=ENERGY_GAP_COLOR, width=LINE_WIDTH, dash="dash"),
        )
    else:
        raise ValueError(f"Unknown scatter type {scatter_type}")

    return scatter


@lru_cache
def get_schedule(qpu_name: str) -> np.ndarray | None:
    """Load the fast-anneal schedule of a QPU from ``schedules_and_embeddings``. The result is
    cached, so the file is read once per QPU.

    Args:
        qpu_name: The name of the QPU for which to load the schedule.

    Returns:
        A numpy array with columns ``(s, A(s), B(s), c)`` in GHz, as published in the schedule
        file, or None if no schedule file exists for the QPU.
    """
    schedule_filepath = get_schedule_filepath(qpu_name)

    if schedule_filepath is None:
        return None

    with open(schedule_filepath, "r") as f:
        schedule_data = np.loadtxt(f, delimiter=",", skiprows=1)
    return schedule_data


def generate_plots(
    job_results: dict,
    qpu_name: str,
    selected_model: ModelType,
    theory_modes: list[str],
    energy_rescaling: float = 1.0,
    problem_type: ProblemType = ProblemType.DEFAULT,
    show_parallel_samples: bool = SHOW_PARALLEL_SAMPLES,
) -> tuple[go.Figure, go.Figure]:
    """Generates both the results plot and the spectrum plot for the selected model.

    Args:
        job_results: Results from the QPU, keyed by model name, then by annealing time (ns),
            giving the excited state probability measured by each parallel embedding.
        qpu_name: The name of the selected QPU.
        selected_model: The model whose spectrum is plotted.
        theory_modes: The ``TheoryCurveMode`` values (as strings) of the theory curves to show.
        energy_rescaling: The factor to rescale energies by.
        problem_type: The type of problem (currently only default available).
        show_parallel_samples: Whether to show the per-embedding samples instead of error bars.

    Returns:
        A tuple containing:

        - go.Figure: The results plot of excited state probability against annealing time.
        - go.Figure: The energy spectrum plot of the selected model.
    """

    qpu_schedule = get_schedule(qpu_name)
    ref_schedule = get_schedule(REFERENCE_QPU)

    results_fig = go.Figure()
    spectrum_fig = go.Figure()

    plot_ref = str(TheoryCurveMode.REFERENCE.value) in theory_modes
    plot_lz = str(TheoryCurveMode.LANDAU_ZENER.value) in theory_modes

    spectrum_to_plot = None
    spectrum_s_vals = None
    tau_qpu_by_model = {}
    has_schedule = qpu_schedule is not None

    for model in ModelType:
        theory_xs, theory_ys = ProblemModel.load_theory_data_from_file(model, problem_type)
        base_spectrum = ProblemModel.load_spectrum_data_from_file(model, problem_type)

        # tau_ref is always computed at full (unscaled) bqm strength, regardless of energy_rescaling
        ref_s_rescaled, ref_spectrum_rescaled = rescale_spectrum(
            base_spectrum, ref_schedule, energy_rescaling=1.0
        )
        tau_ref, _, __ = calculate_tau(ref_spectrum_rescaled, ref_s_rescaled)

        if has_schedule:
            # See demo_configs to check which QPU is used for reference schedule
            s_rescaled, spectrum_rescaled = rescale_spectrum(
                base_spectrum, qpu_schedule, energy_rescaling=energy_rescaling
            )
            tau_qpu, gap, c_gap = calculate_tau(spectrum_rescaled, s_rescaled)
            if tau_qpu <= 0:  # calculate_tau could not fit the gap
                tau_qpu = None
        else:
            # No schedule file for this QPU: display the reference spectrum and skip tau.
            s_rescaled, spectrum_rescaled = ref_s_rescaled, ref_spectrum_rescaled
            _, gap, c_gap = calculate_tau(spectrum_rescaled, s_rescaled)
            tau_qpu = None

        tau_qpu_by_model[model] = tau_qpu

        if model == selected_model:
            spectrum_to_plot = spectrum_rescaled
            spectrum_s_vals = s_rescaled
            spectral_gap = gap
            gap_c_val = c_gap

        if plot_ref and tau_qpu is not None:
            x_mult = tau_ref / tau_qpu
            x_vals = [x / x_mult for x in theory_xs[:-1]]
            y_vals = theory_ys[:-1]
            results_fig.add_trace(
                make_scatter(x_vals, y_vals, ScatterType.THEORY_REF, {"model": model})
            )

    egs_column = spectrum_to_plot[:, 3]
    excited_columns = [spectrum_to_plot[:, i] for i in range(4, 9)]

    for idx, column in enumerate(excited_columns):
        spectrum_fig.add_trace(
            make_scatter(
                spectrum_s_vals, column - egs_column, ScatterType.SPECTRUM, {"col_idx": idx}
            )
        )

    spectrum_fig.add_trace(
        make_scatter([gap_c_val, gap_c_val], [0, spectral_gap], ScatterType.ENERGY_GAP, {})
    )

    spectrum_fig.update_yaxes(type="log", range=[-1, 2])
    spectrum_fig.update_xaxes(
        range=[float(np.min(spectrum_s_vals)), float(np.max(spectrum_s_vals))]
    )

    spectrum_fig.update_layout(
        xaxis_title="Anneal Progress", yaxis_title="Excitation Energy (GHz)", **PLOT_LAYOUT
    )

    x_range = DEFAULT_PLOT_WIDTH
    min_y_val = 1
    model_fit_data = {}

    for model in ModelType:
        if len(job_results[model.name]) > 0:
            anneal_times = []
            egs_probs = []
            y_errs = []
            par_times = []
            par_probs = []

            for ta, stat_set in job_results[model.name].items():
                anneal_times.append(ta)
                egs_probs.append(np.mean(stat_set))
                y_errs.append(np.std(stat_set))

                if show_parallel_samples:
                    par_times.extend([ta] * len(stat_set))
                    par_probs.extend(stat_set)

                set_min = min(stat_set)
                if set_min < min_y_val:
                    min_y_val = set_min

            fit_errs = list(y_errs)
            model_fit_data[model] = (anneal_times, egs_probs, fit_errs)

            if show_parallel_samples:
                results_fig.add_trace(
                    make_scatter(
                        par_times, par_probs, ScatterType.PARALLEL_SAMPLES, {"model": model}
                    )
                )

            # The per-embedding samples already show the spread, so skip the error bars then.
            other_params = {"model": model, "parallel": show_parallel_samples}
            plotted_errs = [] if show_parallel_samples else y_errs
            results_fig.add_trace(
                make_scatter(
                    anneal_times, egs_probs, ScatterType.SAMPLE, other_params, plotted_errs
                )
            )

            if max(anneal_times) > DEFAULT_PLOT_WIDTH - 1:
                x_range = MAX_PLOT_WIDTH

    if plot_lz:
        for model in ModelType:
            anneal_times, egs_probs, fit_errs = model_fit_data.get(model, ([], [], []))

            if len(anneal_times) >= 2:
                try:
                    fit_amplitude, tau, amplitude_err, tau_err = fit_exponential(
                        np.array(anneal_times),
                        np.array(egs_probs),
                        np.array(fit_errs),
                    )
                except (RuntimeError, ValueError) as exc:
                    warnings.warn(f"Landau-Zener fit failed for {model.label}: {exc}")
                    continue
                fit_times = np.linspace(0, max(anneal_times), 100)
                lz_params = {
                    "model": model,
                    "amplitude": fit_amplitude,
                    "tau": tau,
                    "amplitude_err": amplitude_err,
                    "tau_err": tau_err,
                }
            else:
                # Not enough data to fit; default the amplitude and keep the rescaled time scale.
                tau = tau_qpu_by_model[model]
                if tau is None:
                    continue
                fit_amplitude = 1.0
                fit_times = np.linspace(0, x_range, 100)
                lz_params = {"model": model, "amplitude": fit_amplitude, "tau": tau}

            fit_probs = fit_amplitude * np.exp(-fit_times / tau)
            results_fig.add_trace(
                make_scatter(fit_times, fit_probs, ScatterType.THEORY_LZ, lz_params)
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

    if has_schedule:
        info_text = (
            f"Landau-Zener timescale, τ = {round(tau_qpu_by_model[selected_model], 3)},"
            "<br>estimated from schedule"
        )
    else:
        info_text = (
            f"No schedule file for {qpu_name}<br>Spectrum shown for {REFERENCE_QPU}"
            "<br>Landau-Zener timescale not calculated"
        )

    spectrum_fig.add_annotation(
        text=info_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.55,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        align="left",
        font=dict(size=11),
    )

    return results_fig, spectrum_fig
