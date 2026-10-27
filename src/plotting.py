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
    REFERENCE_QPU,
)
from src.demo_enums import ModelType, ProblemType, TheoryCurveMode, ScatterType
from src.model_loading import ProblemModel
from src.calculations import rescale_spectrum, calculate_tau, fit_exponential_amplitude
from src.qpu_resources import get_schedule_filepath
from src.values import (
    DEFAULT_PLOT_WIDTH,
    MAX_PLOT_WIDTH,
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
        x_vals, 
        y_vals, 
        type, 
        other_params: dict, 
        other_data=None):
    """Encapsulates the specific plotting and formatting logic for each of the various
    types of scatter plots created by generate_plots. This keeps that function cleaner
    and allows the plot details to be checked and edited in one place, rather than
    being scattered throughout a function that does many other things.
    
    Args:
        x_vals: The x coordinates for the scatter plot.
        y_vals: The y coordinates.
        type: Which type of scatter plot to make. Each has different formatting,
            and may expect different inputs in the last two arguments.
        other_params: Extra parameters needed for the selected plot type.
        other_data: Extra data, like error bars.

    Returns:
        The scatter plot object.
    """

    if type == ScatterType.THEORY_REF:
        model = other_params["model"]
        scatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=f"{TheoryCurveMode.REFERENCE.extended_label}",
            line=dict(color=MODEL_COLORS[model], width=LINE_WIDTH),
        )
    elif type == ScatterType.THEORY_LZ:
        model = other_params["model"]
        scatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(color=MODEL_COLORS[model], width=LINE_WIDTH, dash="dash"),
            name=f"{TheoryCurveMode.LANDAU_ZENER.extended_label}",
        )

    elif type == ScatterType.SAMPLE: 
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
    elif type == ScatterType.PARALLEL_SAMPLES:
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
    elif type == ScatterType.SPECTRUM:
        series_labels = ["1st", "2nd", "3rd", "4th", "5th"]
        column_idx = other_params["col_idx"]
        scatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            marker=dict(color=EXCITED_STATE_COLORS[column_idx]),
            name=f"{series_labels[column_idx]} Excited State",
        )
    elif type == ScatterType.ENERGY_GAP:
        scatter = go.Scatter(
            x=x_vals,
            y = y_vals,
            mode="lines",
            name=f"Energy Gap {round(y_vals[1], 2)} GHz",
            line=dict(color=ENERGY_GAP_COLOR, width=LINE_WIDTH, dash="dash")
        )
    else:
        raise ValueError(f"Unknown scatter type {type}")

    return scatter
    

def get_schedule(qpu_name: str="Advantage2_system1") -> np.ndarray:
    """Load an annealing schedule, in the convention H/h = -A * sum_i X_i + sum_{i<j} Jij Z_i Z_j.
    """
    schedule_filepath = get_schedule_filepath(qpu_name)
    with open(schedule_filepath, "r") as f:
        schedule_data = np.loadtxt(f, delimiter=",", skiprows=1)
    return schedule_data


def generate_plots(
    job_results: dict|None,
    qpu_name: str,
    selected_model: ModelType,
    theory_mode: TheoryCurveMode,
    energy_rescaling: float = 1.0,
    problem_type: ProblemType = ProblemType.DEFAULT,
    show_parallel_samples: bool = SHOW_PARALLEL_SAMPLES,
) -> tuple[go.Figure, go.Figure]:
    """ Generates both the results plot and the spectrum plot for the selected model. 

    Args:
        job_results: results from QPU
        qpu_name: name of the QPU
        selected_model: which model to plot
        theory_mode: which theory curves to show
        energy_rescaling: factor to rescale energies by
        problem_type: type of problem (currently only default available)
        show_parallel_samples: whether to show parallel samples

    Returns:
        A tuple containing:
        - go.Figure: results plot
        - go.Figure: spectrum plot

    """

    qpu_schedule = get_schedule(qpu_name)
    ref_schedule = get_schedule(REFERENCE_QPU)
    
    results_fig = go.Figure()
    spectrum_fig = go.Figure()

    plot_ref = (theory_mode == TheoryCurveMode.REFERENCE or theory_mode == TheoryCurveMode.BOTH)
    plot_lz = (theory_mode == TheoryCurveMode.LANDAU_ZENER or theory_mode == TheoryCurveMode.BOTH)

    spectrum_to_plot = None
    spectrum_s_vals = None
    tau_qpu_by_model = {}

    for model in ModelType:
        theory_xs, theory_ys = ProblemModel.load_theory_data_from_file(problem_type, model)
        base_spectrum = ProblemModel.load_spectrum_scatter_from_file(model, problem_type)
        s_rescaled, spectrum_rescaled = rescale_spectrum(base_spectrum, qpu_schedule, ref_schedule, energy_rescaling=energy_rescaling)

        # See demo_configs to check which QPU is used for reference schedule
        tau_qpu, gap, c_gap = calculate_tau(spectrum_rescaled, s_rescaled)
        tau_qpu_by_model[model] = tau_qpu

        if model == selected_model:
            spectrum_to_plot = spectrum_rescaled
            spectrum_s_vals = s_rescaled
            spectral_gap = gap
            gap_c_val = c_gap

        # tau_ref is always computed at full (unscaled) bqm strength, regardless of energy_rescaling
        ref_s_rescaled, ref_spectrum_rescaled = rescale_spectrum(base_spectrum, ref_schedule, ref_schedule, energy_rescaling=1.0)
        tau_ref, _, __ = calculate_tau(ref_spectrum_rescaled, ref_s_rescaled)
        x_mult =  tau_ref / tau_qpu

        print(f"Model: {model.label}, QPU{qpu_name}, tau_qpu: {tau_qpu}, tau_ref: {tau_ref}, mult: {x_mult}")

        if plot_ref:
            x_vals = [x / x_mult for x in theory_xs[:-1]]
            y_vals = theory_ys[:-1]
            results_fig.add_trace(make_scatter(x_vals, y_vals, ScatterType.THEORY_REF, {"model": model}))

    egs_column = spectrum_to_plot[:, 3]
    excited_columns = [spectrum_to_plot[:, i] for i in range(4,9)]

    for idx, column in enumerate(excited_columns):
        spectrum_fig.add_trace(
            make_scatter(spectrum_s_vals, column - egs_column, ScatterType.SPECTRUM, {"col_idx": idx})
        )

    spectrum_fig.add_trace( make_scatter([gap_c_val, gap_c_val], [0, spectral_gap], ScatterType.ENERGY_GAP, {}))


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

            if show_parallel_samples:
                results_fig.add_trace(
                    make_scatter(par_times, par_probs, ScatterType.PARALLEL_SAMPLES, {"model": model})
                )


            other_params = {"model": model, "parallel": show_parallel_samples}
            results_fig.add_trace(
                make_scatter(anneal_times, egs_probs, ScatterType.SAMPLE, other_params, y_errs)
            )

            if max(anneal_times) > DEFAULT_PLOT_WIDTH - 1:
                x_range = MAX_PLOT_WIDTH

            if len(anneal_times) >= 2 and plot_lz:
                fit_amplitude = fit_exponential_amplitude(
                    np.array(anneal_times), np.array(egs_probs), np.array(fit_errs), tau_qpu_by_model[model]
                )
                fit_times = np.linspace(0, max(anneal_times), 100)
                fit_probs = fit_amplitude * np.exp(-fit_times / tau_qpu_by_model[model])
                results_fig.add_trace(
                    make_scatter(fit_times, fit_probs, ScatterType.THEORY_LZ, {"model": model})
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

    spectrum_fig.add_trace( #TODO Jack, if you want to change the display of tau, do it here
        go.Scatter(
            x=[gap_c_val], y=[50],
            mode="markers+text",
            marker=dict(size=1),
            text=[f"τ = {round(tau_qpu,3)}"], 
            textposition="top center"
            )
    )

    return results_fig, spectrum_fig
