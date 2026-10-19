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
)
from src.demo_enums import ModelType, ProblemType
from src.model_loading import ProblemModel
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

def rescale_spectrum(
        spectrum_data: np.ndarray, 
        qpu_name: str, 
        reference_schedule: np.ndarray, 
        energy_rescaling: float = 1.0
    ) -> np.ndarray:
    qpu_schedule = None
    new_spectrum_data = spectrum_data * energy_rescaling
    return new_spectrum_data

def rescale_annealing_time(qpu_name: str, energy_rescaling: float = 1.0) -> float:
    qpu_factor = 1.0 if qpu_name not in ENERGY_TIME_RESCALING else ENERGY_TIME_RESCALING[qpu_name]
    return energy_rescaling/qpu_factor

def get_spectrum_plot(
    model_type: ModelType,
    qpu_name: str,
    problem_type: ProblemType = ProblemType.DEFAULT,
    energy_rescaling: float = 1.0,
) -> go.Figure:
    """Generates a plot of the energy spectrum for a given problem and model type.
    Uses pre-computed data from CSV files.

    Args:
        model_type: The type of model for which the spectrum is being plotted.
        problem_type: The type of problem for which the spectrum is being plotted. Currently
            only the default problem type is supported, but more can be added in the future.
        energy_rescaling: A factor by which to rescale the energy values for plotting.

    Returns:
        A Plotly figure object containing the spectrum plot."""

    filedata = ProblemModel.load_spectrum_scatter_from_file(model_type, problem_type)
    fig = go.Figure()

    t_vals = filedata[:, 0]
    e_rows = filedata[:, 3:]
    rescaled_e_rows = rescale_spectrum(qpu_name, e_rows, energy_rescaling)
    egs_column = rescaled_e_rows[:, 0]
    excited_columns = [rescaled_e_rows[:, i] for i in range(1,len(rescaled_e_rows[0]))]

    series_labels = ["1st", "2nd", "3rd", "4th", "5th"]

    gaps = excited_columns[0] - egs_column
    min_gap_idx = np.argmin(gaps)
    min_gap = gaps[min_gap_idx]
    min_gap_x = filedata[min_gap_idx, 0]

    for idx, column in enumerate(excited_columns):
        fig.add_trace(
            go.Scatter(
                x=t_vals,
                y=column - egs_column,
                mode="lines",
                name=series_labels[idx] + " Excited State",
                line=dict(color=EXCITED_STATE_COLORS[idx], width=LINE_WIDTH),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[min_gap_x, min_gap_x],
            y=[0, min_gap],
            mode="lines",
            name=f"Energy Gap {round(min_gap, 2)} GHz",
            line=dict(color=ENERGY_GAP_COLOR, width=LINE_WIDTH, dash="dash"),
        )
    )

    fig.update_yaxes(type="log", range=[-1, 2])

    fig.update_layout(
        xaxis_title="Anneal Progress", yaxis_title="Excitation Energy (GHz)", **PLOT_LAYOUT
    )

    return fig


def get_results_plot(
    job_results: dict,
    qpu_name: str,
    problem_type: ProblemType = ProblemType.DEFAULT,
    show_parallel_samples: bool = SHOW_PARALLEL_SAMPLES,
) -> go.Figure:
    """Generates a plot of the results for a given problem type and QPU. Plots the
    theoretical results as lines so that the experimental results can be compared to them.

    Args:
        job_results: A dictionary containing the results of the jobs for each model type.
        qpu_name: The name of the QPU for which the results are being plotted.
        problem_type: The type of problem for which the results are being plotted. Currently
            only the default problem type is supported, but more can be added in the future.
        show_parallel_samples: When this is set to True, the mean of the samples from
            each parallel embedding will be plotted on the graph as well as the overall
            mean of all samples. This is mostly intended for calibration--it shows the
            spread of the data across different embeddings--but it may also be an interesting
            mode for an actual presentation of the demo.

    Returns:
        A Plotly figure object containing the results plot.
    """
    fig = go.Figure()

    for model in ModelType:
        theory_xs, theory_ys = ProblemModel.load_theory_data_from_file(problem_type, model)

        y_axis_min = -1.0

        x_rescale = rescale_annealing_time(qpu_name)

        fig.add_trace(
            go.Scatter(
                x=[x*x_rescale for x in theory_xs[:-1]],
                y=theory_ys[:-1],
                mode="lines",
                name=f"{model.label} Theory",
                line=dict(color=MODEL_COLORS[model], width=LINE_WIDTH),
            )
        )

    x_range = DEFAULT_PLOT_WIDTH

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

            if show_parallel_samples:
                sample_point_size += 4
                y_errs = []
                fig.add_trace(
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

            fig.add_trace(
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

    fig.update_layout(
        xaxis_title="Annealing Time (ns)", yaxis_title="Excited State Probability", **PLOT_LAYOUT
    )

    fig.update_xaxes(range=[0, x_range])
    fig.update_yaxes(type="log", range=[y_axis_min, 0])

    return fig
