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
    DEFAULT_PLOT_WIDTH,
    MAX_PLOT_WIDTH,
    MODEL_COLORS_RESULTS,
    MODEL_COLORS_THEORY,
    SAMPLE_POINT_SIZE,
    SHOW_PARALLEL_SAMPLES,
)
from src.demo_enums import ModelType, ProblemType
from src.model_loading import ProblemModel
from src.values import ENERGY_TIME_RESCALING, Y_AXIS_RESCALING

PLOT_LAYOUT = dict(
    margin=dict(l=60, r=10, t=10, b=50),
    legend=dict(
        xanchor="left",
        x=0.01,
        yanchor="bottom",
        y=0.01,
        bgcolor="rgba(255, 255, 255, 0.7)",
    ),
)


def get_spectrum_plot(
    model_type: ModelType,
    problem_type: ProblemType = ProblemType.DEFAULT,
    energy_rescaling: float = 1.0,
) -> go.Figure:
    """Generates a plot of the energy spectrum for a given problem and model type.
    Uses pre-computed data from CSV files.

    Args:
        problem_type: The type of problem for which the spectrum is being plotted. Currently
            only the default problem type is supported, but more can be added in the future.
        model_type: The type of model for which the spectrum is being plotted.
        energy_rescaling: A factor by which to rescale the energy values for plotting.

    Returns:
        A Plotly figure object containing the spectrum plot."""

    filedata = ProblemModel.load_spectrum_scatter_from_file(model_type, problem_type)
    fig = go.Figure()

    rows = filedata[:, 2:]
    columns = [rows[:, i] for i in range(len(rows[0]))]

    series_labels = ["1st", "2nd", "3rd", "4th", "5th"]

    min_gap = min(columns[0] - filedata[:, 1]) * energy_rescaling
    min_gap_idx = np.argmin(columns[0] - filedata[:, 1])
    min_gap_x = filedata[min_gap_idx, 0]

    for idx, column in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=filedata[:, 0],
                y=(column - filedata[:, 1]) * energy_rescaling,
                mode="lines",
                name=series_labels[idx] + " Excited State",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[min_gap_x, min_gap_x],
            y=[0, min_gap],
            mode="lines",
            name=f"Energy Gap {round(min_gap, 2)} GHz",
            line=dict(color="black"),
        )
    )

    fig.update_yaxes(type="log", range=[-1.6, 0.6])

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

        # TODO: Replace with better implementation. Currently a crude patch to rescale theory data for specific QPUs.
        for qpu_prefix, rescale in ENERGY_TIME_RESCALING.items():
            if qpu_name.startswith(qpu_prefix):
                theory_ys = [y**rescale for y in theory_ys]
                if qpu_prefix in Y_AXIS_RESCALING:
                    y_axis_min = Y_AXIS_RESCALING[qpu_prefix]  # For Advantage2, rescaling requires
                    # a larger range in the y-axis to see the data.

        fig.add_trace(
            go.Scatter(
                x=theory_xs[:-1],
                y=theory_ys[:-1],
                mode="lines",
                name=f"{model.label} Theory",
                line=dict(color=MODEL_COLORS_THEORY[model]),
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
                            size=sample_point_size - 4,
                            color=MODEL_COLORS_RESULTS[model],
                        ),
                        name=f"{model.label} {sample_point_label}",
                    )
                )
                sample_point_label = "Sample Mean"

            fig.add_trace(
                go.Scatter(
                    x=anneal_times,
                    y=egs_probs,
                    error_y=dict(type="data", array=y_errs, visible=True),
                    mode="markers",
                    marker=dict(
                        symbol="diamond-wide",
                        size=sample_point_size,
                        color=MODEL_COLORS_RESULTS[model],
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
