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

import plotly.graph_objects as go

from demo_configs import ANNEALING_TIME_SERIES, MODEL_COLORS_RESULTS, MODEL_COLORS_THEORY
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


def get_spectrum_plot(model_type: ModelType, problem_type: ProblemType=ProblemType.DEFAULT) -> go.Figure:
    """Generates a plot of the energy spectrum for a given problem and model type.
    Uses pre-computed data from CSV files.

    Args:
        problem_type: The type of problem for which the spectrum is being plotted. Currently
            only the default problem type is supported, but more can be added in the future.
        model_type: The type of model for which the spectrum is being plotted.

    Returns:
        A Plotly figure object containing the spectrum plot."""

    filedata = ProblemModel.load_spectrum_scatter_from_file(model_type, problem_type)
    fig = go.Figure()

    rows = filedata[:, 2:]
    columns = [rows[:, i] for i in range(len(rows[0]))]

    series_labels = ["1st", "2nd", "3rd", "4th", "5th"]

    for idx, column in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=filedata[:, 0],
                y=column - filedata[:, 1],
                mode="lines",
                name=series_labels[idx] + " Excited State",
            )
        )

    fig.update_yaxes(type="log", range=[-1, 0.6])

    fig.update_layout(
        xaxis_title="Anneal Progress", yaxis_title="Excitation Energy (GHz)", **PLOT_LAYOUT
    )

    return fig


def get_results_plot(job_results: dict, qpu_name: str, problem_type: ProblemType=ProblemType.DEFAULT) -> go.Figure:
    """Generates a plot of the results for a given problem type and QPU. Plots the
    theoretical results as lines so that the experimental results can be compared to them.

    Args:
        job_results: A dictionary containing the results of the jobs for each model type.
        problem_type: The type of problem for which the results are being plotted. Currently
            only the default problem type is supported, but more can be added in the future.
        qpu_name: The name of the QPU for which the results are being plotted.

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
                x=theory_xs[:-2],
                y=theory_ys[:-2],
                mode="lines",
                name=f"{model.label} Theory",
                line=dict(color=MODEL_COLORS_THEORY[model]),
            )
        )

    for model in ModelType:
        if len(job_results[model.name]) > 0:
            anneal_times = []
            egs_probs = []

            for idx, stat in enumerate(job_results[model.name]):
                anneal_times.append(ANNEALING_TIME_SERIES[(idx) % len(ANNEALING_TIME_SERIES)])
                egs_probs.append(stat)

            fig.add_trace(
                go.Scatter(
                    x=anneal_times,
                    y=egs_probs,
                    mode="markers",
                    marker=dict(symbol="x", size=8, color=MODEL_COLORS_RESULTS[model]),
                    name=f"{model.label} Samples",
                )
            )

    fig.update_layout(
        xaxis_title="Annealing Time (ns)", yaxis_title="Excited State Probability", **PLOT_LAYOUT
    )

    fig.update_xaxes(range=[0, 21])
    fig.update_yaxes(type="log", range=[y_axis_min, 0])

    return fig
