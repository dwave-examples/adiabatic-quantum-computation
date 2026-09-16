import plotly.graph_objects as go

from demo_configs import ANNEALING_TIME_SERIES, MODEL_COLORS_RESULTS, MODEL_COLORS_THEORY
from src.demo_enums import ModelType, ProblemType
from src.model_loading import ProblemModel
from src.values import ENERGY_TIME_RESCALING, Y_AXIS_RESCALING


def get_spectrum_plot(problem_type: ProblemType, model_type: ModelType) -> go.Figure:
    """Generates a plot of the energy spectrum for a given problem and model type.
    Currently uses pre-computed data from CSV files, though we may want to change
    this later."""

    filedata = ProblemModel.load_spectrum_scatter_from_file(problem_type, model_type)
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
        xaxis_title="Anneal Progress",
        yaxis_title="Excitation Energy (GHz)",
    )

    return fig


def get_results_plot(job_results: dict, problem_type: ProblemType, qpu_name: str) -> go.Figure:
    """Generates a plot of the results for a given problem type and QPU. Plots the
    theoretical results as lines so that the experimental results can be compared to them.

    Args:
        job_results (dict): A dictionary containing the results of the jobs for each model type.
        problem_type (ProblemType): The type of problem for which the results are being plotted.
        qpu_name (str): The name of the QPU for which the results are being plotted.

    Returns:
        go.Figure: A Plotly figure object containing the results plot."""
    fig = go.Figure()

    for model in ModelType:
        theory_xs, theory_ys = ProblemModel.load_theory_data_from_file(problem_type, model)

        y_axis_min = -1.0

        # Crude patch to rescale theory data for specific QPUs. Should be replaced
        # with a better implementation later.
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
                name=model.name.title().replace("_", " ") + " Theory",
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
                    name=model.name.title().replace("_", " ") + " Samples",
                )
            )

    fig.update_layout(
        xaxis_title="Annealing Time (ns)",
        yaxis_title="Excited State Probability",
    )

    fig.update_xaxes(range=[0, 21])
    fig.update_yaxes(type="log", range=[y_axis_min, 0])

    return fig
