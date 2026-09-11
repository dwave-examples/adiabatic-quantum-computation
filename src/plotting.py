import plotly.graph_objects as go

from src.demo_enums import ProblemType, ModelType
from src.model_loading import ProblemModel
from demo_configs import ANNEALING_TIME_SERIES, MODEL_COLORS_THEORY, MODEL_COLORS_RESULTS

def get_spectrum_plot(problem_type: ProblemType, model_type: ModelType) -> go.Figure:
    filedata = ProblemModel.load_scatter_csv_from_file(problem_type, model_type)
    fig = go.Figure()

    rows = filedata[:,2:]
    columns = [rows[:,i] for i in range(len(rows[0]))]

    series_labels = ["1st", "2nd", "3rd", "4th", "5th"]

    for idx, column in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=filedata[:, 0],
                y=column - filedata[:,1],
                mode='lines',
                name=series_labels[idx] + " Excited State",
                )
            )

    fig.update_yaxes(range=[0,4])

    fig.update_layout(
        xaxis_title="Anneal Progress",
        yaxis_title="Excitation Energy (GHz)",
    )

    return fig


def get_results_plot(job_results: dict, problem_type: ProblemType, qpu_name: str) -> go.Figure:
    """"""
    fig = go.Figure()

    for model in ModelType:
        theory_xs, theory_ys = ProblemModel.load_theory_data_from_file(problem_type, model)
        fig.add_trace(
            go.Scatter(
                x=theory_xs[:-2],
                y=theory_ys[:-2],
                mode='lines+markers',
                name=model.name.title().replace("_", " ") + " Theory",
                marker=dict(symbol = 'square', color=MODEL_COLORS_THEORY[model]),
            )
        )

    for model in ModelType:
        if len(job_results[model.name]) > 0:
            anneal_times = []
            egs_probs = []

            for idx, stat in enumerate(job_results[model.name]):
                anneal_times.append(ANNEALING_TIME_SERIES[(idx)%len(ANNEALING_TIME_SERIES)])
                egs_probs.append(stat)

            fig.add_trace(go.Scatter(
                x=anneal_times,
                y=egs_probs,
                mode='markers',
                marker=dict(symbol='x', size=8, color=MODEL_COLORS_RESULTS[model]),
                name=model.name.title().replace("_", " ") + " Samples",
            ))

    fig.update_layout(
        xaxis_title="Annealing Time (ns)",
        yaxis_title="Excited State Probability",
    )

    fig.update_yaxes(type='log', range=[-1,0])

    return fig
