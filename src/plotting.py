import plotly.graph_objects as go

from src.demo_enums import ProblemType, ModelType
from src.model_loading import ProblemModel
from demo_configs import ANNEALING_TIME_SERIES

def get_spectrum_plot(problem_type: ProblemType, model_type: ModelType) -> go.Figure:
    filedata = ProblemModel.load_scatter_csv_from_file(problem_type, model_type)
    fig = go.Figure()

    rows = filedata[:,2:]
    columns = [rows[:,i] for i in range(len(rows[0]))]

    for column in columns:
        fig.add_trace(
            go.Scatter(
                x=filedata[:, 0],
                y=column - filedata[:,1],
                mode='lines',
                )
            )

    return fig

def get_results_plot(job_results: dict) -> go.Figure:
    """"""
    fig = go.Figure()

    anneal_times = []
    egs_probs = []

    for idx, stat in enumerate(job_results):
        anneal_times.append(ANNEALING_TIME_SERIES[(idx)%len(ANNEALING_TIME_SERIES)])
        egs_probs.append(stat)

    fig.add_trace(go.Scatter(
        x=anneal_times,
        y=egs_probs,
        mode='lines+markers',
        marker=dict(symbol='x', size=8),
        #name = ModelType(int(model_val)).name
    ))

    fig.update_xaxes(
        tick0=0.0, 
        dtick=ANNEALING_TIME_SERIES[-1]/len(ANNEALING_TIME_SERIES),
        range = [0, ANNEALING_TIME_SERIES[-1]+1.0]
    )
    fig.update_yaxes(tick0=0.1, range= [0.0,1.0])

    return fig
