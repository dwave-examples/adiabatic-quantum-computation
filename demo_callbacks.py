# Copyright 2024 D-Wave
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

from __future__ import annotations

from src.demo_enums import SolverType

import time

import dash
import dimod
import numpy as np
import plotly.graph_objects as go
from dash import ALL, MATCH, Patch, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
from dwave.system import DWaveSampler, ParallelEmbeddingComposite

from src.demo_enums import JobState, ModelType, ProblemType
from src.data_loading import ProblemModel
from demo_configs import ANNEALING_TIME_SERIES
from src.process_embeddings import EmbeddingSet

@dash.callback(
    Output({"type": "to-collapse-class", "index": MATCH}, "className"),
    Output({"type": "collapse-trigger", "index": MATCH}, "aria-expanded"),
    inputs=[
        Input({"type": "collapse-trigger", "index": MATCH}, "n_clicks"),
        State({"type": "to-collapse-class", "index": MATCH}, "className"),
    ],
    prevent_initial_call=True,
)
def toggle_left_column(collapse_trigger: int, to_collapse_class: str) -> tuple[str, str]:
    """Toggles a 'collapsed' class that hides and shows some aspect of the UI.

    Args:
        collapse_trigger: The (total) number of times a collapse button has been clicked.
        to_collapse_class: Current class name of the thing to collapse, 'collapsed' if not
            visible, empty string if visible.

    Returns:
        A tuple containing:

        - str: The new class name of the thing to collapse.
        - str: The aria-expanded value.
    """

    classes = to_collapse_class.split(" ") if to_collapse_class else []
    if "collapsed" in classes:
        classes.remove("collapsed")
        return " ".join(classes), "true"
    return to_collapse_class + " collapsed" if to_collapse_class else "collapsed", "false"

#===================================================================================================================


#@dash.callback(
#    Output("schedule-filename", "children"),
#    Output("schedule-filename", "className"),
#    Input("qpu-selection", "value"),
#)
#def set_schedule(qpu_name: str) -> tuple[str, str]:
    """Set the schedule for the selected QPU.

    Args:
        qpu_name: The name of the selected QPU.

    Returns:
        A tuple containing:

        - str: The name of the schedule file to use for the selected QPU.
        - str: The class to apply to the schedule filename display.
    """

    #TODO: get schedule and simplify.

 #   schedule_filename = "FALLBACK_SCHEDULE.csv"
 #   schedule_filename_class = "no-schedule"

 #   if qpu_name:
 #       for file in SCHEDULES_EMBEDDINGS_PATH.glob("*_schedule.csv"):  # get schedule files
 #           file_solver_name = _schedule_solver_name(file)
#
 #           if file_solver_name and qpu_name == file_solver_name:
  #              schedule_filename = file.name
   #             schedule_filename_class = ""
#
 #   return schedule_filename, schedule_filename_class

#======================================================================================================================

@dash.callback(
    Output("current-embeddings", "data", allow_duplicate=True),
    Output("run-button", "disabled", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Output("qpu-select-time", "data"),
    Input("qpu-selection", "value"),
    prevent_initial_call=True, #TODO deal with startup QPU behavior
)
def process_qpu_select(qpu_name: str) -> tuple[dict, str]:
    """Removes stored value for current embedding and disables run button.
    This will trigger the set_current_embeddings callback to run. If an appropriate
    embedding is already cached, it will be used and the run button will be re-enabled
    almost immediately. If not, the run button will remain disabled until the embedding 
    is found and cached.
    """

    return {}, True, JobState.FINDING_EMBEDDING.label, time.time()

#======================================================================================================================

@dash.callback(
    Output("current-embeddings", "data", allow_duplicate=True),
    Output("run-button", "disabled", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Input("qpu-select-time", "data"),
    State("current-problem-type", "data"), #TODO add proper target
    State("qpu-selection", "value"),
    State("available-solvers", "data"),
    prevent_initial_call=True,
)
def set_current_embeddings(
        qpu_select_time: float, 
        problem_type: str, 
        qpu_name: str,
        qpu_ids: dict
        ) -> tuple[dict, str]:
    """
    """

    emb_set = EmbeddingSet(qpu_name)
    problem_embeddings = emb_set.get_embeddings()

    return problem_embeddings, False, JobState.READY.label

#======================================================================================================================

#@dash.callback(
#    Output("cached-embeddings", "data"),
#    Input("uncached-embeddings", "data"),
#)
#def cache_new_embeddings(uncached_embeddings: dict) -> dict:
#    """Update the cached embeddings with any new embeddings found during the current run."""
#    embedding_update = Patch() 
#    uncached_key, uncached_value = next(iter(uncached_embeddings.items()))
#    embedding_update[uncached_key] = uncached_value
#    return embedding_update

#======================================================================================================================

#@dash.callback(
 #   Output("current-model", "data", allow_duplicate=True),
  #  Output("run-button", "disabled", allow_duplicate=True),
   # Input("qpu-selection", "value"),
#)
#def process_model_select(qpu_name: str) -> tuple[dict, str]:
 #   """Removes stored value for current embedding and disables run button.
  #  This will trigger the set_current_embeddings callback to run. If an appropriate
   # embedding is already cached, it will be used and the run button will be re-enabled
    #almost immediately. If not, the run button will remain disabled until the embedding 
    #is found and cached.
    #"""

    #return {}, True

#======================================================================================================================

@dash.callback(
    Output("current-model", "data"),
    Input("model-selection", "value"),
    State("current-problem-type", "data"), 
    State("cached-models", "data"),
    prevent_initial_call=True,
)
def set_current_model(model_val: str, problem_type: str, cached_models: dict) -> tuple[dict, str]:
    return cached_models[ProblemModel(ProblemType[problem_type], ModelType(int(model_val))).lookup_key]

#==============================================================================================

@dash.callback(
    Output("run-button", "disabled", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Output("job-submit-time", "data"),
    inputs=[
        Input("run-button", "n_clicks"),

    ],
    prevent_initial_call=True,
)
def run_button_click(
    run_btn_click: int,
) -> tuple[bool, str]:
    """Start simulation run when button is clicked.

    Args:

    Returns:
        A NamedTuple, RunButtonClickReturn, containing:

        - run_button_disabled: Whether the run button should be disabled.
        - wd_job_disabled: Whether the interval component should be disabled.
        - wd_job_n_intervals: The number of intervals that have passed since the simulation started.
        - job_submit_state: The new state of the job submission process.
    """

    return True, JobState.PREPROCESSING.label, time.time()

#======================================================================================================================

@dash.callback(
    Output("job-submit-args", "data"),
    Output("job-submit-state", "children", allow_duplicate=True),
    inputs=[
        Input("job-submit-time", "data"),
        State("job-submit-state", "children"), 
        State("current-model", "data"),
        State("model-selection", "value"),
        State("current-embeddings", "data"),
        State("current-results", "data"),
    ],
    prevent_initial_call=True,
)
def preprocess_job(
    job_submit_time: str,
    job_submit_state: str,
    model_data: dict,
    model_val: int,
    model_embeddings: dict,
    measurement_data: dict,
) -> float:
    """Submit job and provide job ID.

    Args:


    Returns:

    """

    if job_submit_state != JobState.PREPROCESSING.label:
        raise PreventUpdate

    #h = model_data["h"]
    #J = model_data["J"]

    job_kwargs = model_data.copy()
    current_job_results = measurement_data[ModelType(int(model_val)).name]
    measurement_number = len(current_job_results) + 1
    ta = ANNEALING_TIME_SERIES[(measurement_number - 1)%len(ANNEALING_TIME_SERIES)]/1000
    job_kwargs.update({"annealing_time": ta, "embeddings": model_embeddings})

    return job_kwargs, JobState.RUNNING.label

#======================================================================================================================

@dash.callback(
    Output("current-results", "data"),
    inputs=[
        Input("job-submit-args", "data"),
        State ("job-submit-state", "children"),
        State("qpu-selection", "value"),
        State("model-selection", "value"),
    ],
    prevent_initial_call=True,
)
def submit_job(
    job_submit_args: dict,
    job_submit_state: str,
    qpu_name: str,
    model_val: int,
) -> float:
    """Submit job and provide job ID.

    Args:


    Returns:

    """

    if job_submit_state != JobState.RUNNING.label:
        raise PreventUpdate

    print(f"Embeddings 1: {job_submit_args['embeddings']}")
    print()
    embeddings = job_submit_args.pop("embeddings")
    EGS = job_submit_args.pop("EGS")
    job_submit_args['h'] = {int(k): v for k, v in job_submit_args['h'].items()}
    job_submit_args['J'] = {tuple(map(int, k.strip("()").split(","))): v for k, v in job_submit_args['J'].items()}

    print(job_submit_args)
    embeddings = [{int(k): tuple([v]) for k, v in emb.items()} for emb in embeddings]
    print(f"Embeddings: {embeddings}")

    #solver = get_solvers()[qpu_name]
    qpu = DWaveSampler(solver=qpu_name)
    sampler = ParallelEmbeddingComposite(
            qpu,
            embeddings=embeddings,
        )

    qpu_kwargs = dict(
        num_reads=100,
        answer_mode="raw",
        fast_anneal=True,
        auto_scale=False,
        label="AQC Demo",
    )

    qpu_kwargs.update(job_submit_args)
    ss = sampler.sample_ising(**qpu_kwargs)
    stats = 1 - np.mean(ss.record.energy <= EGS + 1e-8)
    results_dict = Patch()
    results_dict[ModelType(model_val).name].append(stats)


    return results_dict

#==========================================================================================================

@dash.callback(
    Output("sample-v-theory-graph", "figure"),
    Output("job-submit-state", "children", allow_duplicate=True),
    Input("current-results", "data"),
    prevent_initial_call=True,
)
def update_main_graph(measurement_data: dict) -> go.Figure:
    fig = go.Figure()

    problem_type = measurement_data["problem_type"]
    data_sets = {k:v for k,v in measurement_data.items() if k != "problem_type"}

        # Add trace for each data se
    for label, stat_set in data_sets.items():
        anneal_times = [x[0] for x in stat_set]
        egs_probs = [x[1] for x in stat_set]
        fig.add_trace(go.Scatter(
            x=anneal_times,
            y=egs_probs,
            mode='lines+markers',
            marker=dict(symbol='x', size=8),
            name=label
        ))

    # Update layout with title, axis labels, and log scale
    fig.update_layout(
        title=f"Probabilty of Excited State vs Annealing Time as a Fucntion of Energy Gap for {problem_type}",
        xaxis_title="Annealing time (ns)",
        yaxis_title="Excited State Probability",
        yaxis_type="log",
        hovermode='x unified',
        template='plotly_white'
    )

    return fig, JobState.COMPLETED.label

#==========================================================================================================