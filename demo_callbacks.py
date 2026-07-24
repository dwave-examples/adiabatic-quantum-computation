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

import datetime
import json
from pathlib import Path
from typing import NamedTuple

import dash
import dimod
import numpy as np
import plotly.graph_objects as go
from dash import ALL, MATCH, Patch, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
from dwave.embedding import embed_bqm, is_valid_embedding
from dwave.system import DWaveSampler, ParallelEmbeddingComposite

from src.demo_enums import ProblemType

from src.qa import create_bqm, find_one_to_one_embedding, get_job_status, get_samples, json_to_dict
from src.qpu_resources import get_client, get_solvers

from src.get_embeddings import get_embeddings

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


@dash.callback(
    Output("schedule-filename", "children"),
    Output("schedule-filename", "className"),
    Input("qpu-selection", "value"),
)
def set_schedule(qpu_name: str) -> tuple[str, str]:
    """Set the schedule for the selected QPU.

    Args:
        qpu_name: The name of the selected QPU.

    Returns:
        A tuple containing:

        - str: The name of the schedule file to use for the selected QPU.
        - str: The class to apply to the schedule filename display.
    """

    #TODO: get schedule and simplify.

    schedule_filename = "FALLBACK_SCHEDULE.csv"
    schedule_filename_class = "no-schedule"

    if qpu_name:
        for file in SCHEDULES_EMBEDDINGS_PATH.glob("*_schedule.csv"):  # get schedule files
            file_solver_name = _schedule_solver_name(file)

            if file_solver_name and qpu_name == file_solver_name:
                schedule_filename = file.name
                schedule_filename_class = ""

    return schedule_filename, schedule_filename_class

#======================================================================================================================

@dash.callback(
    Output("current-embedding", "data"),
    Output("run-button", "disabled"),
    Output("qpu-select-time", "data"),
    Input("qpu-selection", "value"),
)
def process_qpu_select(qpu_name: str) -> tuple[dict, str]:
    """Removes stored value for current embedding and disables run button.
    This will trigger the set_current_embeddings callback to run. If an appropriate
    embedding is already cached, it will be used and the run button will be re-enabled
    almost immediately. If not, the run button will remain disabled until the embedding 
    is found and cached.
    """

    return {}, True

#======================================================================================================================

@dash.callback(
    Output("current-embeddings", "data"),
    Output("uncached-embeddings", "data"),
    Output("run-button", "disabled"),
    Input("qpu-select-time", "data"),
    Input("current-problem-type", "data"),
    State("cached-embeddings", "data"),
    State("qpu-selection", "value"),
)
def set_current_embeddings(
        qpu_select_time: float, 
        problem_type: str, 
        cached_embeddings: dict, 
        qpu_name: str
        ) -> tuple[dict, str]:
    """
    """

    qpu_id = None #TODO: set up schema to get qpu_id from qpu_name

    problem_embeddings, generated = get_embeddings(qpu_name, cached_embeddings, ProblemType[problem_type])
    embeddings_to_cache = {(qpu_id, problem_type): problem_embeddings} if generated else dash.no_update
    return problem_embeddings, embeddings_to_cache, False

#======================================================================================================================

@dash.callback(
    Output("cached-embeddings", "data"),
    Input("uncached-embeddings", "data"),
)
def cache_new_embeddings(uncached_embeddings: dict) -> dict:
    """Update the cached embeddings with any new embeddings found during the current run."""
    embedding_update = Patch()
    uncached_key, uncached_value = next(iter(uncached_embeddings.items()))
    embedding_update[uncached_key] = uncached_value
    return embedding_update

#======================================================================================================================

@dash.callback(
    Output("current-model", "data"),
    Output("run-button", "disabled"),
    Output("qpu-select-time", "data"),
    Input("qpu-selection", "value"),
)
def process_model_select(qpu_name: str) -> tuple[dict, str]:
    """Removes stored value for current embedding and disables run button.
    This will trigger the set_current_embeddings callback to run. If an appropriate
    embedding is already cached, it will be used and the run button will be re-enabled
    almost immediately. If not, the run button will remain disabled until the embedding 
    is found and cached.
    """

    return {}, True

#======================================================================================================================

@dash.callback(
    Output("current-model", "data"),
    Input("model-selection", "value"),
    State("current-problem-type", "data"), 
    State("cached-models", "data"),
)
def set_current_model(model_name: str, problem_type: str, cached_models: dict) -> tuple[dict, str]:
    return cached_models[(problem_type, model_name)]

#======================================================================================================================

@dash.callback(
    Output("job-id", "data"),
    inputs=[
        Input("job-submit-time", "data"), #TODO add necessary pieces to interface
        State("qpu-selection", "value"),
        State("current-model", "data"),
        State("current-embeddings", "data"),
        State("anneal-duration", "value"), #TODO set this
    ],
    prevent_initial_call=True,
)
def submit_job(
    job_submit_time: str,
    qpu_name: str,
    model_data: dict,
    model_embeddings: dict,
    ta: int,
) -> float:
    """Submit job and provide job ID.

    Args:


    Returns:

    """

    h = model_data["h"]
    J = model_data["J"]
    EGS = model_data["EGS"]
    # TODO handle anneal time

    embeddings = json_to_dict(embeddings)
    solver = get_solvers()[qpu_name] 
    sampler = ParallelEmbeddingComposite(
            solver, embedding={k: (v,) for k, v in embeddings[0].items()}
        )

    qpu_kwargs = dict(
        num_reads=100,
        answer_mode="raw",
        fast_anneal=True,
        auto_scale=False,
        label="AQC Demo",
    )

    qpu_kwargs["annealing_time"] = ta / 1000
    ss = sampler.sample_ising(h, J, **qpu_kwargs)
    stats = 1 - np.mean(ss.record.energy <= EGS + 1e-8)

    return stats

#==============================================================================================

class RunButtonClickReturn(NamedTuple):
    """Return type for the ``run_button_click`` callback function."""

    run_button_disabled: bool = True
    wd_job_disabled: bool = False
    wd_job_n_intervals: int = 0
    job_submit_state: str = "EVALUATING"


@dash.callback(
    Output("run-button", "disabled"),
    Output("wd-job", "disabled"),
    Output("wd-job", "n_intervals", allow_duplicate=True),
    Output("job-submit-state", "children"),
    inputs=[
        Input("run-button", "n_clicks"),

    ],
    prevent_initial_call=True,
)
def run_button_click(
    run_btn_click: int,
) -> RunButtonClickReturn:
    """Start simulation run when button is clicked.

    Args:

    Returns:
        A NamedTuple, RunButtonClickReturn, containing:

        - run_button_disabled: Whether the run button should be disabled.
        - wd_job_disabled: Whether the interval component should be disabled.
        - wd_job_n_intervals: The number of intervals that have passed since the simulation started.
        - job_submit_state: The new state of the job submission process.
    """
    #TODO: check to make sure this will trigger all necessary callbacks.
    return RunButtonClickReturn()

#==========================================================================================================

class SimulateReturn(NamedTuple):
    """Return type for the ``simulate`` callback function."""

    run_button_disabled: bool = dash.no_update
    wd_job_disabled: bool = dash.no_update
    wd_job_interval: int = dash.no_update
    wd_job_n_intervals: int = dash.no_update
    job_submit_state: str = dash.no_update
    job_submit_time: str = dash.no_update
    embeddings: dict = dash.no_update
    cached_embeddings: str = dash.no_update


@dash.callback(
    Output("run-button", "disabled", allow_duplicate=True),
    Output("wd-job", "disabled", allow_duplicate=True),
    Output("wd-job", "interval"),
    Output("wd-job", "n_intervals", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Output("job-submit-time", "data", allow_duplicate=True),
    Output("embeddings", "data", allow_duplicate=True),
    Output("cached-embeddings", "children", allow_duplicate=True),
    inputs=[
        Input("wd-job", "n_intervals"),
        State("job-id", "data"),
        State("job-submit-state", "children"),
        State("job-submit-time", "data"),
        State("spins", "value"),
        State("qpu-selection", "value"),
        State("embeddings", "data"),
    ],
    prevent_initial_call=True,
)
def simulate(
    interval: int,
    job_id: str,
    job_submit_state: str,
    job_submit_time: str,
    spins: str,
    qpu_name: str,
    embeddings: dict,
) -> SimulateReturn:
    """Manage simulation: embedding, job submission.

    Args:
        interval: The number of intervals that have passed since the simulation started.
        job_id: The ID of the job that was submitted.
        job_submit_state: The current state of the job submission process.
        job_submit_time: The time that the job was submitted.
        spins: The value of the spins setting.
        qpu_name: The name of the quantum processing unit (QPU) to which the job is being submitted.
        embeddings: A dictionary of cached embeddings for different numbers of spins.

    Returns:
        A NamedTuple, SimulateReturn, containing:

        - run_button_disabled: Whether the run button should be disabled.
        - wd_job_disabled: Whether the interval component should be disabled.
        - wd_job_interval: The number of milliseconds between interval updates.
        - wd_job_n_intervals: The number of intervals that have passed since the simulation started.
        - job_submit_state: The new state of the job submission process.
        - job_submit_time: The time that the job was submitted.
        - embeddings: The dictionary of cached embeddings for different numbers of spins.
        - cached_embeddings: A string representation of which embeddings are cached.
    """

    #TODO modify as necessary to work with the new code

    if job_submit_state == "EMBEDDING":
        try:
            spins = int(spins)
            embedding = find_one_to_one_embedding(spins, get_solvers()[qpu_name].edges)
            if embedding:
                embeddings = json_to_dict(embeddings)
                embeddings.update({spins: embedding})

                return SimulateReturn(
                    wd_job_interval=200,
                    job_submit_state="SUBMITTED",
                    job_submit_time=datetime.datetime.now().strftime("%c"),
                    embeddings=embeddings,
                    cached_embeddings=", ".join(str(em) for em in embeddings.keys()),
                )

            return SimulateReturn(
                run_button_disabled=False,
                wd_job_disabled=True,
                job_submit_state="FAILED",
            )
        except Exception:
            return SimulateReturn(
                run_button_disabled=False,
                wd_job_disabled=True,
                job_submit_state="FAILED",
            )

    if job_submit_state in ["SUBMITTED", "PENDING", "IN_PROGRESS"]:
        job_submit_state = get_job_status(get_client(), job_id, job_submit_time)
        wd_time = 1000

        if not job_submit_state:
            job_submit_state = "SUBMITTED"
            wd_time = 200

        return SimulateReturn(
            wd_job_interval=wd_time,
            wd_job_n_intervals=0,
            job_submit_state=job_submit_state,
        )

    return SimulateReturn(
        run_button_disabled=False,
        wd_job_disabled=True,
        job_submit_state=(
            dash.no_update if job_submit_state in ["COMPLETED", "CANCELLED", "FAILED"] else "ERROR"
        ),
    )