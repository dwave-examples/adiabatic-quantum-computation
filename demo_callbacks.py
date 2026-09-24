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

from __future__ import annotations

import time

import dash
import numpy as np
import plotly.graph_objects as go
from dash import MATCH, Input, Output, Patch, State
from dash.exceptions import PreventUpdate
from dwave.system import DWaveSampler, ParallelEmbeddingComposite

from demo_configs import ANNEALING_TIME_SERIES
from demo_interface import generate_new_results_dict
from src.demo_enums import JobState, ModelType, ProblemType
from src.model_loading import ProblemModel
from src.plotting import get_results_plot, get_spectrum_plot
from src.process_embeddings import EmbeddingSet
from src.qpu_resources import get_qpu_ids
from src.values import ENERGY_THRESHOLD


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


# =================================================================================================


@dash.callback(
    Output("current-embeddings", "data", allow_duplicate=True),
    Output("run-button", "disabled", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Output("qpu-select-time", "data"),
    Output("current-results", "data", allow_duplicate=True),
    Input("qpu-selection", "value"),
    prevent_initial_call=True,
)
def process_qpu_select(qpu_name: str) -> tuple[list[dict], bool, str, float, dict]:
    """Removes stored value for current embedding and disables run button.
    This will trigger the set_current_embeddings callback to run. If an appropriate
    embedding is already cached, it will be used and the run button will be re-enabled
    almost immediately. If not, the run button will remain disabled until the embedding
    is found and cached.

    Args:
        qpu_name: Name of the selected QPU

    Returns:
        A tuple containing:

        - list[dict]: Clears the ``current-embeddings`` store and triggers QPU-specific
            embeddings to be cached for the newly-selected QPU
        - bool: True, to disable the run button until new embeddings have been cached.
        - str: The FINDING_EMBEDDING label for the job-submit-state field.
        - float: The current time, stored in qpu-select-time to trigger set_current_embeddings.
        - dict: A new, empty results dictionary. Results are QPU-specific, so changing the QPU
        clears all the old results.
    """

    return [], True, JobState.FINDING_EMBEDDING.label, time.time(), generate_new_results_dict()


# =================================================================================================


@dash.callback(
    Output("current-embeddings", "data"),
    Output("available-solvers", "data"),
    Input("qpu-select-time", "data"),
    State("qpu-selection", "value"),
)
def set_current_embeddings(
    qpu_select_time: float,
    qpu_name: str,
) -> tuple[list[dict], list[str]]:
    """Finds and sets the current embeddings for the selected QPU and problem type. Checks
    embeddings saved in files first; if they are absent or unusable, finds new embeddings and
    saves them to file. Re-checks the list of available solvers to ensure selected QPU is
    still available: returns the current list in either event.

    Args:
        qpu_select_time: Value is irrelevant, used entirely as a callback trigger.
        qpu_name: The name of the selected QPU. Used to get the QPU graph to pass to the
            embedding function.

    Returns:
        A tuple containing:

        - list[dict]: The embeddings for the selected QPU, or an empty list if none could be found.
        - list[str]: The list of available solvers.
    """

    available_solvers = get_qpu_ids()

    if qpu_name not in available_solvers:
        return [], available_solvers

    try:
        sampler = DWaveSampler(solver=qpu_name)
        target_graph = sampler.to_networkx_graph()
    except Exception as exc:
        print(f"connecting to solver {qpu_name}: {exc}")
        return [], available_solvers

    try:
        emb_set = EmbeddingSet(qpu_name, target_graph)
        problem_embeddings = emb_set.get_embeddings()
    except Exception as exc:
        print(f"Error finding embeddings: {exc}")
        problem_embeddings = []

    return problem_embeddings, available_solvers


# =================================================================================================


@dash.callback(
    Output("run-button", "disabled", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Input("current-embeddings", "data"),
    prevent_initial_call=True,
)
def set_state_to_ready(
    current_embeddings: list[dict],
) -> tuple[bool, str]:
    """Resets the job state and enables the run button after embeddings are found.

    Args:
        current_embeddings: The embeddings for the selected QPU.

    Returns:
        A tuple containing:

        - bool: Whether the run button should be disabled.
        - str: The READY label for the job-submit-state field.
    """

    if not current_embeddings:
        raise PreventUpdate

    return False, JobState.READY.label


# =================================================================================================


@dash.callback(
    Output("current-model", "data"),
    Output("spectrum-graph", "figure"),
    Input("model-selection", "value"),
    State("current-problem-type", "data"),
    State("cached-models", "data"),
)
def set_current_model(
    model_val: str, problem_type: str, cached_models: dict
) -> tuple[dict, go.Figure]:
    """Sets the data and the spectrum plot for the currently selected model. Triggers whenever the
    model selection dropdown is changed. The spectrum plot is currently generated from a CSV file
    that is loaded from disk. The model data is retrieved from the cached models dictionary.

    Args:
        model_val: The value of the currently selected model from the dropdown.
        problem_type: The currently selected problem type.
        cached_models: A dictionary of cached models, keyed by their lookup keys.

    Returns:
        A tuple containing:

        - dict: The data for the currently selected model, including h and J values and its
          ground state energy (EGS).
        - go.Figure: A plotly figure of the spectrum of energies for the selected model.
    """
    spectrum_plot = get_spectrum_plot(ProblemType[problem_type], ModelType(int(model_val)))
    model_data = cached_models[
        ProblemModel(ProblemType[problem_type], ModelType(int(model_val))).lookup_key
    ]
    return model_data, spectrum_plot


# =================================================================================================


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
) -> tuple[bool, str, float]:
    """Start simulation run when button is clicked.

    Args:
        run_btn_click: The (total) number of times the run button has been clicked.

    Returns:
        A tuple containing:

        - bool: True, to disable the run button while the job runs.
        - str: The PREPROCESSING label for the job-submit-state field.
        - float: The current time, stored in job-submit-time to trigger preprocess_job.
    """

    return True, JobState.PREPROCESSING.label, time.time()


# =================================================================================================


@dash.callback(
    Output("job-submit-args", "data"),
    Output("job-submit-state", "children", allow_duplicate=True),
    inputs=[
        Input("job-submit-time", "data"),
        State("job-submit-state", "children"),
        State("current-model", "data"),
        State("model-selection", "value"),
        State("current-results", "data"),
    ],
    prevent_initial_call=True,
)
def preprocess_job(
    job_submit_time: float,
    job_submit_state: str,
    model_data: dict,
    model_val: str,
    measurement_data: dict,
) -> tuple[dict, str]:
    """Sets the annealing time and changes the job state to RUNNING.

    Args:
        job_submit_time: Not accessed, only used to trigger the callback.
        job_submit_state: Current job state. Used to check that the job is in the proper
            state before proceeding.
        model_data: Data dict for the current model. Callback makes a copy of this data and adds
            the annealing time to it before returning it as the job_kwargs dict that submit_job
            will use.
        model_val: Currently selected model. Used to retrieve the correct model data.
        measurement_data: Existing measurement data. Used to determine how many measurements
            have been made for this model to determine what the next annealing time should be.

    Returns:
        A tuple containing:

        - dict: The model data with the next annealing time (in microseconds) added to it.
        - str: The RUNNING label for the job-submit-state field.

    """

    if job_submit_state != JobState.PREPROCESSING.label:
        raise PreventUpdate

    job_kwargs = model_data.copy()
    current_job_results = measurement_data[ModelType(int(model_val)).name]
    measurement_number = len(current_job_results)
    ta = ANNEALING_TIME_SERIES[measurement_number % len(ANNEALING_TIME_SERIES)] / 1000
    job_kwargs.update({"annealing_time": ta})

    return job_kwargs, JobState.RUNNING.label


# =================================================================================================


@dash.callback(
    Output("current-results", "data"),
    Output("run-button", "disabled", allow_duplicate=True),
    Output("available-solvers", "data", allow_duplicate=True),
    inputs=[
        Input("job-submit-args", "data"),
        State("job-submit-state", "children"),
        State("qpu-selection", "value"),
        State("model-selection", "value"),
        State("current-embeddings", "data"),
    ],
    prevent_initial_call=True,
)
def submit_job(
    job_submit_args: dict,
    job_submit_state: str,
    qpu_name: str,
    model_val: str,
    model_embeddings: list[dict],
) -> tuple[Patch, bool, list[str]]:
    """Submit job and provide job ID. Returns the updated results to be stored
    in the current-results data store. Re-enables the run button (disabled by run_button_click)
    after the job is complete. Re-checks the list of available solvers to ensure the selected QPU
    is still available, returning the current list in either event.

    Args:
        job_submit_args: A dictionary containing the job submission arguments including the
            problem Hamiltonian and the annealing time.
        job_submit_state: current job state. Used to ensure that the demo is in a state
            compatible with job submission.
        qpu_name: The name of the QPU to which the job is being submitted.
        model_val: The currently selected model value as a string.
        model_embeddings: A list of embeddings for the model, each a dict mapping source node to
            a chain of target qubits. There should be multiple parallel embeddings for the model,
            as determined by the NUM_PARALLEL_EMBEDDINGS constant in demo_configs.py.

    Returns:
        A tuple containing:

        - Patch: The updated results dictionary with the new measurement appended. If the selected
            solver was unavailable, this dictionary will be returned unchanged.
        - bool: False indicating that the run button should not be disabled.
        - list: The list of available solvers.
    """

    results_dict = Patch()

    available_solvers = get_qpu_ids()
    if qpu_name not in available_solvers:
        return results_dict, False, available_solvers

    if job_submit_state != JobState.RUNNING.label:
        raise PreventUpdate

    EGS = job_submit_args.pop("EGS")
    job_submit_args["h"] = {int(k): v for k, v in job_submit_args["h"].items()}
    job_submit_args["J"] = {
        tuple(map(int, k.strip("()").split(","))): v for k, v in job_submit_args["J"].items()
    }

    embeddings = [{int(k): tuple(v) for k, v in emb.items()} for emb in model_embeddings]

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
        label="Example - Adiabatic Quantum Computation",
    )

    qpu_kwargs.update(job_submit_args)
    ss = sampler.sample_ising(**qpu_kwargs)
    stats = 1 - np.mean(ss.record.energy <= EGS + ENERGY_THRESHOLD)  # Excited state probability
    results_dict[ModelType(int(model_val)).name].append(stats)

    return results_dict, False, available_solvers


# =================================================================================================


@dash.callback(
    Output("sample-v-theory-graph", "figure"),
    Output("job-submit-state", "children"),
    Input("current-results", "data"),
    State("qpu-selection", "value"),
)
def update_main_graph(measurement_data: dict, qpu_name: str) -> tuple[go.Figure, str]:
    """Updates the main graph showing the excited state probability as a function of annealing
    time. This callback triggers whenever a new measurement is added to the current-results
    data store: it updates the graph with the new data point and returns the updated figure.
    It also updates the job submission state to COMPLETED.

    Args:
        measurement_data: A dictionary containing the current measurement results.
        qpu_name: The name of the currently selected QPU, used for rescaling the
            results plot to the QPU-specific energy scale.

    Returns:
        A tuple containing:

        - go.Figure: A plotly figure of the excited state probability as a function of the
            annealing time.
        - str: The new state of the job submission process, set to COMPLETED.
    """

    fig = get_results_plot(measurement_data, ProblemType.DEFAULT, qpu_name)

    return fig, JobState.COMPLETED.label


# ==================================================================================================


@dash.callback(
    Output("no-solver-modal", "opened"),
    Input("available-solvers", "data"),
)
def alert_no_solver(available_solvers: dict) -> bool:
    """Show the no-solver modal on load if no quantum computer is accessible.

    Args:
        available_solvers: The QPUs accessible, empty if none.

    Returns:
        True if the modal should be opened, False otherwise.
    """
    return not available_solvers
