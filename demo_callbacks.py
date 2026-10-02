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
import bisect

import dash
import numpy as np
import plotly.graph_objects as go
from dash import MATCH, Input, Output, Patch, State, ctx
from dash.exceptions import PreventUpdate
from dwave.system import DWaveSampler, ParallelEmbeddingComposite
import dimod

from demo_configs import DEFAULT_ANNEALING_TIMES, NUM_READS, NUM_PARALLEL_EMBEDDINGS, ANNEAL_TIME_STATIC_MARKS, MODEL_COLORS_RESULTS
from demo_interface import generate_new_results_dict
from src.demo_enums import JobState, ModelType
from src.model_loading import ProblemModel
from src.plotting import get_results_plot, get_spectrum_plot
from src.process_embeddings import EmbeddingSet
from src.qpu_resources import get_qpu_ids
from src.values import ENERGY_THRESHOLD
from src.sample_stats import process_results


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
    Output("qpu-select-time", "data"),
    Input("qpu-selection", "value"),
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

    return time.time()

# =================================================================================================


@dash.callback(
    Output("current-embeddings", "data", allow_duplicate=True),
    Output("run-button", "disabled", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Output("current-results", "data", allow_duplicate=True),
    Input("qpu-select-time", "data"),
    prevent_initial_call=True,
)
def reset_embeddings_and_results(qpu_name: str) -> tuple[list[dict], bool, str, float, dict]:
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

    return [], True, JobState.FINDING_EMBEDDING.label, generate_new_results_dict()

@dash.callback(
    Output("current-embeddings", "data"),
    Output("available-solvers", "data"),
    Output("run-button", "disabled", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Input("current-results", "data"),
    State("job-submit-state", "children"),
    State("qpu-selection", "value"),
    prevent_initial_call=True
)
def set_current_embeddings(
    results_dict: dict,
    job_submit_state: str,
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

    if job_submit_state != JobState.FINDING_EMBEDDING.label:
        raise PreventUpdate

    available_solvers = get_qpu_ids()

    if not available_solvers:
        print("No available solvers")
        return [], available_solvers, True, JobState.NO_SOLVER.label

    if qpu_name not in available_solvers:
        print(f"Selected QPU {qpu_name} is not available")
        return [], available_solvers, True, JobState.CONNECTING.label # TODO consider job state label for this case

    try:
        sampler = DWaveSampler(solver=qpu_name)
        target_graph = sampler.to_networkx_graph()
    except Exception as exc:
        print(f"Error connecting to solver {qpu_name}: {exc}")
        return [], available_solvers, False, JobState.FAILED.label

    try:
        emb_set = EmbeddingSet(qpu_name, target_graph)
        problem_embeddings = emb_set.get_embeddings()
    except Exception as exc:
        print(f"Error finding embeddings: {exc}")
        problem_embeddings = []

    return problem_embeddings, available_solvers, False, JobState.READY.label


# =================================================================================================


@dash.callback(
    Output("current-model", "data"),
    Output("spectrum-graph", "figure"),
    Input("model-selection", "value"),
    Input("energy-rescaling", "value"),
    Input("job-submit-time", "data"),
    State("cached-models", "data"),
)
def set_current_model(model_val: str, energy_rescaling: float, job_submit_time: float, cached_models: dict) -> tuple[dict, go.Figure]:
    """Sets the data and the spectrum plot for the currently selected model. Triggers whenever the
    model selection dropdown is changed. The spectrum plot is currently generated from a CSV file
    that is loaded from disk. The model data is retrieved from the cached models dictionary.

    Args:
        model_val: The value of the currently selected model from the dropdown.
        cached_models: A dictionary of cached models, keyed by their lookup keys.

    Returns:
        A tuple containing:

        - dict: The data for the currently selected model, including h and J values and its
          ground state energy (EGS).
        - go.Figure: A plotly figure of the spectrum of energies for the selected model.
    """
    spectrum_plot = get_spectrum_plot(ModelType(int(model_val)), energy_rescaling=energy_rescaling)
    if ctx.triggered_id == "energy-rescaling":
        model_data = dash.no_update
    else:
        model_data = cached_models[ProblemModel(ModelType(int(model_val))).lookup_key]
    return model_data, spectrum_plot

# =================================================================================================

@dash.callback(
    Output("run-button", "disabled", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Output("job-submit-time", "data"),
    Output("energy-rescaling", "disabled"),
    Input("run-button", "n_clicks"),
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

    return True, JobState.PREPROCESSING.label, time.time(), True


# =================================================================================================


@dash.callback(
    Output("job-submit-args", "data"),
    Output("job-submit-state", "children", allow_duplicate=True),
    inputs=[
        Input("current-model", "data"),
        State("job-submit-state", "children"),
        State("current-model", "data"),
        State("model-selection", "value"),
        State("annealing-time", "value"),
        State("energy-rescaling", "value"),
    ],
    prevent_initial_call=True,
)
def preprocess_job(
    job_submit_time: float,
    job_submit_state: str,
    model_data: dict,
    model_val: str,
    annealing_time: float,
    energy_rescaling: float,
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

    try:
        job_kwargs = model_data.copy()
        print(job_kwargs)
        ta = annealing_time / 1000
        job_kwargs.update({"annealing_time": ta})
        print(f"Energy Rescaling: {energy_rescaling}")
        print(f"Job kwargs before energy rescaling: {job_kwargs}")
        job_kwargs['h'] = {k: v*energy_rescaling for k, v in job_kwargs['h'].items()}
        job_kwargs['J'] = {k: v*energy_rescaling for k, v in job_kwargs['J'].items()}
        job_kwargs['EGS'] *= energy_rescaling
        print(f"Job kwargs after energy rescaling: {job_kwargs}")
    except:
        return dash.no_update, JobState.FAILED.label

    return job_kwargs, JobState.RUNNING.label

# =================================================================================================


@dash.callback(
    Output("current-results", "data"),
    Output("run-button", "disabled", allow_duplicate=True),
    Output("available-solvers", "data", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Output("energy-rescaling", "disabled", allow_duplicate=True),
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
        return results_dict, False, available_solvers, JobState.FAILED.label, False

    if job_submit_state != JobState.RUNNING.label:
        raise PreventUpdate


    try:
        EGS = job_submit_args["EGS"]
        h = {int(k): v for k, v in job_submit_args["h"].items()}
        J = {tuple(map(int, k.strip("()").split(","))): v for k, v in job_submit_args["J"].items()}
        bqm = dimod.BinaryQuadraticModel(dimod.SPIN).from_ising(h=h, J=J)
        parallel_bqms = [bqm]*NUM_PARALLEL_EMBEDDINGS

        ta = job_submit_args['annealing_time']
        embeddings = [{int(k): tuple(v) for k, v in emb.items()} for emb in model_embeddings]

        qpu = DWaveSampler(solver=qpu_name)

        sampler = ParallelEmbeddingComposite(
            qpu,
            embeddings=embeddings,
        )

        qpu_kwargs = dict(
            num_reads=NUM_READS,
            answer_mode="raw",
            fast_anneal=True,
            auto_scale=False,
            label="Example - Adiabatic Quantum Computation",
            annealing_time=ta
        )

        sample_sets, _ = sampler.sample_multiple(**qpu_kwargs, bqms = parallel_bqms)
        stats = []
        for s_set in sample_sets:
            set_df = s_set.to_pandas_dataframe()
            set_energies = set_df["energy"].to_numpy()
            stats.append(1 - np.mean(set_energies <= EGS + ENERGY_THRESHOLD))  # Excited state probability
        results_dict[ModelType(int(model_val)).name].update({ta*1000: stats})
    except:
        return results_dict, False, available_solvers, JobState.FAILED.label, False

    return results_dict, False, available_solvers, JobState.READY.label, False


# =================================================================================================


@dash.callback(
    Output("sample-v-theory-graph", "figure"),
    Output("annealing-time", "value"),
    Output("annealing-time", "marks"),
    Input("current-results", "data"),
    Input("model-selection", "value"),
    Input("energy-rescaling", "value"),
    State("qpu-selection", "value"),
)
def update_main_graph(measurement_data: dict, model_val: str, energy_rescaling: float, qpu_name: str) -> tuple[go.Figure, str, float]:
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

    static_marks = [{"value": ta, "label": f"{ta}"} for ta in ANNEAL_TIME_STATIC_MARKS]

    if ctx.triggered_id == "model-selection":
        return dash.no_update, DEFAULT_ANNEALING_TIMES[0], static_marks

    fig = get_results_plot(measurement_data, qpu_name)

    results_dict = measurement_data[ModelType(int(model_val)).name]

    if results_dict:
        sample_annealing_times = list(map(float, results_dict.keys()))
        largest_ta = max(sample_annealing_times) if results_dict else 0
        static_marks = [
            {"value": ta, "label": f"{ta}"} 
            for ta in ANNEAL_TIME_STATIC_MARKS 
            if ta not in sample_annealing_times
            ]
        sample_color = MODEL_COLORS_RESULTS[ModelType(int(model_val))]
        sample_marks = [{"value": ta, "label": f"{ta}", "style": {"color":"#1cf21c"}} for ta in sample_annealing_times]
        annealing_time_marks = static_marks + sample_marks
    else:
        largest_ta = 0
        annealing_time_marks = static_marks

    next_ta_index = bisect.bisect_right(DEFAULT_ANNEALING_TIMES, largest_ta)

    next_ta = DEFAULT_ANNEALING_TIMES[min(next_ta_index, len(DEFAULT_ANNEALING_TIMES) - 1)]

    return fig, next_ta, annealing_time_marks


# ==================================================================================================


@dash.callback(
    Output("no-solver-modal", "opened"),
    Input("available-solvers", "data"),
)
def alert_no_solver(available_solvers: list) -> bool:
    """Show the no-solver modal on load if no quantum computer is accessible.

    Args:
        available_solvers: The QPUs accessible, empty if none.

    Returns:
        True if the modal should be opened, False otherwise.
    """
    return not available_solvers
