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

import bisect
import time

import dash
import dimod
import numpy as np
import plotly.graph_objects as go
from dash import MATCH, Input, Output, Patch, State, ctx, html
from dash.exceptions import PreventUpdate
from dwave.system import DWaveSampler, ParallelEmbeddingComposite

from demo_configs import NUM_READS, SUGGESTED_ANNEALING_TIMES
from demo_interface import generate_new_results_dict
from src.demo_enums import JobState, ModelType, TheoryCurveMode
from src.model_loading import ProblemModel
from src.plotting import generate_plots
from src.process_embeddings import EmbeddingSet
from src.qpu_resources import get_qpu_ids, get_schedule_filepath
from src.values import ENERGY_THRESHOLD, MIN_ANNEALING_TIME


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
    Output("theory-curve", "value"),
    Output("theory-curve", "disabled"),
    Input("qpu-selection", "value"),
)
def process_qpu_select(qpu_name: str) -> float:
    """Triggers when the selected QPU is altered, writing the selection time to the
    relevant store. Also resets the Theory Curve selection to the default, and
    disables it if no schedule is available for the selected QPU.

    Args:
        qpu_name: Name of the selected QPU

    Returns:
        A tuple containing:

        -The current time, stored in qpu-select-time to trigger reset_embeddings_and_results.
        -The value of the Theory Curve mode that should be pre-selected (always LANDAU_ZENER)
        -Whether or not to disable the theory curve selection, based on whether the selected
            QPU has a schedule available.
    """

    schedule_check = get_schedule_filepath(qpu_name)

    return time.time(), str(TheoryCurveMode.LANDAU_ZENER.value), (schedule_check is None)


# =================================================================================================


@dash.callback(
    Output("current-embeddings", "data", allow_duplicate=True),
    Output("run-button", "disabled", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Output("current-results", "data", allow_duplicate=True),
    Input("qpu-select-time", "data"),
    prevent_initial_call=True,
)
def reset_embeddings_and_results(qpu_select_time: float) -> tuple[list, bool, str, dict]:
    """Removes stored values for current embedding and result.

    Args:
        qpu_select_time: The time at which the QPU was selected, used as a callback trigger.

    Returns:
        A tuple containing:

        - list[dict]: Clears the ``current-embeddings`` store and triggers QPU-specific
            embeddings to be cached for the newly-selected QPU
        - bool: True, to disable the run button until new embeddings have been cached.
        - str: The FINDING_EMBEDDING label for the job-submit-state field.
        - dict: A new, empty results dictionary. Results are QPU-specific, so changing the QPU
        clears all the old results.
    """

    return [], True, JobState.FINDING_EMBEDDING.label, generate_new_results_dict()


# =================================================================================================


@dash.callback(
    Output("current-results", "data", allow_duplicate=True),
    Input("energy-rescaling", "value"),
    prevent_initial_call=True,
)
def reset_results_on_energy_rescaling(energy_rescaling: float) -> dict:
    """Clears stored results when the energy rescaling factor changes.

    Previously measured results were taken at the old energy scale, so they are not
    comparable to results taken at the new one.

    Args:
        energy_rescaling: The newly selected energy rescaling factor.

    Returns:
        A new, empty results dictionary.
    """

    return generate_new_results_dict()


@dash.callback(
    Output("current-embeddings", "data"),
    Output("available-solvers", "data"),
    Output("run-button", "disabled", allow_duplicate=True),
    Output("job-submit-state", "children", allow_duplicate=True),
    Input("current-results", "data"),
    State("job-submit-state", "children"),
    State("qpu-selection", "value"),
    prevent_initial_call=True,
)
def set_current_embeddings(
    results_dict: dict,
    job_submit_state: str,
    qpu_name: str,
) -> tuple[list[dict], list[str], bool, str]:
    """Finds and sets the current embeddings for the selected QPU and problem type. Checks
    embeddings saved in files first; if they are absent or unusable, finds new embeddings and
    saves them to file. Re-checks the list of available solvers to ensure selected QPU is
    still available: returns the current list in either event.

    Args:
        results_dict: The current results dictionary, used solely as a trigger.
        job_submit_state: The current state of the job submission process, used to determine
            whether embeddings should be set or updated.
        qpu_name: The name of the selected QPU. Used to get the QPU graph to pass to the
            embedding function.

    Returns:
        A tuple containing:

        - list[dict]: The embeddings for the selected QPU, or an empty list if none could be found.
        - list[str]: The list of available solvers.
        - bool: Returns False to re-enable the run button once embeddings have been set. If there is
            an error finding embeddings, returns True, keeping the run button disabled.
        - str: The current job state label, which will be 'READY' if embeddings are successfully set,
            or an appropriate error state otherwise.
    """

    if job_submit_state != JobState.FINDING_EMBEDDING.label:
        raise PreventUpdate

    available_solvers = get_qpu_ids()

    if not available_solvers:
        print("No available solvers")
        return [], available_solvers, True, JobState.NO_SOLVER.label

    if qpu_name not in available_solvers:
        print(f"Selected QPU {qpu_name} is not available")
        return (
            [],
            available_solvers,
            True,
            JobState.FAILED.label,
        )

    try:
        sampler = DWaveSampler(solver=qpu_name)
        target_graph = sampler.to_networkx_graph()
    except Exception as exc:
        print(f"Error connecting to solver {qpu_name}: {exc}")
        return [], available_solvers, True, JobState.FAILED.label

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
    Input("model-selection", "value"),
    Input("job-submit-time", "data"),
    State("cached-models", "data"),
)
def set_current_model(
    model_val: str, job_submit_time: float, cached_models: dict
) -> tuple[dict, go.Figure]:
    """Sets the data and the spectrum plot for the currently selected model. Triggers whenever the
    model selection dropdown is changed or the energy rescaling value is modified. Also triggers
    on job submission to ensure that a current model is set: there were issues especially around
    startup with the model not getting set properly otherwise.

    Args:
        model_val: The value of the currently selected model from the dropdown.
        energy_rescaling: The current value of the energy rescaling factor.
        job_submit_time: Job submission time: value is not used, just used as a callback trigger.
        cached_models: A dictionary of cached models, keyed by their lookup keys.

    Returns:
        A tuple containing:

        - dict: The data for the currently selected model, including h and J values and its
          ground state energy (EGS).
        - go.Figure: A plotly figure of the spectrum of energies for the selected model.
    """

    model_data = cached_models[ProblemModel(ModelType(int(model_val))).lookup_key]
    return model_data


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
) -> tuple[bool, str, float, bool]:
    """Start simulation run when button is clicked.

    Args:
        run_btn_click: The (total) number of times the run button has been clicked.

    Returns:
        A tuple containing:

        - bool: True, to disable the run button while the job runs.
        - str: The PREPROCESSING label for the job-submit-state field.
        - float: The current time, stored in job-submit-time to trigger preprocess_job.
        - bool: True to disable the energy rescaling input while the job runs.
    """

    return True, JobState.PREPROCESSING.label, time.time(), True


# =================================================================================================


@dash.callback(
    Output("job-submit-args", "data"),
    Output("job-submit-state", "children", allow_duplicate=True),
    inputs=[
        Input("current-model", "data"),
        State("job-submit-state", "children"),
        State("annealing-time", "value"),
        State("energy-rescaling", "value"),
    ],
    prevent_initial_call=True,
)
def preprocess_job(
    model_data: dict,
    job_submit_state: str,
    annealing_time: float,
    energy_rescaling: float,
) -> tuple[dict, str]:
    """Sets the annealing time and changes the job state to RUNNING.

    Args:
        model_data: Data dict for the current model. Callback makes a copy of this data and adds
            the annealing time to it before returning it as the job_kwargs dict that submit_job
            will use.
        job_submit_state: Current job state. Used to check that the job is in the proper
            state before proceeding.
        annealing_time: The selected annealing time for the simulation, in nanoseconds.
        energy_rescaling: The factor by which to rescale the energy terms in the model data.

    Returns:
        A tuple containing:

        - dict: The model data with the next annealing time (in microseconds) added to it.
        - str: The RUNNING label for the job-submit-state field.

    """

    if job_submit_state != JobState.PREPROCESSING.label:
        raise PreventUpdate

    try:
        job_kwargs = model_data.copy()
        ta = annealing_time / 1000
        job_kwargs.update({"annealing_time": ta})
        job_kwargs["h"] = {k: v * energy_rescaling for k, v in job_kwargs["h"].items()}
        job_kwargs["J"] = {k: v * energy_rescaling for k, v in job_kwargs["J"].items()}
        job_kwargs["EGS"] *= energy_rescaling
    except Exception as exc:
        print(f"Error preparing job: {exc}")
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
) -> tuple[Patch, bool, list[str], str, bool]:
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
        - str: The final job state label, indicating whether the job succeeded or failed.
        - bool: False indicating that the energy rescaling slider should not be disabled.
    """

    if job_submit_state != JobState.RUNNING.label:
        raise PreventUpdate

    results_dict = Patch()

    available_solvers = get_qpu_ids()
    if qpu_name not in available_solvers:
        return results_dict, False, available_solvers, JobState.FAILED.label, False

    try:
        EGS = job_submit_args["EGS"]
        h = {int(k): v for k, v in job_submit_args["h"].items()}
        J = {tuple(map(int, k.strip("()").split(","))): v for k, v in job_submit_args["J"].items()}
        bqm = dimod.BinaryQuadraticModel(dimod.SPIN).from_ising(h=h, J=J)
        parallel_bqms = [bqm] * len(model_embeddings)

        ta = job_submit_args["annealing_time"]
        embeddings = [{int(k): tuple(v) for k, v in emb.items()} for emb in model_embeddings]

        qpu = DWaveSampler(solver=qpu_name)
        sampler = ParallelEmbeddingComposite(qpu, embeddings=embeddings)

        qpu_kwargs = dict(
            num_reads=NUM_READS,
            answer_mode="raw",
            fast_anneal=True,
            auto_scale=False,
            label="Example - Adiabatic Quantum Computation",
            annealing_time=ta,
        )

        sample_sets, _ = sampler.sample_multiple(**qpu_kwargs, bqms=parallel_bqms)
        stats = [  # Calculate and record excited state probabilities for each sample set
            float(1 - np.mean(s_set.record.energy <= EGS + ENERGY_THRESHOLD))
            for s_set in sample_sets
        ]
        results_dict[ModelType(int(model_val)).name].update({ta * 1000: stats})
    except Exception as exc:
        print(f"Error submitting job to {qpu_name}: {exc}")
        return results_dict, False, available_solvers, JobState.FAILED.label, False

    return results_dict, False, available_solvers, JobState.READY.label, False


# =================================================================================================


@dash.callback(
    Output("sample-v-theory-graph", "figure"),
    Output("spectrum-graph", "figure"),
    Output("spectrum-title", "children"),
    Output("annealing-time", "value"),
    Input("current-results", "data"),
    Input("model-selection", "value"),
    Input("energy-rescaling", "value"),
    Input("qpu-selection", "value"),
    Input("theory-curve", "value"),
    State("annealing-time-suggest", "checked"),
)
def update_graphs(
    sample_data: dict,
    model_val: str,
    energy_rescaling: float,
    qpu_name: str,
    theory_curve_mode: str,
    auto_suggest: bool,
) -> tuple[go.Figure, go.Figure, list, float]:
    """Both the results graph and the spectrum plot. This callback triggers whenever a new
    measurement is added to the current-results data store: it updates the graph with the new
    data point and returns the updated figure. It will also trigger if the model, energy
    rescaling, or theory curve mode changes, making the appropriate updates to the spectrum
    and results plot, but leaving plotted data unchanged. Finally, this callback will also
    trigger when a new QPU is selected, clearing all data plotted for the previous QPU.

    Args:
        sample_data: A dictionary containing the current measurement results.
        model_val: The currently selected model value.
        energy_rescaling: The current energy rescaling factor.
        qpu_name: The name of the currently selected QPU, used for rescaling the
            results plot to the QPU-specific energy scale.
        theory_curve_mode: The mode determining which sort of theory curve is displayed on
            the results plot.
        auto_suggest: Whether the auto-suggest option for the annealing time slider is enabled.

    Returns:
        A tuple containing:

        - go.Figure: A plotly figure of the excited state probability as a function of the
            annealing time.
        - go.Figure: A plot showing the spectrum of excited state energy values as a function
            of anneal progress.
        - list: The spectrum plot title, reflecting the currently selected model.
        - float: The next suggested annealing time. Will reset to the minimum value when
            the model or QPU is changed. If the auto-suggest option is enabled, it will
            set the slider to the next time in sequence. Otherwise it will leave the slider
            value where it is.
    """

    results = {}
    largest_ta = 0.0

    current_model = ModelType(int(model_val)).name
    for model in ModelType:
        results[model.name] = {float(ta): stats for ta, stats in sample_data[model.name].items()}
        if results[model.name] and model.name == current_model:
            largest_ta = max(results[model.name])

    results_plot, spectrum_plot = generate_plots(
        job_results=results,
        qpu_name=qpu_name,
        selected_model=ModelType(int(model_val)),
        theory_mode=TheoryCurveMode(int(theory_curve_mode)),
        energy_rescaling=energy_rescaling,
    )

    if auto_suggest:
        next_ta_index = bisect.bisect_right(SUGGESTED_ANNEALING_TIMES, largest_ta)
        next_ta = SUGGESTED_ANNEALING_TIMES[min(next_ta_index, len(SUGGESTED_ANNEALING_TIMES) - 1)]
    elif largest_ta < MIN_ANNEALING_TIME:
        next_ta = MIN_ANNEALING_TIME
    else:
        next_ta = dash.no_update

    spectrum_title = [
        f"{ModelType(int(model_val)).label} spectrum",
        html.Br(),
        "(positive parity sector)",
    ]

    return results_plot, spectrum_plot, spectrum_title, next_ta


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
