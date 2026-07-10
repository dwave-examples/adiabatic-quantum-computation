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

import dash
from dash import MATCH
from dash.dependencies import Input, Output, State

from demo_interface import generate_table
from src.demo_enums import SolverType


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
    Output("input", "children"),
    inputs=[
        Input("slider", "value"),
    ],
)
def render_initial_state(slider_value: int) -> str:
    """Runs on load and any time the value of the slider is updated.
        Add `prevent_initial_call=True` to skip on load runs.

    Args:
        slider_value: The value of the slider.

    Returns:
        The content of the input tab.
    """
    return f"Put demo visuals here. The current slider value is {slider_value}."
#======================================================================================================================

@dash.callback(
    Output("sample-v-theory-graph", "figure", allow_duplicate=True),
    Output("kz-data", "data", allow_duplicate=True),
    inputs=[
        Input("job-submit-state", "children"),
        State("graph-selection-radio", "value"),
        State("coupling-strength", "value"),  # previously input
        State("job-id", "data"),
        State("anneal-duration", "value"),
        State("spins", "value"),
        State("selected-problem-type", "data"),
        State("embeddings", "data"),
        State("sample-v-theory-graph", "figure"),
        State("kz-data", "data"),  # get kibble zurek data point
    ],
    prevent_initial_call=True,
)
def add_graph_point_kz(
    job_submit_state: str,
    graph_selection: str,
    J: float,
    job_id: str,
    ta: int | str,
    spins: str,
    problem_type: str,
    embeddings: dict,
    figure: go.Figure,
    kz_data: list,
) -> tuple[go.Figure, list]:
    """Add new point to kink density graph when KZ job finishes.

    Args:
        job_submit_state: The current state of the job submission process.
        graph_selection: The value of the graph selection radio, either "schedule", "kink_density",
            or "both".
        J: The value of the coupling strength setting.
        job_id: The ID of the submitted job.
        ta: The value of the anneal duration setting, in nanoseconds.
        spins: The value of the spins setting.
        problem_type: Either KZ (``0`` or ``ProblemType.KZ``) or
            KZ_NM (``1`` or ``ProblemType.KZ_NM``).
        embeddings: A dictionary of cached embeddings for different numbers of spins.
        figure: The current figure for the sample vs theory graph.
        kz_data: The existing data points for the kink density graph.

    Returns:
        A tuple containing:

        - go.Figure: The updated figure for the sample vs theory graph, with a new point added.
        - list: Data points for the kink density graph, either unchanged or with a new point added.
    """
    if job_submit_state != "COMPLETED" or problem_type is ProblemType.KZ_NM.value:
        raise PreventUpdate

    spins = int(spins)
    ta = int(ta.split(" ")[0]) if isinstance(ta, str) else ta
    embeddings = json_to_dict(embeddings)
    sampleset_unembedded = get_samples(get_client(), job_id, spins, J, embeddings[spins])
    _, kink_density = kink_stats(sampleset_unembedded, J)

    # Append the new data point
    kz_data.append((kink_density, ta))

    fig = (
        dash.no_update
        if graph_selection == "schedule"
        else plot_kink_density(
            graph_selection, figure, kink_density, ta, J, problem_type=ProblemType.KZ
        )
    )
    return fig, kz_data

#======================================================================================================================

@dash.callback(
    Output("kink-v-noise-graph", "figure", allow_duplicate=True),
    Output("kink-v-anneal-graph", "figure", allow_duplicate=True),
    Output("coupling-data", "data", allow_duplicate=True),  # store data using dcc
    Output("zne-estimates", "data", allow_duplicate=True),  # update zne_estimates
    Output("modal-trigger", "data"),
    inputs=[
        Input("job-submit-state", "children"),
        State("qpu-selection", "value"),
        State("coupling-strength", "value"),  # previously input
        State("schedule-filename", "children"),
        State("job-id", "data"),
        State("anneal-duration", "value"),
        State("spins", "value"),
        State("selected-problem-type", "data"),
        State("embeddings", "data"),
        State("kink-v-noise-graph", "figure"),
        State("kink-v-anneal-graph", "figure"),
        State("coupling-data", "data"),  # access previously stored data
        State("zne-estimates", "data"),  # Access ZNE estimates
    ],
    prevent_initial_call=True,
)
def add_graph_point_kz_nm(
    job_submit_state: str,
    qpu_name: str,
    J: float,
    schedule_filename: str,
    job_id: str,
    ta: int | str,
    spins: str,
    problem_type: str,
    embeddings: dict,
    figure_noise: go.Figure,
    figure_anneal: go.Figure,
    coupling_data: dict,
    zne_estimates: dict,
) -> tuple[go.Figure, go.Figure, dict, dict, bool]:
    """Add new point to Noise Ratio and Annealing Duration graphs when KZ Noise Mitigation job
    finishes.

    Args:
        job_submit_state: The current state of the job submission process.
        qpu_name: The name of the selected QPU.
        J: The value of the coupling strength setting.
        schedule_filename: The name of the quench schedule file being used.
        job_id: The ID of the submitted job.
        ta: The value of the anneal duration setting, in nanoseconds.
        spins: The value of the spins setting.
        problem_type: Either KZ (``0`` or ``ProblemType.KZ``) or
            KZ_NM (``1`` or ``ProblemType.KZ_NM``).
        embeddings: A dictionary of cached embeddings for different numbers of spins.
        figure_noise: The current figure for the kink density vs noise graph.
        figure_anneal: The current figure for the kink density vs anneal duration graph.
        coupling_data: The existing data points for the coupling strength vs kink density graph.
        zne_estimates: The existing zero noise extrapolation estimates.

    Returns:
        A tuple containing:

        - go.Figure: The figure for the kink density vs noise graph, with a new point added.
        - go.Figure: The figure for the kink density vs anneal duration graph, with a new point added.
        - dict: Data points for the coupling strength vs kink density graph, with a new point added.
        - dict: The updated zero noise extrapolation estimates, with a new estimate added.
        - bool: Whether the ZNE modal should be open, based on whether a new ZNE estimate was added.
    """
    if job_submit_state != "COMPLETED" or problem_type is ProblemType.KZ.value:
        raise PreventUpdate

    spins = int(spins)
    ta = int(ta.split(" ")[0]) if isinstance(ta, str) else ta
    embeddings = json_to_dict(embeddings)
    sampleset_unembedded = get_samples(get_client(), job_id, spins, J, embeddings[spins])
    _, kink_density = kink_stats(sampleset_unembedded, J)

    # Calculate lambda (previously kappa)
    # Added _ to avoid keyword restriction
    lambda_ = calclambda_(J=J, schedule_name=schedule_filename)

    fig_noise = plot_kink_density("coupling", figure_noise, kink_density, ta, J, lambda_)
    fig_anneal = plot_kink_density("kink_density", figure_anneal, kink_density, ta, J, lambda_)

    # Initialize the list for this anneal_time if not present
    ta_str = str(ta)
    if ta_str not in coupling_data:
        coupling_data[ta_str] = []

    # Append the new data point
    coupling_data[ta_str].append(
        {
            "lambda": lambda_,
            "kink_density": kink_density,
            "coupling-strength": J,
        }
    )

    zne_estimates, modal_trigger = plot_zne_fitted_line(
        fig_noise, coupling_data, zne_estimates, ta_str
    )
    fig_anneal = plot_ze_estimates(fig_anneal, zne_estimates)

    return fig_noise, fig_anneal, coupling_data, zne_estimates, modal_trigger

#==============================================================================================================================

@dash.callback(
    Output("sample-v-theory-graph", "figure"),
    Output("kz-data", "data"),
    inputs=[
        Input("selected-problem-type", "data"),
        Input("graph-selection-radio", "value"),
        Input("schedule-filename", "children"),
        Input("coupling-strength", "value"),  # previously input
        Input("spins", "value"),
        Input("anneal-duration", "value"),
        State("kz-data", "data"),  # get kibble zurek data point
    ],
)
def load_new_graph_kz(
    problem_type: str,
    graph_selection: str,
    schedule_filename: str,
    J: float,
    spins: str,
    ta: int | str,
    kz_data: list,
) -> tuple[go.Figure, list]:
    """Initiates graphics for kink density based on theory and QPU samples on page load and when
    when settings change.

    Args:
        problem_type: Either KZ (``0`` or ``ProblemType.KZ``) or
            KZ_NM (``1`` or ``ProblemType.KZ_NM``).
        graph_selection: The value of the graph selection radio, either "schedule", "kink_density",
            or "both".
        schedule_filename: The name of the quench schedule file being used.
        J: The value of the coupling strength setting.
        spins: The value of the spins setting.
        ta: The value of the anneal duration setting, in nanoseconds.
        kz_data: The existing data points for the kink density graph.

    Returns:
        A tuple containing:

        - go.Figure: The figure for the sample vs theory graph, either initialized or updated with
        new data point.
        - list: Data points for the kink density graph, either unchanged or with new point added.
    """
    if problem_type is ProblemType.KZ_NM.value:
        raise PreventUpdate

    if ctx.triggered_id in ["schedule-filename", "spins", "coupling-strength"]:
        kz_data = []

    fig = plot_kink_densities_bg(
        graph_selection,
        [2, 350],
        J,
        schedule_filename,
        kz_data,
    )
    return fig, kz_data

#==============================================================================================

@dash.callback(
    Output("kink-v-noise-graph", "figure"),
    Output("kink-v-anneal-graph", "figure"),
    Output("coupling-data", "data"),  # store data using dcc
    Output("zne-estimates", "data"),  # update zne_estimates
    inputs=[
        Input("schedule-filename", "children"),
        Input("spins", "value"),
        Input("selected-problem-type", "data"),
    ],
)
def load_new_graphs_kz_nm(
    schedule_filename: str, spins: str, problem_type: str
) -> tuple[go.Figure, go.Figure, dict, dict]:
    """Initiates KZ Noise Mitigation graphs on page load and when settings change.

    Args:
        schedule_filename: The name of the quench schedule file being used.
        spins: The value of the spins setting.
        problem_type: Either KZ (``0`` or ``ProblemType.KZ``) or
            KZ_NM (``1`` or ``ProblemType.KZ_NM``).

    Returns:
        A tuple containing:

        - go.Figure: The initialized figure for the kink density vs noise graph.
        - go.Figure: The initialized figure for the kink density vs anneal duration graph.
        - dict: An empty dictionary to store coupling data points.
        - dict: An empty dictionary to store zero noise extrapolation estimates.
    """
    if problem_type is ProblemType.KZ.value:
        raise PreventUpdate

    time_range = [2, 1500]

    if not schedule_filename:
        schedule_filename = "FALLBACK_SCHEDULE.csv"

    n = theoretical_kink_density(time_range, J_BASELINE, schedule_filename)

    fig_noise = kink_v_noise_init_graph(n)
    fig_anneal = kink_v_anneal_init_graph(time_range, n)

    return fig_noise, fig_anneal, {}, {}

#=======================================================================================================================

@dash.callback(
    Output("job-id", "data"),
    Output("wd-job", "n_intervals"),
    inputs=[
        Input("job-submit-time", "data"),
        State("qpu-selection", "value"),
        State("spins", "value"),
        State("coupling-strength", "value"),
        State("anneal-duration", "value"),
        State("embeddings", "data"),
        State("selected-problem-type", "data"),
        State("schedule-filename", "children"),
    ],
    prevent_initial_call=True,
)
def submit_job(
    job_submit_time: str,
    qpu_name: str,
    spins: str,
    J: float,
    ta: int | str,
    embeddings: dict,
    problem_type: str,
    filename: str,
) -> tuple(str, int):
    """Submit job and provide job ID.

    Args:
        job_submit_time: The time that the job was submitted.
        qpu_name: The name of the quantum processing unit (QPU) to which the job is being submitted.
        spins: The value of the spins setting.
        J: The value of the coupling strength setting.
        ta: The value of the anneal duration setting, in nanoseconds.
        embeddings: A dictionary of cached embeddings for different numbers of spins.
        problem_type: Either KZ (``0`` or ``ProblemType.KZ``) or
            KZ_NM (``1`` or ``ProblemType.KZ_NM``).
        filename: The name of the schedule file being used.

    Returns:
        - job_id: The ID of the job that was submitted.
        - wd_job_n_intervals: The number of intervals that have passed since the simulation
        started.
    """

    solver = get_solvers()[qpu_name]
    spins = int(spins)
    ta = int(ta.split(" ")[0]) if isinstance(ta, str) else ta

    bqm = create_bqm(num_spins=spins, coupling_strength=J)

    embeddings = json_to_dict(embeddings)
    embedding = embeddings[spins]
    annealing_time = ta / 1000

    bqm_embedded = embed_bqm(bqm, embedding, DWaveSampler(solver=solver.name).adjacency)

    # ta_multiplier should be 1, unless (withNoiseMitigation and [J or schedule]) changes,
    # shouldn't change for MockSampler. In which case recalculate as
    # ta_multiplier=calclambda_(coupling_strength, schedule) as a function of the
    # correct schedule
    # State("ta_multiplier", "value") ? Should recalculate when J or schedule changes IFF noise mitigation tab?
    ta_multiplier = 1

    if problem_type is ProblemType.KZ_NM.value:
        ta_multiplier = calclambda_(J, schedule_name=filename)

    computation = solver.sample_bqm(
        bqm=bqm_embedded,
        fast_anneal=True,
        annealing_time=annealing_time * ta_multiplier,
        auto_scale=False,
        answer_mode="raw",  # Easier than accounting for num_occurrences
        num_reads=100,
        label=f"Examples - Kibble-Zurek Simulation, submitted: {job_submit_time}",
    )

    return computation.wait_id(), 0

#==============================================================================================

class RunButtonClickReturn(NamedTuple):
    """Return type for the ``run_button_click`` callback function."""

    run_button_disabled: bool = True
    wd_job_disabled: bool = False
    wd_job_n_intervals: int = 0
    job_submit_state: str = dash.no_update
    job_submit_time: str = dash.no_update


@dash.callback(
    Output("run-button", "disabled"),
    Output("wd-job", "disabled"),
    Output("wd-job", "n_intervals", allow_duplicate=True),
    Output("job-submit-state", "children"),
    Output("job-submit-time", "data"),
    inputs=[
        Input("run-button", "n_clicks"),
        State("cached-embeddings", "children"),
        State("spins", "value"),
    ],
    prevent_initial_call=True,
)
def run_button_click(
    run_btn_click: int,
    cached_embeddings: str,
    spins: str,
) -> RunButtonClickReturn:
    """Start simulation run when button is clicked.

    Args:
        run_btn_click: The number of times the run button has been clicked.
        cached_embeddings: A string representation of which embeddings are cached.
        spins: The value of the spins setting.

    Returns:
        A NamedTuple, RunButtonClickReturn, containing:

        - run_button_disabled: Whether the run button should be disabled.
        - wd_job_disabled: Whether the interval component should be disabled.
        - wd_job_n_intervals: The number of intervals that have passed since the simulation started.
        - job_submit_state: The new state of the job submission process.
        - job_submit_time: The time that the job was submitted.
    """
    if str(spins) in cached_embeddings.split(", "):  # If we have a cached embedding
        return RunButtonClickReturn(
            job_submit_state="SUBMITTED",
            job_submit_time=datetime.datetime.now().strftime("%c"),
        )

    return RunButtonClickReturn(job_submit_state="EMBEDDING")

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