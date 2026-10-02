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

"""This file stores the Dash HTML layout for the app."""

from __future__ import annotations

from enum import EnumMeta

import dash_mantine_components as dmc
from dash import dcc, html

from demo_configs import DEFAULT_MODEL, DEFAULT_QPU, DESCRIPTION, MAIN_HEADER, THUMBNAIL, COUPLING_STRENGTH_SLIDER_MARKS, DEFAULT_ANNEALING_TIMES, MIN_ANNEALING_TIME, MAX_ANNEALING_TIME, ANNEAL_TIME_STATIC_MARKS
from src.demo_enums import JobState, ModelType, ProblemType
from src.model_loading import ProblemModel
from src.qpu_resources import get_qpu_ids


def dropdown(label: str, id: str, options: list, value: str | None = None) -> html.Div:
    """Dropdown element for option selection.

    Args:
        label: The title that goes above the dropdown.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        value: The initial value for the dropdown. If None, the first option will be used.

    Returns:
        A Div containing the dropdown element.
    """
    return html.Div(
        className="dropdown-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Select(
                id=id,
                data=options,
                value=value if value else options[0]["value"],
                allowDeselect=False,
            ),
        ],
    )

def slider(label: str, id: str, config: dict, marks: list | None = None) -> html.Div:
    """Slider element for value selection.

    Args:
        label: The title that goes above the slider.
        id: A unique selector for this element.
        config: A dictionary of slider configurations, see dmc.Slider Dash Mantine docs.
        marks: A list of values that should be marked on the slider.
    """
    return html.Div(
        className="slider-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Slider(
                id=id,
                className="slider",
                **config,
                marks=(
                    marks
                    if marks
                    else [
                        {"value": config["min"], "label": f'{config["min"]}'},
                        {"value": config["max"], "label": f'{config["max"]}'},
                    ]
                ),
                labelAlwaysOn=True,
                thumbLabel=f"{label} slider",
            ),
        ],
    )


def generate_options(options: list | EnumMeta | dict) -> list[dict]:
    """Format options for dropdowns, checklists, radios, etc.

    Args:
        options: A list, EnumMeta, or dictionary of options to format.

    Returns:
        A list of dictionaries with "label" and "value" keys for each option.
    """
    if isinstance(options, EnumMeta):
        return [{"label": option.label, "value": f"{option.value}"} for option in options]

    if isinstance(options, dict):
        return [{"label": f"{key}", "value": f"{value}"} for key, value in options.items()]

    return [{"label": f"{option}", "value": f"{option}"} for option in options]


def generate_settings_form(solvers: list[str]) -> html.Div:
    """Generate settings for selecting the scenario, model, and solver.

    Returns:
        A Div containing the settings for selecting the scenario, model, and solver.
    """
    # TODO link to specific solver list
    qpu_options = generate_options(solvers)

    model_options = generate_options(ModelType)

    if DEFAULT_QPU in solvers:
        default_solver = DEFAULT_QPU
    elif solvers:
        default_solver = solvers[0]
    else:
        default_solver = "No Leap Access"

    coupling_strength_marks = [{"value": mark, "label": f"{mark}"} for mark in COUPLING_STRENGTH_SLIDER_MARKS]
    coupling_strength_value = 1.0 # TODO add to configs

    anneal_time_marks = [{"value": mark, "label": f"{mark}"} for mark in ANNEAL_TIME_STATIC_MARKS]

    return html.Div(
        className="settings",
        children=[
            dropdown(
                "QPU",
                "qpu-selection",
                qpu_options,
                value=default_solver,
            ),
            dropdown(
                "Model",
                "model-selection",
                model_options,
                value=f"{DEFAULT_MODEL.value}",
            ),
            html.Div(
                slider(
                    "Annealing Time (ns)",
                    "annealing-time",
                    {
                        "value": DEFAULT_ANNEALING_TIMES[0],
                        "min": MIN_ANNEALING_TIME,
                        "max": MAX_ANNEALING_TIME,
                        "step": 1.0,
                        "restrictToMarks": False,
                    },
                    marks=anneal_time_marks,
                ),
                id="annealing-time-wrapper",
            ),
            html.Div(
                slider(
                    "Energy Rescaling",
                    "energy-rescaling",
                    {
                        "value": coupling_strength_value,
                        "min": coupling_strength_marks[0]["value"],
                        "max": coupling_strength_marks[-1]["value"],
                        "step": 0.1,
                        "restrictToMarks": True,
                    },
                    marks=coupling_strength_marks,
                ),
                id="coupling-strength-wrapper",
            ),
        ],
    )


def generate_run_buttons() -> html.Div:
    """Generate run button to run the experiment"""
    return html.Div(
        id="button-group",
        children=[
            html.Button("Run Experiment", id="run-button", className="button", disabled=True),
        ],
    )


def default_graph(title: str, id: str) -> html.Div:
    """Default graph element with a title.

    Args:
        title: The title that goes above the graph.
        id: A unique selector for this graph element.

    Returns:
        A Div containing the title and the graph.
    """

    return html.Div(
        className="graph-wrapper",
        children=[
            html.H3(title),
            dcc.Graph(
                id=f"{id}-graph",
                responsive=True,
                config={"displayModeBar": False},
            ),
        ],
    )


def generate_new_results_dict(problem_type: str = ProblemType.DEFAULT.name) -> dict:
    """Generates a new standardized dictionary for storing measurement results. Result sets
    should be specific to a given QPU, so this function is called whenever a new QPU is selected
    to replace the current-results data store with a new, empty results dictionary.

    Args:
        problem_type: The type of problem for which to generate the results dictionary.

    Returns:
        A dictionary with keys for each model type and an empty list for storing results.
    """

    results_dict = {m.name: {} for m in ModelType}  # TODO use more durable keys
    results_dict["problem_type"] = problem_type
    return results_dict


def show_status(solvers: list[str]) -> html.Div:
    """Show job submission state.

    Returns:
        A Div containing the job submission status and progress bar.

    #TODO: Progress bar has been left out of V1 of the KZ demo as it adds extra, unnecessary complexity.
    It should probably be added back later, but it's much lower priority than the functional pieces.
    """

    init_job_status = JobState.NO_SOLVER if not solvers else JobState.READY

    job_status_style = {"color": "#AA3A3C"} if init_job_status == JobState.NO_SOLVER else {}

    return html.Div(
        [
            html.P(
                [
                    "Status: ",
                    html.Span(
                        id="job-submit-state",
                        children=init_job_status.label,
                        style=job_status_style,
                    ),
                ],
                className="caption",
            ),
        ],
        className="progress-wrapper",
    )


def no_solver_modal() -> dmc.Modal:
    """Modal to alert users that no solvers are available."""
    return dmc.Modal(
        title="Leap's Quantum Computers Inaccessible",
        id="no-solver-modal",
        centered=True,
        children=[
            html.Div(
                [
                    html.H2("Could not connect to a Leap quantum computer"),
                    html.P(
                        [
                            """
                            If you are running locally, set environment variables or a
                            dwave-cloud-client configuration file as described in the
                            """,
                            html.A(
                                " Ocean",
                                href="https://docs.dwavequantum.com/en/latest/ocean/sapi_access_basic.html",
                                target="_blank",
                                rel="noopener",
                            ),
                            " documentation.",
                        ],
                    ),
                    html.P(
                        [
                            "If you are running in an online IDE, see the ",
                            html.A(
                                "system documentation",
                                href="https://docs.dwavequantum.com/en/latest/leap_sapi/dev_env.html",
                                target="_blank",
                                rel="noopener",
                            ),
                            " on supported IDEs.",
                        ],
                    ),
                ]
            )
        ],
    )


def create_interface() -> html.Div:
    """Create the main application interface."""

    solvers = get_qpu_ids(print_available=True)

    return html.Div(
        id="app-container",
        children=[
            html.A(  # Skip link for accessibility
                "Skip to main content",
                href="#main-content",
                id="skip-to-main",
                className="skip-link",
                tabIndex=1,
            ),
            dcc.Store(id="available-solvers", data=solvers),
            dcc.Store(id="qpu-select-time", data=0.0),
            dcc.Store(id="job-submit-time", data=0.0),
            dcc.Store(id="job-submit-args", data={}),
            dcc.Store(id="current-embeddings", data=[]),
            dcc.Store(
                id="cached-models", data=ProblemModel.load_all_models_from_file(as_dicts=True)
            ),
            dcc.Store(id="current-model", data={}),
            dcc.Store(id="current-results", data=generate_new_results_dict()),
            no_solver_modal(),
            # Settings and results columns
            html.Main(
                className="columns-main",
                id="main-content",
                children=[
                    # Left column
                    html.Div(
                        id={"type": "to-collapse-class", "index": 0},
                        className="left-column",
                        children=[
                            html.Div(
                                className="left-column-layer-1",  # Fixed width Div to collapse
                                children=[
                                    html.Div(
                                        className="left-column-layer-2",  # Padding and content wrapper
                                        children=[
                                            html.Div(
                                                [
                                                    html.H1(MAIN_HEADER),
                                                    html.P(DESCRIPTION),
                                                ],
                                                className="title-section",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        html.Div(
                                                            [
                                                                generate_settings_form(solvers),
                                                                generate_run_buttons(),
                                                                show_status(solvers),
                                                            ],
                                                            className="settings-and-buttons",
                                                        ),
                                                        className="settings-and-buttons-wrapper",
                                                    ),
                                                    # Left column collapse button
                                                    html.Div(
                                                        html.Button(
                                                            id={
                                                                "type": "collapse-trigger",
                                                                "index": 0,
                                                            },
                                                            className="left-column-collapse",
                                                            title="Collapse sidebar",
                                                            children=[
                                                                html.Div(className="collapse-arrow")
                                                            ],
                                                            **{"aria-expanded": "true"},
                                                        ),
                                                    ),
                                                ],
                                                className="form-section",
                                            ),
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        className="right-column",
                        children=[
                            html.Header(
                                className="banner",
                                children=[
                                    html.Img(src=THUMBNAIL, alt="D-Wave logo"),
                                ],
                            ),
                            html.Div(
                                className="content-wrapper",
                                children=[
                                    html.Div(
                                        className="plots-wrapper",
                                        children=[
                                            default_graph(
                                                "Experiment vs Theory",
                                                "sample-v-theory",
                                            ),
                                            default_graph(
                                                "Energy Spectrum",
                                                "spectrum",
                                            ),
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
