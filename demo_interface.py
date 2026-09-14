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

from dash import dcc, html
import dash_mantine_components as dmc

from demo_configs import (
    DEFAULT_QPU,
    DEFAULT_MODEL,
    DESCRIPTION,
    MAIN_HEADER,
    THUMBNAIL,
)
from src.demo_enums import SolverType, ProblemType, ModelType, JobState
from src.qpu_resources import get_init_job_status, get_solvers, get_qpu_ids
from src.model_loading import ProblemModel

THEME_COLOR = "#2d4376"

def dropdown(label: str, id: str, options: list, value: str | None = None) -> html.Div:
    """Dropdown element for option selection.

    Args:
        label: The title that goes above the dropdown.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
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


def generate_options(options: list | EnumMeta) -> list[dict]:
    """Generates options for dropdowns, checklists, radios, etc."""
    if isinstance(options, EnumMeta):
        return [{"label": option.label, "value": f"{option.value}"} for option in options]

    return [{"label": f"{option}", "value": f"{option}"} for option in options]


def generate_settings_form() -> html.Div:
    """This function generates settings for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the settings for selecting the scenario, model, and solver.
    """
    solvers = get_solvers() #TODO link to specific solver list
    qpu_options = generate_options(solvers)

    model_options = generate_options(ModelType)

    if DEFAULT_QPU in solvers:
        default_solver = DEFAULT_QPU
    elif len(solvers):
        default_solver = list(solvers.keys())[0]
    else:
        default_solver = "No Leap Access"

    return html.Div(
        className="settings",
        children=[
            dropdown(
                "QPU",
                "qpu-selection",
                sorted(qpu_options, key=lambda op: op["value"]),
                value=default_solver,
            ),
            dropdown(
                "Model",
                "model-selection",
                model_options,
                value=DEFAULT_MODEL.name,
            ),
        ],
    )


def generate_run_buttons() -> html.Div:
    """Run and cancel buttons to run the simulation."""
    return html.Div(
        id="button-group",
        children=[
            html.Button("Run Simulation", id="run-button", className="button", disabled=True),
            html.Button(
                "Cancel Simulation",
                id="cancel-button",
                className="button",
                style={"display": "none"},
            ),
        ],
    )


def default_graph(title: str, id: str) -> html.Div:
    """Default graph element with a title and optional radio buttons for graph selection.

    Args:
        title: The title that goes above the graph.
        id: A unique selector for this graph element.
        load_radio: Whether to create radio buttons for graph selection or not.

    Returns:
        A Div containing the graph and optional radio buttons.
    """

    return html.Div(
        [
            html.H3(title),
            dcc.Graph(
                id=f"{id}-graph",
                responsive=True,
                config={"displayModeBar": False},
            ),
        ]
    )

def generate_new_results_dict(problem_type: str = ProblemType.DEFAULT.name) -> dict:
    """ Generates a new standardized dictionary for storing measurement results. At present there
    is only one problem type and no functionality to change or reset exisitng results, so this will
    be called once at initialization and never again. If the demo is expanded to include more problem
    types or other options that would make clearing and resetting results useful, this function will
    make it easy to prepare the current-results store for a fresh batch of measurements."""

    results_dict = {m.name: [] for m in ModelType} #TODO use more durable keys
    results_dict["problem_type"] = problem_type
    return results_dict

def show_status() -> html.Div:
    """Show job submission state.

    Returns:
        A Div containing the job submission status and progress bar.

    #TODO: Progress bar has been left out of V1 of the KZ demo as it adds extra, unnecessary complexity.
    It should probably be added back later, but it's much lower priority than the functional pieces.
    """

    init_job_status = get_init_job_status()
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


def error_modal() -> dmc.Modal:
    """Modal to alert users that an error occurred creating a graph."""
    return dmc.Modal(
        title="Error",
        id="error-modal",
        centered=True,
        children=[
            html.P(
                "Fitting function failed likely due to ill conditioned data, please collect more."
            )
        ],
    )


def create_interface() -> html.Div:
    """Set the application HTML."""
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

            dcc.Store(id="available-solvers", data = get_qpu_ids()),
            dcc.Store(id="qpu-select-time", data=0.0), 
            dcc.Store(id="modal-trigger", data=False),
            dcc.Store(id="job-submit-time", data=0.0),
            dcc.Store(id="job-submit-args", data={}),
            dcc.Store(id="current-embeddings", data={}),
            dcc.Store(id="cached-models", data=ProblemModel.load_all_models_from_file(as_dicts=True)),
            dcc.Store(id="current-model", data={}),
            dcc.Store(id="current-results", data = generate_new_results_dict()),
            dcc.Store(id="current-problem-type", data=ProblemType.DEFAULT.name), # Currently no functionality for
            # changing this as we only have one problem type. But it's a lot easier to design around having
            # more problem types from the start than it is to add that in later.

            no_solver_modal(),
            error_modal(),
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
                                                    html.H1(MAIN_HEADER, id="main-header"),
                                                    html.P(DESCRIPTION, id="main-description"),
                                                ],
                                                className="title-section",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        html.Div(
                                                            [
                                                                generate_settings_form(),
                                                                generate_run_buttons(),
                                                                show_status(),
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
                            dmc.Tabs(
                                id="tabs",
                                value=f"tab-{ProblemType.DEFAULT.value + 1}",
                                color="white",
                                children=[
                                    html.Header(
                                        className="banner",
                                        children=[
                                            html.Img(src=THUMBNAIL, alt="D-Wave logo"),
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value=f"tab-{ProblemType.DEFAULT.value + 1}",
                                        tabIndex="12",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    html.Div(
                                                        className="plots-wrapper",
                                                        children=[
                                                            html.Div(
                                                                className="graph-wrapper",
                                                                children=[
                                                                    default_graph(
                                                                    "Experiment vs Theory",
                                                                    "sample-v-theory",
                                                                    ),
                                                                ],
                                                            ),
                                                            html.Div(
                                                                className="spectrum-plot-wrapper",
                                                                children=[
                                                                    default_graph(
                                                                    "Energy Spectrum",
                                                                    "spectrum",
                                                                    ),
                                                                ],
                                                            )
                                                        ]
                                                    )   
                                                ],
                                            )
                                        ],
                                    ),

                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )