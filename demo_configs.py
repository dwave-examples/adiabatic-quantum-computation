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

from src.demo_enums import ModelType

"""This file stores input parameters for the app."""

THUMBNAIL = "static/dwave_logo.svg"

APP_TITLE = "Adiabatic Quantum Computation"
MAIN_HEADER = "Adiabatic Quantum Computation"
DESCRIPTION = """\
Demonstration of adiabatic quantum computation in frustrated 25-qubit Ising models, compared with exact modeling.
"""

#######################################
# Sliders, buttons and option entries #
#######################################

# an example slider
SLIDER = {
    "min": 1,
    "max": 10,
    "step": 1,
    "value": 5,
}

# an example dropdown
DROPDOWN = ["Option 1", "Option 2"]

# an example checklist
CHECKLIST = ["Option 1", "Option 2"]

# an example radio list
RADIO = ["Option 1", "Option 2"]

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 10,
    "max": 300,
    "step": 5,
    "value": 10,
}

THUMBNAIL = "static/dwave_logo.svg"

# config settings for ZNE tab
MAIN_HEADER_NM = "Zero-Noise Extrapolation"
DESCRIPTION_NM = """
    The statistics of a target J chain can be inferred using weaker coupling and longer quench
    durations, though this increases noise. By collecting data at multiple noise levels, it is
    possible to extrapolate to a denoised result.
    """

ANNEALING_TIME_SERIES = [5, 7, 10, 14, 20]
NUM_PARALLEL_EMBEDDINGS = 10

# TODO link to enums
DEFAULT_QPU = "Advantage_system4"  # If not available, the first returned will be default
DEFAULT_MODEL = ModelType.LARGE_GAP

MODEL_COLORS_THEORY = {
    ModelType.LARGE_GAP: "#1f77b4",
    ModelType.INTERMEDIATE_GAP: "#2ca02c",
    ModelType.SMALL_GAP: "#ff7f0e",
}

MODEL_COLORS_RESULTS = {
    ModelType.LARGE_GAP: "#299ef1",
    ModelType.INTERMEDIATE_GAP: "#1cf21c",
    ModelType.SMALL_GAP: "#9F520E",
}

JOB_BAR_DISPLAY = { # TODO make enum for this
    "READY": [0, "#737373"],
    "EMBEDDING": [20, "#FF7006"],
    "NO SOLVER": [100, "#AA3A3C"],
    "SUBMITTED": [40, "#03B8FF"],
    "PENDING": [60, "#2A7DE1"],
    "IN_PROGRESS": [85, "#2A7DE1"],
    "COMPLETED": [100, "#17BEBB"],
    "CANCELLED": [100, "#737373"],
    "FAILED": [100, "#AA3A3C"],
    "ERROR": [100, "#AA3A3C"],
}

