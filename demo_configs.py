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

APP_TITLE = "Adiabatic Quantum Computation"
MAIN_HEADER = "Adiabatic Quantum Computation"
DESCRIPTION = """\
Demonstration of adiabatic quantum computation in frustrated 25-qubit Ising models, compared with exact modeling.
"""

THUMBNAIL = "static/dwave_logo.svg"

# config settings for ZNE tab
MAIN_HEADER_NM = "Zero-Noise Extrapolation"
DESCRIPTION_NM = """
    The statistics of a target J chain can be inferred using weaker coupling and longer quench
    durations, though this increases noise. By collecting data at multiple noise levels, it is
    possible to extrapolate to a denoised result.
    """

ANNEALING_TIME_SERIES = [5, 7, 10, 14, 20]
NUM_PARALLEL_EMBEDDINGS = 10 # Number of parallel embeddings for the model. More embeddings mean more accurate results
                             # but they'll take longer to find if they're not already saved.

# TODO link to enums
DEFAULT_QPU = "Advantage_system4"  # If not available, the first returned will be default
DEFAULT_MODEL = ModelType.INTERMEDIATE_GAP

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