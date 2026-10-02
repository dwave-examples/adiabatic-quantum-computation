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

"""This file stores input parameters for the app."""

from src.demo_enums import ModelType

APP_TITLE = "Adiabatic Quantum Computation"
MAIN_HEADER = "Adiabatic Quantum Computation"
DESCRIPTION = """\
Demonstration of adiabatic quantum computation in frustrated 16-qubit Ising models, compared with exact modeling.
"""

THUMBNAIL = "static/dwave_logo.svg"

DEFAULT_ANNEALING_TIMES = [5, 7, 10, 14, 20]
MIN_ANNEALING_TIME = 5.0
MAX_ANNEALING_TIME = 30.0
ANNEAL_TIME_STATIC_MARKS = [5.0, 10.0, 20.0, 30.0]

# Number of parallel embeddings for the model. More embeddings mean more accurate results
# but they'll take longer to find if they're not already saved.
NUM_PARALLEL_EMBEDDINGS = 20
SHOW_PARALLEL_SAMPLES = (
    False  # Mostly for calibration, but can make for an interesting visual as well
)
NUM_READS = 100

COUPLING_STRENGTH_SLIDER_MARKS = [0.2, 0.4, 0.6, 0.8, 1.0]

# TODO link to enums
DEFAULT_QPU = "Advantage_system4"  # If not available, the first returned will be default
DEFAULT_MODEL = ModelType.INTERMEDIATE_GAP


SAMPLE_POINT_SIZE = 6
DEFAULT_PLOT_WIDTH = 21
MAX_PLOT_WIDTH = 31
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
