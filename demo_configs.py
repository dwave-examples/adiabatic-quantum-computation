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
from src.values import MAX_ANNEALING_TIME, MIN_ANNEALING_TIME

APP_TITLE = "Adiabatic Quantum Computation"
MAIN_HEADER = "Adiabatic Quantum Computation"
DESCRIPTION = """\
Demonstration of adiabatic quantum computation in frustrated 16-qubit Ising models, compared with exact modeling.
"""

THUMBNAIL = "static/dwave_logo.svg"


ANNEAL_TIME_STATIC_MARKS = [5, 10, 20, 30]

# Suggested annealing times for the auto-suggest feature. When that feature is enabled,
# repeatedly clicking 'run' will step through these annealing times automatically, without
# needing to manually adjust the slider for each sample.
SUGGESTED_ANNEALING_TIMES = [MIN_ANNEALING_TIME, 7.0, 10.0, 14.0, 20.0, MAX_ANNEALING_TIME]

# Together these determine how many parallel embeddings will be used. More embeddings mean more
# accurate results but they'll take longer to find if they're not already saved, up to the number
# of seconds defined by EMBEDDING_TIMEOUT. Leaving NUM_PARALLEL_EMBEDDINGS at the default value of
# None will result in finding the maximum number of embeddings possible within the allotted time.
NUM_PARALLEL_EMBEDDINGS = None
EMBEDDING_TIMEOUT = 25

# When set to True, instead of error bars, this will show the full spread of mean results
# from each parallel embedding, highlighting the variability between embeddings. This is
# mostly for calibration, but it could provide a nice visual for presenting as well.
SHOW_PARALLEL_SAMPLES = False

# The number of reads for each job sent to the QPU. More reads are more accurate, but take longer.
NUM_READS = 400

COUPLING_STRENGTH_SLIDER_MARKS = [0.2, 0.4, 0.6, 0.8, 1.0]

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
