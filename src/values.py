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

import pathlib

REPO_PATH = pathlib.Path(__file__).parent.parent.resolve()
EMBEDDINGS_PATH = REPO_PATH / "embeddings"
MODELS_PATH = REPO_PATH / "static" / "models"

ENERGY_THRESHOLD = 1e-8  # Relative to ground state energy
MIN_ANNEALING_TIME = 5
MAX_ANNEALING_TIME = 30

# Results plot tuning. The x-axis shows DEFAULT_PLOT_WIDTH ns of annealing time and widens to
# MAX_PLOT_WIDTH once a sample lands near the right edge.
SAMPLE_POINT_SIZE = 10
DEFAULT_PLOT_WIDTH = 21
MAX_PLOT_WIDTH = 31

ENERGY_TIME_RESCALING = {  # Used to rescale energy axis for plotting samples from
    "Advantage2_system5": 2.48,  # processors that have different energy scales.
    "Advantage2_system1": 2.85,
    "Advantage_system4": 1.09,
}

Y_AXIS_RESCALING = {
    "Advantage2_system5": -1.9,
    "Advantage2_system1": -1.9,
}
