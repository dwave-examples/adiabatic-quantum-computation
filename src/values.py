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
STATIC_PATH = pathlib.Path(REPO_PATH) / "static"
EMBEDDINGS_PATH = pathlib.Path(REPO_PATH) / "embeddings"
MODELS_PATH = pathlib.Path(STATIC_PATH) / "models"
SCHEDULES_PATH = pathlib.Path(STATIC_PATH) / "annealing_schedules"

ENERGY_THRESHOLD = 1e-8  # Relative to ground state energy

pathlib.Path(EMBEDDINGS_PATH).mkdir(exist_ok=True)

ENERGY_TIME_RESCALING = {  # Used to rescale energy axis for plotting samples from
    "Advantage2_system5": 2.48,  # processors that have different energy scales.
    "Advantage2_system1": 2.85,
    "Advantage_system4": 1.09,
}

Y_AXIS_RESCALING = {
    "Advantage2_system5": -1.9,
    "Advantage2_system1": -1.9,
}

