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
STATIC_PATH = REPO_PATH / "static"
SCHEDULES_AND_EMBEDDINGS_PATH = REPO_PATH / "schedules_and_embeddings"
MODELS_PATH = STATIC_PATH / "models"

ENERGY_THRESHOLD = 1e-8  # Relative to ground state energy
MIN_ANNEALING_TIME = 5
MAX_ANNEALING_TIME = 30

# Results plot tuning. The x-axis shows DEFAULT_PLOT_WIDTH ns of annealing time and widens to
# MAX_PLOT_WIDTH once a sample lands near the right edge.
SAMPLE_POINT_SIZE = 14
DEFAULT_PLOT_WIDTH = 21
MAX_PLOT_WIDTH = 31

# Candidate lower limits (log10) for the results plot y-axis. The plot steps down through these
# until the smallest sampled probability is visible.
RESULTS_FIG_Y_MINS = [-1.0, -1.5, -2.0]
