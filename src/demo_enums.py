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

from enum import Enum


class SolverType(Enum):
    """Add a list of solver options here. If this demo only requires 1 solver,
    this functionality can be removed.

    """

    SOLVER_1 = 0
    SOLVER_2 = 1

    @property
    def label(self):
        return {
            SolverType.SOLVER_1: "Advantage2_system1",
            SolverType.SOLVER_2: "Advantage_system4",
        }[self]


### If any settings or variables are being used repeatedly, throughout the code, create a new
### Enum for the setting here to avoid string comparisons or other fragile code practices.


class ProblemType(Enum):
    """Future proofing for when we want to add problems with other graph structures to the demo.
    We'll need a consistent way to refer to them programmatically, and an Enum is easier than
    trying to distil the graph structure itself into a unique identifier. The intent is that
    every graph structure should get a unique Enum name, but different models (i.e. different
    sets of weights on the same grpah structure) can share the same Enum name. When we need
    to recover the actual graph structure, it can be inferred from any file of model weights
    that matches the Enum name."""

    DEFAULT = 0

    @property
    def label(self):
        return {
            ProblemType.DEFAULT: "Default AQC Workshop Problem",
        }[self]


class ModelType(Enum):
    SMALL_GAP = 0
    INTERMEDIATE_GAP = 1
    LARGE_GAP = 2

    @property
    def label(self):
        return self.name.replace("_", " ").title()


class JobState(Enum):

    FINDING_EMBEDDING = 0
    READY = 1
    PREPROCESSING = 2
    CONNECTING = 3
    RUNNING = 4
    COMPLETED = 5
    NO_SOLVER = 6

    @property
    def label(self):
        return self.name.replace("_", " ").title()

    @property
    def normal_job_states(self):
        return [
            js
            for js in JobState
            if js not in [JobState.FINDING_EMBEDDING, JobState.READY, JobState.NO_SOLVER]
        ]
