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


class ProblemType(Enum):
    """TODO: Future proofing for when we want to add problems with other graph structures to the demo.
    We'll need a consistent way to refer to them programmatically, and an Enum is easier than
    trying to distill the graph structure itself into a unique identifier. The intent is that
    every graph structure should get a unique Enum name, but different models (i.e. different
    sets of weights on the same graph structure) can share the same Enum name. When we need
    to recover the actual graph structure, it can be inferred from any file of model weights
    that matches the Enum name."""

    DEFAULT = 0


class ModelType(Enum):
    """Encapsulates options for different models--i.e. different sets of weights on the same problem
    graph. The demo depends on showing the differences in behavior between the different models."""

    SMALL_GAP = 0
    INTERMEDIATE_GAP = 1
    LARGE_GAP = 2

    @property
    def label(self):
        return self.name.replace("_", " ").title()


class JobState(Enum):
    """Represents the possible states of a job on the QPU. The demo uses this both to
    to display as user feedback and to check the state of a job to control which callbacks
    can execute."""

    FINDING_EMBEDDING = 0
    READY = 1
    PREPROCESSING = 2
    CONNECTING = 3
    RUNNING = 4
    NO_SOLVER = 5
    FAILED = 6

    @property
    def label(self):
        return self.name.replace("_", " ").title()


class TheoryCurveMode(Enum):
    """Represents the theory curves that can be plotted alongside the sampled data: the published
    Schrödinger-evolution reference, the Landau-Zener fit, or both."""

    LANDAU_ZENER = 0
    REFERENCE = 1

    @property
    def label(self):
        return self.name.replace("_", "-").title()

    @property
    def extended_label(self):
        """Longer description of the curve, used in plot legends."""
        long_labels = {
            self.REFERENCE: "Rescaled Published Reference",  # TODO link to data source
            self.LANDAU_ZENER: "Landau-Zener Theory",
        }
        return long_labels[self]


class ScatterType(Enum):
    """Tracks the different types of scatter plots created by the plotting function. Each
    type of plot has different requirements for formatting and data labelling."""

    THEORY_REF = 0
    THEORY_LZ = 1
    SAMPLE = 2
    PARALLEL_SAMPLES = 3
    SPECTRUM = 4
    ENERGY_GAP = 5


class DataSource(Enum):
    """Source of the precomputed theory curves and energy spectra in ``static/models``. Each
    source has a matching set of data files and a QPU whose annealing schedule they were
    computed under."""

    ZENODO_REFERENCE = 0
    ADVANTAGE_SYSTEM6 = 1
    ADVANTAGE2_SYSTEM1 = 2

    @property
    def suffix(self):
        """File name suffix of the theory curve files for this data source."""
        labels = {
            self.ZENODO_REFERENCE: "ZENODO",
            self.ADVANTAGE_SYSTEM6: "ADV_6",
            self.ADVANTAGE2_SYSTEM1: "ADV2_1",
        }
        return labels[self]

    @property
    def spectrum_suffix(self):
        """File name suffix of the energy spectrum files for this data source. The Zenodo
        theory curves have no spectrum of their own, so they use the Advantage_system6 one."""
        if self is DataSource.ZENODO_REFERENCE:
            return DataSource.ADVANTAGE_SYSTEM6.suffix
        return self.suffix

    @property
    def qpu_name(self):
        """Name of the QPU whose annealing schedule this data source was computed under."""
        names = {
            self.ZENODO_REFERENCE: "Advantage_system6",
            self.ADVANTAGE_SYSTEM6: "Advantage_system6",
            self.ADVANTAGE2_SYSTEM1: "Advantage2_system1",
        }
        return names[self]
