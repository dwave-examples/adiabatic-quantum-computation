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

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from demo_configs import DATA_SOURCE
from src.demo_enums import ModelType, ProblemType
from src.values import MODELS_PATH


class ProblemModel:
    """This class exists to simplify and encapsulate the loading and saving of specific problem instances:
    i.e. graphs including h and J values as they'll be submitted to the QPU. It helps with tracking and
    loading any other files that have relevant information for a given problem instance, such as
    the points to create an energy spectrum scatter plot, and the theoretical predictions for
    ground state probabilities."""

    DEFINITION_FILE_SUFFIX = "_PROBLEM_DEFINITION.json"
    QPU_SUFFIX = DATA_SOURCE.suffix
    SCATTER_SUFFIX = (
        DATA_SOURCE.ADVANTAGE_SYSTEM6.suffix
        if DATA_SOURCE == DATA_SOURCE.ZENODO_REFERENCE
        else DATA_SOURCE.suffix
    )

    def __init__(
        self,
        model_type: ModelType,
        problem_type: ProblemType = ProblemType.DEFAULT,
        h: dict | None = None,
        J: dict | None = None,
        EGS: float | None = None,
    ):
        self.problem_type = problem_type
        self.model_type = model_type
        self.h = h
        self.J = J
        self.EGS = EGS

    def __eq__(self, other):
        if not isinstance(other, ProblemModel):
            return False
        if self.h is None or self.J is None:
            return False
        return (
            self.problem_type == other.problem_type
            and self.model_type == other.model_type
            and self.h == other.h
            and self.J == other.J
            and self.EGS == other.EGS
        )

    @property
    def lookup_key(self) -> str:
        """Combination of problem type and model type should be sufficient to uniquely
        identify a problem instance, so it's currently used as the basis for associated file
        names and the key for dictionaries of problem instances."""
        return f"{self.problem_type.name}_{self.model_type.name}"

    @property
    def problem_definition_json_filename(self) -> str:
        """File name of the JSON problem definition for this model."""
        return f"{self.lookup_key}{ProblemModel.DEFINITION_FILE_SUFFIX}"

    @property
    def spectrum_scatter_filename(self) -> str:
        """File name of the CSV energy spectrum data for this model."""
        return f"{self.lookup_key}_SCATTER_{ProblemModel.SCATTER_SUFFIX}.csv"

    @property
    def theory_data_filename(self) -> str:
        """File name of the theoretical excited state probability curve for this model."""
        return f"{self.lookup_key}_THEORY_{ProblemModel.QPU_SUFFIX}.csv"

    @property
    def graph_nodes(self) -> set:
        """All nodes of the problem graph."""
        edge_nodes = {n for edge in self.J.keys() for n in edge}
        return set(self.h.keys()).union(edge_nodes)

    @property
    def graph_edges(self) -> set:
        """All edges of the problem graph."""
        return set(self.J.keys())

    def to_dict(self) -> dict:
        """JSON-serializable dict of h, J, and EGS."""
        model_dict = {"h": self.h, "J": {str(n): v for n, v in self.J.items()}, "EGS": self.EGS}

        return model_dict

    def problem_to_json_file(self, overwrite_existing: bool = False):
        """Save this model to its JSON problem definition file in MODELS_PATH.

        Args:
            overwrite_existing: If False, raise ValueError when a different model is already saved
                under the same lookup key.
        """
        model_dict = self.to_dict()
        model_path = MODELS_PATH / self.problem_definition_json_filename

        if not overwrite_existing and model_path.exists():
            existing_model = ProblemModel.problem_from_json_file(model_path.name)
            if self != existing_model:
                raise ValueError(
                    f"Model {self.lookup_key} already exists with different properties."
                )

        with model_path.open("w") as f:
            json.dump(model_dict, f)

    @staticmethod
    def from_dict(lookup_key: str, props_dict: dict) -> ProblemModel:
        """Build a ProblemModel from a lookup key and a dict.

        Args:
            lookup_key: The lookup key identifying the problem and model type.
            props_dict: A dictionary containing the properties of the problem model.

        Returns:
            A ProblemModel instance built from the lookup key and properties dictionary.
        """
        problem_type_name, model_type_name = lookup_key.split(
            "_", 1
        )  # TODO replace with more durable method
        problem_type = ProblemType[problem_type_name]
        model_type = ModelType[model_type_name]
        h = {int(u): v for u, v in props_dict["h"].items()}
        J = {tuple(map(int, k.strip("()").split(","))): v for k, v in props_dict["J"].items()}
        EGS = props_dict["EGS"]
        return ProblemModel(model_type, problem_type, h, J, EGS)

    @staticmethod
    def problem_from_json_file(json_filename: str) -> ProblemModel:
        """Loads a problem from its JSON definition file.

        Args:
            json_filename: The name of the JSON file containing the problem definition.

        Returns:
            A ProblemModel instance built from the JSON file.
        """
        with (MODELS_PATH / json_filename).open("r") as f:
            model_dict = json.load(f)
        lookup_key = json_filename.replace(ProblemModel.DEFINITION_FILE_SUFFIX, "")
        return ProblemModel.from_dict(lookup_key, model_dict)

    @staticmethod
    def load_all_models_from_file(as_dicts: bool = False) -> dict:
        """Load every saved problem definition, keyed by lookup key.

        Args:
            as_dicts: If True, values are JSON-safe dicts instead of ProblemModels.

        Returns:
            A dictionary containing all loaded problem models, keyed by their lookup keys.
        """
        models_dict = {}
        for model_path in MODELS_PATH.glob(f"*{ProblemModel.DEFINITION_FILE_SUFFIX}"):
            model = ProblemModel.problem_from_json_file(model_path.name)
            models_dict[model.lookup_key] = model.to_dict() if as_dicts else model

        return models_dict

    @staticmethod
    @lru_cache
    def load_spectrum_scatter_from_file(
        model_type: ModelType, problem_type: ProblemType = ProblemType.DEFAULT
    ) -> np.ndarray:
        """Loads the data to create a scatter plot of the energy spectrum for a given problem instance.
        The result is cached, so each file is read once.

        Args:
            problem_type: The type of the problem instance.
            model_type: The type of the model instance.

        Returns:
            A numpy ndarray containing the scatter plot data.
        """

        scatter_path = (
            MODELS_PATH / ProblemModel(model_type, problem_type).spectrum_scatter_filename
        )
        if not scatter_path.exists():
            raise FileNotFoundError(f"Scatter CSV file not found at {scatter_path}.")
        return np.loadtxt(scatter_path, delimiter=",", skiprows=3)

    @staticmethod
    @lru_cache
    def load_theory_data_from_file(
        problem_type: ProblemType, model_type: ModelType
    ) -> tuple[np.ndarray, np.ndarray]:
        """Loads the data for the theoretical predictions of ground state probability as a
        function of annealing time for a given problem instance. The result is cached, so each
        file is read once.

        Args:
            problem_type: The type of the problem instance.
            model_type: The type of the model instance.

        Returns:
            A tuple containing:

            - np.ndarray: The annealing times (ns).
            - np.ndarray: The corresponding excited state probabilities (1 - P_GS) predicted by
              the theoretical model.
        """
        theory_data_path = MODELS_PATH / ProblemModel(model_type, problem_type).theory_data_filename
        if not theory_data_path.exists():
            raise FileNotFoundError(f"Theory data file not found at {theory_data_path}.")

        theory_vals = np.loadtxt(theory_data_path, delimiter=",", skiprows=2)
        xs = theory_vals[:, 0]
        ys = theory_vals[:, 1]

        return xs, ys
