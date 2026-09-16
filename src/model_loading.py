import json
import os

import numpy as np

from src.demo_enums import ModelType, ProblemType
from src.values import MODELS_PATH


class ProblemModel:
    """This class exists to simplify and encapsulate the loading and saving of specific problem instances:
    i.e. graphs including h and J values as they'll be submitted to the QPU. It helps with tracking and
    loading any other files that have relevant information for a given problem instance, such as
    the points to create an energy spectrum scatter plot, and the theoretical predictions for
    ground state probabilities."""

    DEFINITION_FILE_SUFFIX = "_PROBLEM_DEFINITION.json"

    def __init__(
        self,
        problem_type: ProblemType,
        model_type: ModelType,
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
        return f"{self.lookup_key}{ProblemModel.DEFINITION_FILE_SUFFIX}"

    @property
    def spectrum_scatter_filename(self) -> str:
        return f"{self.lookup_key}_SCATTER.csv"

    @property
    def theory_data_filename(self) -> str:
        return f"{self.lookup_key}_THEORY.txt"

    @property
    def graph_nodes(self):
        edge_nodes = {n for edge in self.J.keys() for n in edge}
        return set(self.h.keys()).union(edge_nodes)

    @property
    def graph_edges(self):
        return set(self.J.keys())

    def to_dict(self):
        model_dict = {"h": self.h, "J": {str(n): v for n, v in self.J.items()}, "EGS": self.EGS}

        return model_dict

    def problem_to_json_file(self, overwrite_existing: bool = False):
        model_dict = self.to_dict()

        if not overwrite_existing:
            pre_saved_models = os.listdir(MODELS_PATH)
            for model_filename in pre_saved_models:
                if model_filename == self.problem_definition_json_filename:
                    existing_model = ProblemModel.problem_from_json_file(model_filename)
                    if self != existing_model:
                        raise ValueError(
                            f"Model {self.lookup_key} already exists with different properties."
                        )

        model_path = os.path.join(MODELS_PATH, self.problem_definition_json_filename)
        with open(model_path, "w") as f:
            json.dump(model_dict, f)

    @staticmethod
    def from_dict(lookup_key: str, props_dict: dict):
        problem_type_name, model_type_name = lookup_key.split(
            "_", 1
        )  # TODO replace with more durable method
        problem_type = ProblemType[problem_type_name]
        model_type = ModelType[model_type_name]
        h = {int(u): v for u, v in props_dict["h"].items()}
        J = {tuple(map(int, k.strip("()").split(","))): v for k, v in props_dict["J"].items()}
        EGS = props_dict["EGS"]
        return ProblemModel(problem_type, model_type, h, J, EGS)

    @staticmethod
    def problem_from_json_file(json_filename: str):
        model_path = os.path.join(MODELS_PATH, json_filename)
        with open(model_path, "r") as f:
            model_dict = json.load(f)
        lookup_key = json_filename.replace(ProblemModel.DEFINITION_FILE_SUFFIX, "")
        return ProblemModel.from_dict(lookup_key, model_dict)

    @staticmethod
    def load_all_models_from_file(as_dicts: bool = False):
        models_dict = {}
        for model_filename in os.listdir(MODELS_PATH):
            if model_filename.endswith(ProblemModel.DEFINITION_FILE_SUFFIX):
                model = ProblemModel.problem_from_json_file(model_filename)
                if as_dicts:
                    models_dict[model.lookup_key] = model.to_dict()
                else:
                    models_dict[model.lookup_key] = model

        return models_dict

    @staticmethod
    def load_spectrum_scatter_from_file(problem_type: ProblemType, model_type: ModelType):
        """Loads the data to create a scatter plot of the energy spectrum for a given problem instance."""
        scatter_path = os.path.join(
            MODELS_PATH, ProblemModel(problem_type, model_type).spectrum_scatter_filename
        )
        if not os.path.exists(scatter_path):
            raise FileNotFoundError(f"Scatter CSV file not found at {scatter_path}.")
        filedata = np.loadtxt(scatter_path, delimiter=",", skiprows=3)
        return filedata

    @staticmethod
    def load_theory_data_from_file(problem_type: ProblemType, model_type: ModelType):
        """Loads the data for the theoretical predictions of ground state probability as a
        function of annealing time for a given problem instance."""
        theory_data_path = os.path.join(
            MODELS_PATH, ProblemModel(problem_type, model_type).theory_data_filename
        )
        if not os.path.exists(theory_data_path):
            raise FileNotFoundError(f"Theory data file not found at {theory_data_path}.")
        with open(theory_data_path, "r") as f:
            lines = f.readlines()
        xs = [float(num) for num in lines[0].strip().split(",")]
        ys = [float(num) for num in lines[1].strip().split(",")]
        return xs, ys
