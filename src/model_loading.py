import json
import os

import numpy as np

from src.demo_enums import ProblemType, ModelType
from src.values import MODELS_PATH

class ProblemModel:
    def __init__(self, problem_type: ProblemType, model_type: ModelType, h: dict|None = None, J: dict|None = None, EGS: float|None = None):
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
        return (self.problem_type == other.problem_type and
                self.model_type == other.model_type and
                self.h == other.h and
                self.J == other.J and
                self.EGS == other.EGS)

    @property
    def lookup_key(self) -> str:
        return f"{self.problem_type.name}_{self.model_type.name}"

    @property
    def json_filename(self) -> str:
        return f"{self.lookup_key}.json"

    @property
    def scatter_filename(self) -> str:
        return f"{self.lookup_key}_scatter.csv"

    @property
    def graph_nodes(self):
        edge_nodes = {n for edge in self.J.keys() for n in edge}
        return set(self.h.keys()).union(edge_nodes)

    @property
    def graph_edges(self):
        return set(self.J.keys())

    def to_dict(self):
        model_dict = {
            "h": self.h,
            "J": {str(n): v for n, v in self.J.items()},
            "EGS": self.EGS
        }

        return model_dict

    def set_min_energy_gap(self):
        try:
            filedata =ProblemModel.load_scatter_csv_from_file(self.problem_type, self.model_type)
            energy_series = filedata[:,2] -filedata[:,1]  # Energy of the first excited state minus ground state energy
            E_first_excited = min(energy_series)  # Minimum energy of the first excited state
        except FileNotFoundError:
            E_first_excited = None

    def to_json_file(self, overwrite_existing: bool = False):
        model_dict = self.to_dict()
        #model_dict["J"] = {str(k): v for k, v in model_dict["J"].items()} #TODO this looks redundant. Examine.

        if not overwrite_existing:
            pre_saved_models = os.listdir(MODELS_PATH)
            for model_filename in pre_saved_models:
                if model_filename == self.json_filename:
                    existing_model = ProblemModel.from_json_file(model_filename)
                    if self != existing_model:
                        raise ValueError(
                            f"Model {self.lookup_key} already exists with different properties."
                        )

        model_path = os.path.join(MODELS_PATH, self.json_filename)
        with open(model_path, "w") as f:
            json.dump(model_dict, f)

    @staticmethod
    def from_dict(lookup_key: str, props_dict: dict):
        problem_type_name, model_type_name = lookup_key.split("_", 1) #TODO replace with more durable method
        problem_type = ProblemType[problem_type_name]
        model_type = ModelType[model_type_name]
        h = {int(u): v for u, v in props_dict["h"].items()}
        J = {tuple(map(int, k.strip("()").split(","))): v for k, v in props_dict["J"].items()}
        EGS = props_dict["EGS"]
        return ProblemModel(problem_type, model_type, h, J, EGS)

    @staticmethod
    def from_json_file(json_filename: str):
        model_path = os.path.join(MODELS_PATH, json_filename)
        with open(model_path, "r") as f:
            model_dict = json.load(f)
        lookup_key = json_filename.replace(".json", "")
        return ProblemModel.from_dict(lookup_key, model_dict)

    @staticmethod
    def load_all_models_from_file(as_dicts: bool = False):
        models_dict = {}
        for model_filename in os.listdir(MODELS_PATH):
            if model_filename.endswith(".json"):
                model = ProblemModel.from_json_file(model_filename)
                if as_dicts:
                    models_dict[model.lookup_key] = model.to_dict()
                else:
                    models_dict[model.lookup_key] = model

        return models_dict

    @staticmethod
    def load_scatter_csv_from_file(problem_type: ProblemType, model_type: ModelType):
        lookup_key = f"{problem_type.name}_{model_type.name}"
        scatter_filename = f"{lookup_key}_SCATTER.csv"
        scatter_path = os.path.join(MODELS_PATH, scatter_filename)
        if not os.path.exists(scatter_path):
            raise FileNotFoundError(f"Scatter CSV file not found for model {scatter_filename}.")
        filedata = np.loadtxt(scatter_path, delimiter=',', skiprows=3)
        return filedata

    @staticmethod
    def load_theory_data_from_file(problem_type: ProblemType, model_type: ModelType):
        lookup_key = f"{problem_type.name}_{model_type.name}"
        data_filename = f"{lookup_key}_THEORY.txt"
        theory_data_path = os.path.join(MODELS_PATH, data_filename)
        if not os.path.exists(theory_data_path):
            raise FileNotFoundError(f"Theory data file not found for model {data_filename}.")
        with open(theory_data_path, "r") as f:
            lines = f.readlines()
        xs = [float(num) for num in lines[0].strip().split(",")]
        ys = [float(num) for num in lines[1].strip().split(",")]
        return xs, ys