import json
import os
import networkx as nx

from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.system import DWaveSampler

from src.demo_enums import ProblemType
from src.model_loading import ProblemModel
from demo_configs import NUM_PARALLEL_EMBEDDINGS
from src.values import EMBEDDINGS_PATH

class EmbeddingSet():
    def __init__(
            self, qpu_name: str,
            problem_type: ProblemType = ProblemType.DEFAULT,
            embeddings_dir = EMBEDDINGS_PATH
            ):
        
        self.qpu_name = qpu_name
        self.problem_type = problem_type
        self.embedding_id = f"{qpu_name}_{problem_type.name}"
        self.embeddings_path = os.path.join(embeddings_dir, f"{self.embedding_id}.json")
        self.target_graph = self.get_processor_graph()

    def get_processor_graph(self) -> nx.graph:
        sampler = DWaveSampler(solver=self.qpu_name)
        return sampler.to_networkx_graph()

    def get_embeddings(
        self, stringify_keys: bool = False
        ) -> dict:
        """
        Get the embedding for the selected QPU.

        Args:
            qpu_name (str): The name of the QPU.
            embeddings_dict (dict): A dictionary containing embeddings for different QPUs.

        Returns:
            dict: The embedding for the selected QPU.
        """


        if os.path.exists(self.embeddings_path):
            embeddings = self.load_from_json_file()
        else:
            # Find an embedding the hard way. If we find one, save it to file so we can re-use it later.
            models_dict = ProblemModel.load_all_models_from_file()
            source_graph = None
            for model in models_dict.values():
                if model.problem_type == self.problem_type:
                    source_graph = nx.Graph()
                    source_graph.add_nodes_from(model.h.keys())
                    source_graph.add_edges_from(model.J.keys())
                    break

            if source_graph is None: 
                raise ValueError(f"No model found for problem type {self.problem_type.name}")
        
            embeddings = find_multiple_embeddings(
                source_graph,
                self.target_graph,
                max_num_emb=NUM_PARALLEL_EMBEDDINGS,
                timeout=30
            )

            self.save_to_json_file(embeddings)

        #if stringify_keys:
            #embeddings = {str(k): v for k, v in embeddings.items()}

        return embeddings

    def save_to_json_file(self, embeddings: dict):
        with open(self.embeddings_path, "w") as f:
            json.dump(embeddings, f)

    def load_from_json_file(self):
        with open(self.embeddings_path, "r") as f:
            embedding_dict = json.load(f)
        return embedding_dict