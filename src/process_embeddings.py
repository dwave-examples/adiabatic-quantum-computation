import json
import os
import networkx as nx

from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.system import DWaveSampler
from dwave.embedding import verify_embedding

from src.demo_enums import ProblemType
from src.model_loading import ProblemModel
from demo_configs import NUM_PARALLEL_EMBEDDINGS
from src.values import EMBEDDINGS_PATH

class EmbeddingSet():
    def __init__(
            self, qpu_name: str,
            target_graph: nx.graph,
            problem_type: ProblemType = ProblemType.DEFAULT,
            embeddings_dir = EMBEDDINGS_PATH,
            ):
        
        self.qpu_name = qpu_name
        self.problem_type = problem_type
        self.embedding_id = f"{qpu_name}_{problem_type.name}"
        self.embeddings_path = os.path.join(embeddings_dir, f"{self.embedding_id}.json")
        self.source_graph = self.get_source_graph()
        self.target_graph = target_graph

    def get_source_graph(self):
        """ Retrieves the source graph for the problem type from the saved model data."""
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

        return source_graph

    def get_embeddings(self) -> dict:
        """
        Get the embedding for the selected QPU. Checks for saved embeddings is file first,
        if none or saved or they are invalid or incomplete, finds new embeddings and saves 
        them to file before returning them.

        Args:
            qpu_name (str): The name of the QPU.
            embeddings_dict (dict): A dictionary containing embeddings for different QPUs.

        Returns:
            dict: The embedding for the selected QPU.
        """

        valid_embeddings = False

        if os.path.exists(self.embeddings_path):
            embeddings = self.load_from_json_file()
            if len(embeddings) >= NUM_PARALLEL_EMBEDDINGS:
                for emb in embeddings:
                    try:
                        verify_embedding(emb, self.source_graph, self.target_graph)
                    except:
                        print(f"Invalid embedding found: {emb}")
                        break

                valid_embeddings = True
            
        else:
            print("No saved embeddings found. Finding embeddings now...")

        if not valid_embeddings:
            # Find an embedding the hard way. If we find one, save it to file so we can re-use it later.

            embeddings = find_multiple_embeddings(
                self.source_graph,
                self.target_graph,
                max_num_emb=NUM_PARALLEL_EMBEDDINGS,
                timeout=30
            )

            self.save_to_json_file(embeddings)

        return embeddings

    def save_to_json_file(self, embeddings: dict):
        with open(self.embeddings_path, "w") as f:
            json.dump(embeddings, f)

    def load_from_json_file(self):
        with open(self.embeddings_path, "r") as f:
            embedding_list = json.load(f)
        return embedding_list