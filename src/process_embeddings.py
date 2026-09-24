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

import json
import os

import networkx as nx
from dwave.embedding import verify_embedding
from minorminer.utils.parallel_embeddings import find_multiple_embeddings

from demo_configs import NUM_PARALLEL_EMBEDDINGS
from src.demo_enums import ProblemType
from src.model_loading import ProblemModel
from src.values import EMBEDDINGS_PATH


class EmbeddingSet:
    """This class finds, saves and manages embeddings for a given problem type and QPU."""

    def __init__(
        self,
        qpu_name: str,
        target_graph: nx.Graph,
        problem_type: ProblemType = ProblemType.DEFAULT,
        embeddings_dir: str = EMBEDDINGS_PATH,
    ):
        """Initializes the EmbeddingSet for a particular QPU. The QPU name and graph must
        both be provided. Currently the demo only supports a single problem graph structure,
        so it's not necessary to provide a problem type, but if more graph structures are added
        they should be easy to integrate.

        Args:
            qpu_name: The name of the QPU for which embeddings are being managed
            target_graph: The graph structure of the QPU for which embeddings are being managed
            problem_type: The type of problem for which embeddings are being managed.
            embeddings_dir: The directory where embeddings are saved. Defaults to EMBEDDINGS_PATH.
        """

        self.qpu_name = qpu_name
        self.problem_type = problem_type
        self.embedding_id = f"{qpu_name}_{problem_type.name}"
        self.embeddings_path = os.path.join(embeddings_dir, f"{self.embedding_id}.json")
        self.source_graph = self.get_source_graph()
        self.target_graph = target_graph

    def get_source_graph(self) -> nx.Graph:
        """Retrieves the source graph for the problem type from the saved model data."""
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

    def get_embeddings(self) -> list[dict]:
        """Get the embeddings for the selected QPU. Checks for saved embeddings in file first,
        if none are saved or they are invalid or incomplete, finds new embeddings and saves
        them to file before returning them.

        Returns:
            A list of embeddings, each a dict mapping source node to target qubit.
        """

        valid_embeddings = []

        if os.path.exists(self.embeddings_path):
            print("Loading saved embeddings from file...")
            loaded_embeddings = self.load_from_json_file()

            for emb in loaded_embeddings:
                emb = {int(k): tuple(v) for k, v in emb.items()} 
                try:
                    verify_embedding(emb, self.source_graph, self.target_graph)
                    valid_embeddings.append(emb)
                except Exception:
                    print(f"Invalid embedding found: {emb}")

            if len(valid_embeddings) > 0:
                print(f"Loaded {len(valid_embeddings)} embeddings from file.")

        else:
            print("No saved embeddings found.")

        if len(valid_embeddings) < NUM_PARALLEL_EMBEDDINGS:
            print(f"Require {NUM_PARALLEL_EMBEDDINGS - len(valid_embeddings)} embeddings for full embedding set.")
            print("Finding embeddings now...")
            num_needed = NUM_PARALLEL_EMBEDDINGS - len(valid_embeddings)
            print(f"Finding {num_needed} new embeddings...")
            # Find an embedding the hard way. If we find one, save it to file so we can re-use it later.

            new_embeddings = find_multiple_embeddings(
                self.source_graph,
                self.target_graph,
                max_num_emb=num_needed,
                timeout=30,
            )

            valid_embeddings.extend([{int(k): (v,) for k, v in emb.items()} for emb in new_embeddings])

            self.save_to_json_file(valid_embeddings)

        if len(valid_embeddings) < NUM_PARALLEL_EMBEDDINGS:
            raise RuntimeError(
                f"Could not find enough valid embeddings. Needed {NUM_PARALLEL_EMBEDDINGS}, but only found {len(valid_embeddings)}."
            )

        return valid_embeddings

    def save_to_json_file(self, embeddings: dict):
        """Save the embeddings to a JSON file for later re-use.
        
        Args:
            embeddings: A list of embeddings, each a dict mapping source node to target chain.
        """

        with open(self.embeddings_path, "w") as f:
            json.dump(embeddings, f)

    def load_from_json_file(self) -> list[dict]:
        """Load saved embeddings from file. This is used to retrieve previously-found embeddings
        and avoid having to find them again.
        
        Returns:
            A list of embeddings, each a dict mapping source node to target qubit.
        """

        with open(self.embeddings_path, "r") as f:
            embedding_list = json.load(f)
        return embedding_list
