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
import os

import networkx as nx
from dwave.embedding import verify_embedding
from minorminer.utils.parallel_embeddings import find_multiple_embeddings

from demo_configs import EMBEDDING_TIMEOUT, NUM_PARALLEL_EMBEDDINGS
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
            problem_type: The type of problem for which embeddings are being managed. Currently
                only the default problem type is supported, but more can be added in the future.
            embeddings_dir: The directory where embeddings are saved. Defaults to EMBEDDINGS_PATH.
        """

        self.qpu_name = qpu_name
        self.problem_type = problem_type
        self.embedding_id = f"{qpu_name}_{problem_type.name}"
        self.embeddings_path = os.path.join(embeddings_dir, f"{self.embedding_id}.json")
        self.max_size_embeddings_path = os.path.join(
            embeddings_dir, f"{self.embedding_id}_max_size.json"
        )
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

    def get_embeddings(
        self,
        num_parallel_embeddings: int | None = NUM_PARALLEL_EMBEDDINGS,
        timeout: int = EMBEDDING_TIMEOUT,
    ) -> list[dict]:
        """Get the embeddings for the selected QPU. Checks for saved embeddings in file first,
        if none are saved or they are invalid or incomplete, finds new embeddings and saves
        them to file before returning them.

        Args:
            num_parallel_embeddings: The number of parallel embeddings to find. If None, finds
                the maximum number of embeddings possible within the allotted time.
            timeout: The maximum time to wait for embeddings to be found.

        Returns:
            A list of embeddings, each a dict mapping source node to target qubit.
        """

        get_max_embeddings = num_parallel_embeddings is None

        embeddings_to_return = []
        valid_loaded_embeddings = []
        found_embeddings = []

        if get_max_embeddings:
            loading_path = self.max_size_embeddings_path
            num_embeddings_needed = self.source_graph.number_of_nodes()
        else:
            loading_path = self.embeddings_path
            num_embeddings_needed = num_parallel_embeddings

        if os.path.exists(loading_path):
            print("Loading saved embeddings from file...")
            loaded_embeddings = self.load_from_json_file(maximal=get_max_embeddings)

            for emb in loaded_embeddings:
                emb = {int(k): tuple(v) for k, v in emb.items()}
                try:
                    verify_embedding(emb, self.source_graph, self.target_graph)
                    valid_loaded_embeddings.append(emb)
                except Exception:
                    print(f"Invalid embedding found: {emb}")

            # If we're loading a max-size embedding set, we don't know how many embeddings we
            # actually need, so checking that a large enough fraction of the loaded embeddings
            # are valid is the best we can do.
            if get_max_embeddings:
                num_embeddings_needed = max(1, int(0.8 * len(loaded_embeddings)))

            if len(valid_loaded_embeddings) > 0:
                print(f"Loaded {len(valid_loaded_embeddings)} embeddings from file.")
                embeddings_to_return = valid_loaded_embeddings

        else:
            print("No saved embeddings found.")

        if len(embeddings_to_return) < num_embeddings_needed:
            print("Not enough valid embeddings found.")
            print("Finding embeddings now...")
            # Find embeddings the hard way, then save them to file so we can re-use them later.

            new_embeddings = find_multiple_embeddings(
                self.source_graph,
                self.target_graph,
                max_num_emb=num_parallel_embeddings,
                timeout=timeout,
            )

            found_embeddings = [{int(k): (v,) for k, v in emb.items()} for emb in new_embeddings]

            print(f"Found {len(found_embeddings)} new embeddings.")

            if found_embeddings:
                self.save_to_json_file(found_embeddings, maximal=get_max_embeddings)

            embeddings_to_return = found_embeddings

        if len(embeddings_to_return) < num_embeddings_needed:
            raise RuntimeError(
                f"Could not find enough valid embeddings. Needed {num_embeddings_needed}, "
                f"but only found {len(embeddings_to_return)}."
            )

        return embeddings_to_return

    def save_to_json_file(self, embeddings: list[dict], maximal: bool = False):
        """Save the embeddings to a JSON file for later re-use.

        Args:
            embeddings: A list of embeddings, each a dict mapping source node to target chain.
            maximal: Whether to save the embeddings to the max size file. Defaults to False.
        """

        filepath = self.max_size_embeddings_path if maximal else self.embeddings_path
        with open(filepath, "w") as f:
            json.dump(embeddings, f)

    def load_from_json_file(self, maximal: bool = False) -> list[dict]:
        """Load saved embeddings from file. This is used to retrieve previously-found embeddings
        and avoid having to find them again.

        Args:
            maximal: Whether to load the embeddings from the max size file. Defaults to False.

        Returns:
            A list of embeddings, each a dict mapping source node to target qubit.
        """

        filepath = self.max_size_embeddings_path if maximal else self.embeddings_path
        with open(filepath, "r") as f:
            embedding_list = json.load(f)
        return embedding_list
