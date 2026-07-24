
from demo_configs import NUM_PARALLEL_EMBEDDINGS
from demo_configs import DEFAULT_GRAPH
from src.qpu_resources import get_solvers
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from src.demo_enums import ProblemType
from src.data_loading import embedding_to_json_file

def get_embeddings(
        qpu_name: str, 
        embeddings_dict: dict, 
        problem_type: ProblemType = ProblemType.DEFAULT
        ) -> dict:
    """
    Get the embedding for the selected QPU.

    Args:
        qpu_name (str): The name of the QPU.
        embeddings_dict (dict): A dictionary containing embeddings for different QPUs.

    Returns:
        dict: The embedding for the selected QPU.
    """
    embedding_id = (qpu_name, problem_type.name)
    if embedding_id in embeddings_dict:
        return embeddings_dict[embedding_id]
    else:
        # Find an embedding the hard way. If we find one, save it to file so we can re-use it later.
        qpu = get_solvers()[qpu_name]
        target_graph = qpu.to_networkx_graph()

        embeddings = find_multiple_embeddings(
            DEFAULT_GRAPH,
            target_graph,
            max_num_emb=NUM_PARALLEL_EMBEDDINGS,
            timeout=30,
        )

        embedding_to_json_file(embeddings[0], qpu_name, problem_type)

        if embeddings is None:
            raise ValueError(f"No embedding found for QPU {qpu_name}")
        return embeddings
