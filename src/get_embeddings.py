import networkx as nx
from src.demo_enums import ProblemType, ModelType

from demo_configs import NUM_PARALLEL_EMBEDDINGS
from src.qpu_resources import get_solvers
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from src.data_loading import embeddings_to_json_file, ProblemModel

def get_graph_from_model(model_dict: dict) -> nx.Graph:
    """
    Get the graph representation of a model based on the problem type and model type.

    Args:
        problem_type (ProblemType): The type of problem.
        model_type (str): The type of model.

    Returns:
        networkx.Graph: The graph representation of the model.
    """

    G = nx.Graph()
    G.add_nodes_from(model_dict['h'].keys())
    G.add_edges_from(model_dict['J'].keys())
    return G

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
        model_dicts = ProblemModel.load_all_models_from_file(as_dicts=True)
        source_graph = None
        for model_name, model in model_dicts.values():
            if model_name.startswith(problem_type.name):
                source_graph = get_graph_from_model(model)
                break
        if source_graph is None:
            raise ValueError(f"No model found for problem type {problem_type.name}")

        target_graph = qpu.to_networkx_graph()

        embeddings = find_multiple_embeddings(
            source_graph,
            target_graph,
            max_num_emb=NUM_PARALLEL_EMBEDDINGS,
            timeout=30,
        )

        embeddings_to_json_file(embeddings[0], qpu_name, problem_type)

        if embeddings is None:
            raise ValueError(f"No embedding found for QPU {qpu_name}")
        return embeddings
