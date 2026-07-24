import json
import os
from src.demo_enums import ProblemType, ModelType
from src.values import EMBEDDINGS_PATH, MODELS_PATH

def model_to_json_file(model_type: ModelType, h: dict, J: dict, EGS: float, problem_type: ProblemType):
    model_dict = {"h": h, "J": J, "EGS": EGS}
    pre_saved_models = os.listdir(MODELS_PATH)
    for model_filename in pre_saved_models:
        if model_filename.startswith(problem_type.name):
            with open(os.path.join(MODELS_PATH, model_filename), "r") as f:
                existing_model = json.load(f)
                existing_model_nodes = set(existing_model["h"].keys())
                existing_model_edges = set(existing_model["J"].keys())
                model_nodes = set(h.keys())
                model_edges = set(J.keys())
                if existing_model_nodes != model_nodes or existing_model_edges != model_edges:
                    raise ValueError(
                        f"Model {problem_type.name} already exists with different nodes or edges."
                        )
                else:
                    break 

    model_filename = f"{problem_type.name}_{model_type.name}.json"
    model_path = os.path.join(MODELS_PATH, model_filename)
    with open(model_path, "w") as f:
        json.dump(model_dict, f)

def model_from_json_file(json_filename):
    model_path = os.path.join(MODELS_PATH, json_filename)
    with open(model_path, "r") as f:
        model_dict = json.load(f)
    return model_dict

def load_all_models_from_file():
    models_dict = {}
    for model_filename in os.listdir(MODELS_PATH):
        if model_filename.endswith(".json"):
            model_dict = model_from_json_file(model_filename)
            name_pieces = model_filename.split("_")
            problem_type_name = name_pieces[0]
            model_type_name = name_pieces[1].replace(".json", "")
            models_dict[(problem_type_name, model_type_name)] = model_dict
    return models_dict

def embeddings_to_json_file(embeddings: dict, solver_graph_id: str, problem_type: ProblemType):
    embedding_filname = f"{solver_graph_id}_{problem_type.name}_embedding.json"
    embedding_path = os.path.join(EMBEDDINGS_PATH, embedding_filname)
    with open(embedding_path, "w") as f:
        json.dump(embeddings, f)

def embedding_from_json_file(json_filename):
    model_path = os.path.join(EMBEDDINGS_PATH, json_filename)
    with open(model_path, "r") as f:
        embedding_dict = json.load(f)
    name_pieces = json_filename.split("_")
    solver_graph_id = name_pieces[0]
    problem_type_name = name_pieces[1].replace(".json", "")
    return embedding_dict, solver_graph_id, problem_type_name

def load_all_embeddings_from_file():
    embeddings_dict = {}
    for embedding_filename in os.listdir(EMBEDDINGS_PATH):
        if embedding_filename.endswith(".json"):
            emb_dict, solver_graph_id, problem_type = embedding_from_json_file(embedding_filename)
            embeddings_dict[(solver_graph_id, problem_type)] = emb_dict
    return embeddings_dict

