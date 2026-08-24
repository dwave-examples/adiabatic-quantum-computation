import os

ENERGY_THRESHOLD = 1e-8 # Relative to ground state energy

REPO_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
STATIC_PATH = os.path.join(REPO_PATH, "static")
EMBEDDINGS_PATH = os.path.join("embeddings")
MODELS_PATH = os.path.join(STATIC_PATH, "models")

if not os.path.exists(EMBEDDINGS_PATH):
    os.makedirs(EMBEDDINGS_PATH)

