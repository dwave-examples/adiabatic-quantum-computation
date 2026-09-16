import os

ENERGY_THRESHOLD = 1e-8  # Relative to ground state energy

REPO_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
STATIC_PATH = os.path.join(REPO_PATH, "static")
EMBEDDINGS_PATH = os.path.join("embeddings")
MODELS_PATH = os.path.join(STATIC_PATH, "models")
SCHEDULES_PATH = os.path.join(STATIC_PATH, "annealing_schedules")

if not os.path.exists(EMBEDDINGS_PATH):
    os.makedirs(EMBEDDINGS_PATH)

ENERGY_TIME_RESCALING = { # Used to rescale energy axis for plotting samples from
    "Advantage2_system5": 2.48, # processors that have different energy scales.
    "Advantage_system4": 1.09,
}

Y_AXIS_RESCALING = {
    "Advantage2_system5": -1.9,
}
