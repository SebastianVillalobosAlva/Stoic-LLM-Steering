from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# Model Config
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cpu"

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = DATA_DIR / "config"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKED_DIR = DATA_DIR / "chunked"
VECTORS_DIR = DATA_DIR / "steering_vectors"
LORA_TRAINING_DIR = DATA_DIR / "lora_training"

# Model Paths
MODELS_DIR = PROJECT_ROOT / "models"

# Results Paths
RESULTS_DIR = PROJECT_ROOT / "results"
SWEEPS_DIR = RESULTS_DIR / "sweeps"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"
JUDGES_DIR = RESULTS_DIR / "judges"

# Sources Config
SOURCES_CONFIG = CONFIG_DIR / "sources.json"

# Steering Defaults
LAYER_IDX = 12
COEFFICIENT = 0.11
TEMPERATURE = 0.7
MAX_TOKENS = 100
DEFAULT_PROMPTS = [
    "When facing difficulty, one should",
    "The nature of virtue is",
    "To live well means",
]

# Create Directories
for d in [
    RAW_DIR,
    PROCESSED_DIR,
    CONFIG_DIR,
    CHUNKED_DIR,
    VECTORS_DIR,
    LORA_TRAINING_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SWEEPS_DIR,
    COMPARISONS_DIR,
    JUDGES_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
