from pathlib import Path

# Model config
MODEL_NAME = "meta-llama/Llama-3.2-1B"  # Base model, not Instruct
DEVICE = "mps"  # Apple Silicon

# Data paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTORS_DIR = PROJECT_ROOT / "data" / "steering_vectors"

# Create directories
VECTORS_DIR.mkdir(parents=True, exist_ok=True)
