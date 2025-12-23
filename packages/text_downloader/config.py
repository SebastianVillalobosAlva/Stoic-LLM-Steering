from pathlib import Path

# Base directories
DATA_DIR = Path("data")
CONFIG_DIR = DATA_DIR / "config"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKED_DIR = DATA_DIR / "chunked"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CHUNKED_DIR.mkdir(parents=True, exist_ok=True)

# Sources config file
SOURCES_CONFIG = CONFIG_DIR / "sources.json"
