import json
from stoic_llm.data.downloader import TextDownloader
from stoic_llm.config import DATA_DIR  # adjust if sources.json lives elsewhere

SOURCES_PATH = DATA_DIR / "config" / "sources.json"

with open(SOURCES_PATH) as f:
    sources = json.load(f)

for key, cfg in sources.items():
    print(f"\n{'='*60}")
    print(f"{cfg['author']}  ({key})")
    print(f"{'='*60}")

    downloader = TextDownloader(cfg)
    downloader.download()
    downloader.clean_gutenberg()
