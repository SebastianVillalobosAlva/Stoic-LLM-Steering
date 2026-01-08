import os
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from packages.text_downloader.config import SOURCES_CONFIG, CHUNKED_DIR
from packages.text_downloader.data_downloader import TextDownloader
from packages.text_downloader.data_processer import TextProcessor
from packages.text_downloader.neutral_pair_creator import NeutralPairCreator


def main():
    with open(SOURCES_CONFIG) as f:
        sources = json.load(f)

    for book_key, book_config in sources.items():
        print(f"Processing {book_config['author']}...")
        downloader = TextDownloader(book_config)
        downloader.download()
        downloader.clean_gutenberg()

    processor = TextProcessor()
    processor.chunk_by_paragraph()

    print("\n=== Generating Neutral Pairs ===")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not set, skipping neutral pair generation")
        return
    chunk_files = list(CHUNKED_DIR.rglob("*.json"))
    for chunk_file in chunk_files:
        print(f"\nGenerating pairs for {chunk_file}...")
        creator = NeutralPairCreator(chunk_file, api_key=api_key)
        creator.create_pairs(num_pairs=30)  # Generate 30 pairs per book


if __name__ == "__main__":
    main()
