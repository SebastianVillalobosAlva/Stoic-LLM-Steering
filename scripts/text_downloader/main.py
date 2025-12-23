import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from packages.text_downloader.config import SOURCES_CONFIG
from packages.text_downloader.data_downloader import TextDownloader
from packages.text_downloader.data_processer import TextProcessor


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


if __name__ == "__main__":
    main()
