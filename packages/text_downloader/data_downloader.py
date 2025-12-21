import urllib.request
import re

from .config import RAW_DIR, PROCESSED_DIR


class TextDownloader:
    def __init__(self, source_config):
        self.url = source_config.get("url")
        self.author = source_config.get("author")

        author_folder = source_config.get("author_folder")
        raw_author_dir = RAW_DIR / author_folder
        processed_author_dir = PROCESSED_DIR / author_folder

        raw_author_dir.mkdir(parents=True, exist_ok=True)
        processed_author_dir.mkdir(parents=True, exist_ok=True)

        self.raw_filename = raw_author_dir / source_config.get("raw_filename")
        self.clean_filename = processed_author_dir / source_config.get("clean_filename")

    def download(self):
        """Download the file from URL"""
        print(f"Downloading from {self.url}...")
        urllib.request.urlretrieve(self.url, self.raw_filename)
        print(f"✓ Saved to {self.raw_filename}")
        return self.raw_filename

    def clean_gutenberg(self):
        """Clean file from Gutenberg text add-ons"""
        with open(self.raw_filename, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Cleaning {self.raw_filename}")

        clean_text = text
        start_idx, end_idx = self._find_content_boundaries(text)
        if start_idx and end_idx:
            clean_text = text[start_idx:end_idx]

        with open(self.clean_filename, "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"✓ Saved to {self.clean_filename}")

    def _find_content_boundaries(self, text):
        """Find start and end indices of actual book content"""
        start_match = re.search(
            r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text
        )
        end_match = re.search(
            r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text
        )

        if start_match and end_match:
            return start_match.end(), end_match.start()
        return None, None
