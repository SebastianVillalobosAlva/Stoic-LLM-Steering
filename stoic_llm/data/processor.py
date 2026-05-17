import json
from pathlib import Path
from .config import CHUNKED_DIR, PROCESSED_DIR


class TextProcessor:
    def __init__(self, text_path=None):
        """
        Args:
            text_path: Specific file to process, or None to process all
        """
        self.text_path = Path(text_path) if text_path else None

    def chunk_by_paragraph(self):
        """Chunk text(s) into paragraphs"""
        if self.text_path:
            # Process single file
            author = self.text_path.parent.name
            file_name = str(self.text_path.name).replace(".txt", "")
            self._chunk_single_file(author, self.text_path, file_name)
        else:
            # Process all files in PROCESSED_DIR
            all_files = list(PROCESSED_DIR.rglob("*.txt"))
            for file_path in all_files:
                author = file_path.parent.name
                file_name = str(file_path.name).replace(".txt", "")
                self._chunk_single_file(author, file_path, file_name)

    def _chunk_single_file(self, author, file_path, file_name):
        """Chunk a single file and save to CHUNKED_DIR"""
        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split by paragraphs
        paragraphs = text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks_data = {
            "source_file": str(file_path),
            "author": author,
            "total_chunks": len(paragraphs),
            "chunks": [{"id": i, "text": para} for i, para in enumerate(paragraphs, 1)],
        }

        # Save chunks to CHUNKED_DIR/author_name/chunk_001.txt, etc.
        chunked_file_path = CHUNKED_DIR / author / f"{file_name}.json"
        chunked_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(chunked_file_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(paragraphs)} chunks to {chunked_file_path}")
