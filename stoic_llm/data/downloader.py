import urllib.request
import re
from stoic_llm.config import RAW_DIR, PROCESSED_DIR


class TextDownloader:
    def __init__(self, source_config):
        self.url = source_config.get("url")
        self.author = source_config.get("author")

        author_folder = source_config.get("author_folder")
        raw_author_dir = RAW_DIR / author_folder
        processed_author_dir = PROCESSED_DIR / author_folder

        raw_author_dir.mkdir(parents=True, exist_ok=True)
        processed_author_dir.mkdir(parents=True, exist_ok=True)

        self.raw_filename = raw_author_dir / source_config.get("filename")
        self.clean_filename = processed_author_dir / source_config.get("filename")

        # Per-author content boundaries (regex). Optional — fall back to the
        # Gutenberg license markers if not provided.
        # content_start: regex marking where the WORK PROPER begins (skips intro/bio)
        # content_end:   regex marking where appendices/notes begin (cut from here)
        self.content_start = source_config.get("content_start")
        self.content_end = source_config.get("content_end")

    def download(self):
        """Download the file from URL"""
        print(f"Downloading from {self.url}...")
        urllib.request.urlretrieve(self.url, self.raw_filename)
        print(f"✓ Saved to {self.raw_filename}")
        return self.raw_filename

    def clean_gutenberg(self):
        """Strip Gutenberg license wrapper, then narrow to the work proper
        (skip front-matter intro/biography and back-matter appendices)."""
        with open(self.raw_filename, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Cleaning {self.raw_filename}")

        start_idx, end_idx = self._find_content_boundaries(text)
        if start_idx is not None and end_idx is not None:
            clean_text = text[start_idx:end_idx]
        else:
            print("  ⚠ License markers not found — keeping full text.")
            clean_text = text

        # Sanity check: warn loudly if the slice looks wrong.
        pct = 100 * len(clean_text) / max(len(text), 1)
        print(f"  Sliced to {len(clean_text):,} chars ({pct:.0f}% of raw)")
        if pct < 20:
            print(
                "  ⚠⚠ WARNING: kept <20% of file — a boundary marker may have "
                "mismatched and truncated the work. Verify before chunking."
            )

        with open(self.clean_filename, "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"✓ Saved to {self.clean_filename}")

    def _find_content_boundaries(self, text):
        """Three-stage boundary detection:
        1. Gutenberg license wrapper (always).
        2. content_start regex (optional) — skip intro/biography at the front.
        3. content_end regex (optional) — cut appendices/notes at the back.
        Stages 2/3 operate INSIDE the license-stripped body and only apply
        if the per-author regex is provided AND matches.
        """
        start_match = re.search(
            r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text
        )
        end_match = re.search(
            r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text
        )
        if not (start_match and end_match):
            return None, None

        inner_start = start_match.end()
        inner_end = end_match.start()
        body = text[inner_start:inner_end]

        # Stage 2 — front boundary (skip intro/biography)
        start_offset = 0
        if self.content_start:
            m = re.search(self.content_start, body, re.IGNORECASE)
            if m:
                start_offset = m.start()
            else:
                print(
                    f"  ⚠ content_start /{self.content_start}/ not matched — "
                    "keeping from start of body."
                )

        # Stage 3 — back boundary (cut appendix/notes), searched AFTER start
        end_offset = len(body)
        if self.content_end:
            tail = body[start_offset:]
            m = re.search(self.content_end, tail, re.IGNORECASE)
            if m:
                end_offset = start_offset + m.start()
            else:
                print(
                    f"  ⚠ content_end /{self.content_end}/ not matched — "
                    "keeping to end of body."
                )

        return inner_start + start_offset, inner_start + end_offset
