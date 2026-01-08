import json
import random
import time
import anthropic
from pathlib import Path
from .config import PROCESSED_DIR


class NeutralPairCreator:
    def __init__(self, chunks_file, api_key=None):
        self.chunks_file = chunks_file
        self.neutral_pair_path = PROCESSED_DIR
        self.client = anthropic.Anthropic(api_key=api_key)

    def read_chunks(self):
        """Read file and its chunks"""
        chunks_file = Path(self.chunks_file)
        with open(chunks_file) as f:
            chunks = json.load(f)
        return chunks

    def _is_bibliography(self, text):
        """Check if chunk is likely a citation or bibliography"""
        biblio_markers = [
            "pp.",
            "Vol.",
            "ed.",
            "Trans.",
            "Chapter",
            "Chap.",
            "ISBN",
            "Published",
            "Editor",
            "Reprinted",
        ]
        # Check if it's very citation-heavy (multiple markers)
        marker_count = sum(1 for marker in biblio_markers if marker in text)
        return marker_count >= 2

    def filter_chunks_by_length(self, chunks, min_chars=300, max_chars=1000):
        """Keep chunks within character count range"""
        filtered = []
        for chunk in chunks:
            char_count = len(chunk["text"])
            if min_chars <= char_count <= max_chars and not self._is_bibliography(
                chunk["text"]
            ):
                filtered.append(chunk)
        return filtered

    def generate_neutral_text(
        self, stoic_text, max_tokens=500, model="claude-sonnet-4-20250514"
    ):
        """Generate neutral version using Claude API"""
        message = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": f"Rewrite this philosophical text in plain, neutral language without any poetic or philosophical style. Keep the same meaning but make it straightforward:\n\n{stoic_text}",
                }
            ],
        )

        neutral = message.content[0].text
        return neutral

    def save_neutral_pairs(self, pairs):
        author = Path(self.chunks_file).parent.name
        self.neutral_pair_path = self.neutral_pair_path / author / "neutral_pairs.json"

        with open(self.neutral_pair_path, "w") as f:
            json.dump({"pairs": pairs}, f, indent=2)
        print(f"Saved to {self.neutral_pair_path}")

    def create_pairs(self, num_pairs=30):
        """Generate N pairs and save to file"""
        chunks = self.read_chunks()
        filtered_chunks = self.filter_chunks_by_length(chunks["chunks"])
        if len(filtered_chunks) > num_pairs:
            chunks_to_process = random.sample(filtered_chunks, num_pairs)
        else:
            chunks_to_process = filtered_chunks
        pairs = []

        print(f"Found {len(filtered_chunks)} filtered chunks")
        print(f"Generating {len(chunks_to_process)} neutral pairs...\n")

        for i, chunk in enumerate(chunks_to_process, 1):
            print(
                f"Processing chunk {i}/{len(chunks_to_process)} (ID: {chunk['id']})..."
            )
            original = chunk["text"]
            neutral = self.generate_neutral_text(original)
            pairs.append(
                {"id": chunk["id"], "stoic_text": original, "neutral_text": neutral}
            )
            time.sleep(0.5)

        print(f"\n✓ Generated {len(pairs)} pairs!")
        self.save_neutral_pairs(pairs)
