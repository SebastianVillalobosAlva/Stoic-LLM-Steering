import json
import re
import random
import time
import anthropic
from pathlib import Path
from stoic_llm.config import PROCESSED_DIR, NEUTRAL_PAIR_PROMPT


class NeutralPairCreator:
    def __init__(self, chunks_file, author_name, api_key=None):
        self.chunks_file = chunks_file
        self.author_name = author_name
        self.neutral_pair_path = PROCESSED_DIR
        self.client = anthropic.Anthropic(api_key=api_key)

    def read_chunks(self):
        """Read file and its chunks"""
        chunks_file = Path(self.chunks_file)
        with open(chunks_file) as f:
            chunks = json.load(f)
        return chunks

    def _is_non_philosophical(self, text):
        """Flag chunks that are citations or biographical/editorial prose ABOUT
        the author rather than philosophy BY the author.

        NOTE: religious-vocabulary filtering removed — for a Stoic corpus,
        'God/gods/divine/Providence/nature' are core philosophical vocabulary,
        not contamination. Filtering on them amputates real Meditations content
        (e.g. Book II opening). Only biographical/editorial markers remain.
        """
        biblio_markers = [
            "pp.",
            "Vol.",
            "ISBN",
            "Published",
            "published",
            "Editor",
            "Reprinted",
            "translation",
            "emendation",
            "corrupt",
            "Casaubon",
        ]
        biographical_markers = [
            "was born",
            "his death",
            "his life",
            "his daily life",
            "we meet with",
            "translator",
            "preface",
            "biography",
            "the author",
            "his reign",
            "Faustina",
            "Commodus",
            "Hadrian",
            "A.D.",
            "B.C.",
            "born in",
            "died in",
            "rhetorician",
            "bestseller",
            "vernacular",
            "founder of",
            "Cyprus",
            "present century",
        ]

        # Citation pattern: "Word, 1933." / "New York, 1955" — city/publisher + year.
        # Catches bibliography entries regardless of length.
        has_citation = bool(re.search(r",\s+\d{4}\b", text))

        # "pp. 153-162" page ranges
        has_pages = bool(re.search(r"pp\.\s*\d+", text))

        biblio_count = sum(1 for m in biblio_markers if m in text)
        biographical_count = sum(1 for m in biographical_markers if m in text)

        surname = self.author_name.split()[-1]
        names_self = surname in text

        return (
            biblio_count >= 2
            or biographical_count >= 2
            or (names_self and biographical_count >= 1)
        )

    def filter_chunks_by_length(self, chunks, min_chars=300, max_chars=1000):
        """Keep chunks within character count range, excluding non-philosophical text."""
        return [
            c
            for c in chunks
            if min_chars <= len(c["text"]) <= max_chars
            and not self._is_non_philosophical(c["text"])
        ]

    def generate_neutral_text(
        self, stoic_text, max_tokens=1000, model="claude-sonnet-4-20250514"
    ):
        """Generate neutral version using Claude API with contrastive prompt"""
        prompt = NEUTRAL_PAIR_PROMPT.format(
            author_name=self.author_name, stoic_text=stoic_text
        )
        msg = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def save_neutral_pairs(self, pairs):
        author = Path(self.chunks_file).parent.name
        out = self.neutral_pair_path / author / "neutral_pairs.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"pairs": pairs}, f, indent=2)
        print(f"Saved {len(pairs)} pairs to {out}")

    def create_pairs(self, num_pairs=100, min_chars=300, max_chars=1000, seed=613):
        """Generate N pairs and save to file"""
        chunks = self.read_chunks()
        filtered = self.filter_chunks_by_length(
            chunks["chunks"], min_chars=min_chars, max_chars=max_chars
        )

        if len(filtered) > num_pairs:
            random.seed(seed)
            to_process = random.sample(filtered, num_pairs)
        else:
            to_process = filtered
            if len(filtered) < num_pairs:
                print(
                    f"⚠ Only {len(filtered)} chunks available (requested {num_pairs})"
                )

        pairs = []
        print(f"Found {len(filtered)} filtered chunks")
        print(f"Generating {len(to_process)} neutral pairs for {self.author_name}...\n")

        for i, chunk in enumerate(to_process, 1):
            print(f"[{i}/{len(to_process)}] Chunk {chunk['id']}...")
            try:
                original = chunk["text"]
                neutral = self.generate_neutral_text(original)
                pairs.append(
                    {
                        "id": chunk["id"],
                        "stoic_text": original,
                        "neutral_text": neutral,
                    }
                )
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
            time.sleep(0.5)

        print(f"\n✓ Generated {len(pairs)} pairs for {self.author_name}!")
        self.save_neutral_pairs(pairs)
        return pairs
