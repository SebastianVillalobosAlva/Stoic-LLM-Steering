import os
from stoic_llm.config import CHUNKED_DIR
from stoic_llm.data.pair_generator import NeutralPairCreator

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Set ANTHROPIC_API_KEY in .env or environment")

# Equal pair count across all three to control for data QUANTITY —
# so any cross-philosopher difference reflects pair QUALITY, not volume.
N_PAIRS = 63

# Per-author length bounds: Epictetus min_chars=150 (Enchiridion is short/aphoristic),
# others 300. max_chars=1000 for all (raise Seneca later if long essays matter).
authors = {
    "marcus_aurelius": {
        "display": "Marcus Aurelius",
        "min_chars": 300,
        "max_chars": 1000,
    },
    "seneca": {"display": "Seneca", "min_chars": 300, "max_chars": 1000},
    "epictetus": {"display": "Epictetus", "min_chars": 150, "max_chars": 1000},
}

for author, cfg in authors.items():
    print(f"\n{'='*60}")
    print(f"Generating {N_PAIRS} pairs for {cfg['display']}")
    print(f"min_chars={cfg['min_chars']}, max_chars={cfg['max_chars']}")
    print(f"{'='*60}")

    chunk_files = list((CHUNKED_DIR / author).glob("*.json"))
    if not chunk_files:
        print(f"No chunk files found for {author}, skipping")
        continue

    creator = NeutralPairCreator(
        chunks_file=chunk_files[0],
        author_name=cfg["display"],
        api_key=api_key,
    )

    # Check available chunks BEFORE generating — bail if too few for N_PAIRS
    chunks = creator.read_chunks()
    filtered = creator.filter_chunks_by_length(
        chunks["chunks"],
        min_chars=cfg["min_chars"],
        max_chars=cfg["max_chars"],
    )
    print(f"Available chunks after filtering: {len(filtered)}")
    if len(filtered) < N_PAIRS:
        print(
            f"⚠ Only {len(filtered)} chunks — fewer than {N_PAIRS} requested. "
            f"Will generate {len(filtered)}."
        )

    creator.create_pairs(
        num_pairs=N_PAIRS,
        min_chars=cfg["min_chars"],
        max_chars=cfg["max_chars"],
    )

print(f"\n{'='*60}\nDone! All pairs generated.\n{'='*60}")
