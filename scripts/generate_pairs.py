import os
from stoic_llm.config import CHUNKED_DIR
from stoic_llm.data.pair_generator import NeutralPairCreator

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Set ANTHROPIC_API_KEY in .env or environment")

# Config per philosopher
# Epictetus gets lower min_chars (200) because the Enchiridion is short/aphoristic
authors = {
    "marcus_aurelius": {
        "display": "Marcus Aurelius",
        "num_pairs": 100,
        "min_chars": 300,
    },
    "seneca": {"display": "Seneca", "num_pairs": 100, "min_chars": 300},
    "epictetus": {"display": "Epictetus", "num_pairs": 58, "min_chars": 200},
}

for author, cfg in authors.items():
    print(f"\n{'='*60}")
    print(f"Generating pairs for {cfg['display']}")
    print(f"Target: {cfg['num_pairs']} pairs, min_chars={cfg['min_chars']}")
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

    # Check how many chunks survive filtering
    chunks = creator.read_chunks()
    filtered = creator.filter_chunks_by_length(
        chunks["chunks"],
        min_chars=cfg["min_chars"],
        max_chars=1000,
    )
    print(f"Available chunks after filtering: {len(filtered)}")

    # create_pairs handles sampling, generation, and saving
    creator.create_pairs(num_pairs=cfg["num_pairs"], min_chars=cfg["min_chars"])

print(f"\n{'='*60}")
print("Done! All pairs generated.")
print(f"{'='*60}")
