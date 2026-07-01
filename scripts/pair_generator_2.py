import os
from stoic_llm.data.pair_generator import NeutralPairCreator
from stoic_llm.config import CHUNKED_DIR  # wherever the chunked source texts live

creator = NeutralPairCreator(
    chunks_file=str(
        CHUNKED_DIR / "marcus_aurelius" / "meditations.json"
    ),  # adjust filename
    author_name="Marcus Aurelius",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

pairs = creator.create_pairs(num_pairs=5, min_chars=300, max_chars=2000)
for p in pairs:
    print(f"\n--- id {p.get('id','?')} " + "-" * 50)
    print(f"STOIC:\n{p['stoic_text'][:400]}")
    print(f"\nNEUTRAL:\n{p['neutral_text'][:400]}")
