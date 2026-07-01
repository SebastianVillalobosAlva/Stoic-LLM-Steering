"""
scripts/audit_filter.py — see what _is_non_philosophical excludes vs keeps

Runs the filter over a philosopher's chunks WITHOUT calling any API.
Prints excluded chunks (with which marker type triggered) and a sample of
kept chunks, so you can confirm it catches biographical/editorial text
without discarding real philosophy.

Run: python scripts/audit_filter.py marcus_aurelius
"""

import sys
import json
from pathlib import Path
from stoic_llm.data.pair_generator import NeutralPairCreator
from stoic_llm.config import CHUNKED_DIR  # adjust to where chunked files live

author = sys.argv[1] if len(sys.argv) > 1 else "epictetus"

# Map author dir-name -> display name (for the surname check in the filter)
DISPLAY = {
    "marcus_aurelius": "Marcus Aurelius",
    "seneca": "Seneca",
    "epictetus": "Epictetus",
}

# Build the creator just to reuse its filter logic — no API key needed,
# since we never call generate_neutral_text.
chunks_file = CHUNKED_DIR / author / "enchiridion.json"  # adjust filename
if not chunks_file.exists():
    chunks_file = CHUNKED_DIR / f"{author}_chunks.json"

creator = NeutralPairCreator(
    chunks_file=str(chunks_file),
    author_name=DISPLAY.get(author, author),
    api_key="not-needed-for-filtering",
)

chunks = creator.read_chunks()["chunks"]

MIN_CHARS, MAX_CHARS = 150, 1000  # match your create_pairs defaults

excluded = []
kept = []
too_short = []
too_long = []

for c in chunks:
    text = c["text"]
    n = len(text)
    if n < MIN_CHARS:
        too_short.append(c)
        continue
    if n > MAX_CHARS:
        too_long.append(c)
        continue
    if creator._is_non_philosophical(text):
        excluded.append(c)
    else:
        kept.append(c)

print(f"\n{'='*70}")
print(f"FILTER AUDIT — {author}")
print(f"{'='*70}")
print(f"  total chunks      : {len(chunks)}")
print(f"  too short (<{MIN_CHARS}) : {len(too_short)}")
print(f"  too long  (>{MAX_CHARS}) : {len(too_long)}")
print(f"  EXCLUDED (filter) : {len(excluded)}")
print(f"  KEPT              : {len(kept)}")

# ---- Show ALL excluded chunks — this is the critical audit ----
# You want to confirm every one of these really is biographical/editorial/
# citation/religious, NOT real philosophy wrongly nuked.
print(f"\n{'='*70}\nEXCLUDED CHUNKS (verify these are all junk)\n{'='*70}")
for c in excluded:
    print(f"\n--- id {c['id']} ({len(c['text'])} chars) " + "-" * 40)
    print(c["text"][:300])

# ---- Sample of kept chunks — confirm real philosophy survived ----
import random

random.seed(0)
sample_kept = random.sample(kept, min(8, len(kept)))
print(
    f"\n{'='*70}\nSAMPLE OF KEPT CHUNKS (confirm these are real philosophy)\n{'='*70}"
)
for c in sample_kept:
    print(f"\n--- id {c['id']} ({len(c['text'])} chars) " + "-" * 40)
    print(c["text"][:300])

#     # quick check: is the bibliography in the kept chunks?
# import json
# from stoic_llm.config import CHUNKED_DIR

# with open(CHUNKED_DIR / "epictetus" / "enchiridion.json") as f:
#     chunks = json.load(f)["chunks"]

# for c in chunks:
#     t = c["text"]
#     if any(
#         name in t
#         for name in [
#             "Dilthey",
#             "Busson",
#             "Groethuysen",
#             "Saunders",
#             "Wenley",
#             "1955",
#             "1927",
#         ]
#     ):
#         print(f"--- id {c['id']} ({len(t)} chars) — BIBLIO CONTENT ---")
#         print(t[:200])
#         print()
