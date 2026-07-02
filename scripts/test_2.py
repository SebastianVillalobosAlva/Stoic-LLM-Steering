# LEGACY (Exp 9 prep): one-off migration, task complete. Produced the
# current neutral_pairs.json (N=53) from neutral_pairs_63.json for Marcus
# and Seneca. Retained for the record of how N=53 was derived; rerunning
# would overwrite the current clean pair files.
"""
scripts/trim_pairs.py — sample 53 pairs from Marcus & Seneca's neutral_pairs.json
"""

import json
import random
from stoic_llm.config import PROCESSED_DIR

N = 53
SEED = 0

for author in ["marcus_aurelius", "seneca"]:
    path = PROCESSED_DIR / author / "neutral_pairs_63.json"
    with open(path) as f:
        pairs = json.load(f).get("pairs")

    random.seed(SEED)
    sampled = random.sample(pairs, N)

    out = PROCESSED_DIR / author / "neutral_pairs.json"
    with open(out, "w") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"{author}: {len(pairs)} → {len(sampled)} → {out}")
