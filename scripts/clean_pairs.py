"""
scripts/clean_pairs.py — strip preamble, drop refusals, sample for review

For each philosopher's neutral_pairs.json:
  1. Strip leading chat preamble from neutral_text ("Here's the modern rewrite:", etc.)
  2. Quarantine refusal/meta pairs into a separate _rejected.json for audit
  3. Write cleaned pairs to _clean.json
  4. Print a random sample of N cleaned pairs for manual reasoning-vs-style scoring

Does NOT regenerate or call any API. Pure local post-processing.
Originals are never overwritten — outputs go to new files.

Run: python scripts/clean_pairs.py
"""

import json
import re
import random
from pathlib import Path
from stoic_llm.config import PROCESSED_DIR  # adjust if needed

AUTHORS = ["marcus_aurelius", "seneca", "epictetus"]
SAMPLE_N = 15
random.seed(0)  # reproducible sample

# Preamble patterns to strip from the START of neutral_text.
# Anchored to start (after optional whitespace), case-insensitive.
PREAMBLE_PATTERNS = [
    r"^here'?s the modern rewrite:?\s*",
    r"^here is the modern rewrite:?\s*",
    r"^here'?s a modern (rewrite|version):?\s*",
    r"^here'?s the rewrite:?\s*",
    r"^here'?s my (modern )?(rewrite|version):?\s*",
    r"^modern rewrite:?\s*",
    r"^here'?s how .*?:\s*",  # "Here's how a modern person might put it:"
    r"^sure[,!]?\s*",
    r"^okay[,!]?\s*",
    r"^alright[,!]?\s*",
]

# If neutral_text contains these, the pair is a REFUSAL / meta-commentary —
# quarantine it, don't try to salvage.
REFUSAL_MARKERS = [
    "i notice",
    "i can't",
    "i cannot",
    "could you provide",
    "this passage",
    "this appears to be",
    "this is actually",
    "isn't really",
    "there isn't",
    "i'd be happy to",
    "i would be happy",
    "as an ai",
    "rather than philosophical",
    "biographical",
    "provide an actual",
]


def strip_preamble(text: str) -> tuple[str, bool]:
    """Remove leading chat preamble. Returns (cleaned, was_stripped).
    Applies patterns repeatedly in case of stacked preamble + blank lines."""
    original = text
    t = text.lstrip()
    changed = True
    while changed:
        changed = False
        for pat in PREAMBLE_PATTERNS:
            new = re.sub(pat, "", t, count=1, flags=re.IGNORECASE)
            if new != t:
                t = new.lstrip()
                changed = True
    return t, (t != original.lstrip())


def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in REFUSAL_MARKERS)


def process_author(author: str) -> dict:
    path = PROCESSED_DIR / author / "neutral_pairs.json"
    if not path.exists():
        path = PROCESSED_DIR / author / "neutral_pairs.json"
    if not path.exists():
        print(f"⚠ {author}: no neutral_pairs.json found")
        return {}

    with open(path) as f:
        pairs = json.load(f).get("pairs", [])

    clean, rejected = [], []
    stripped_count = 0

    for p in pairs:
        neutral = p.get("neutral_text", "")

        # Refusals are unsalvageable — quarantine the whole pair.
        if is_refusal(neutral):
            rejected.append({**p, "_reject_reason": "refusal/meta"})
            continue

        cleaned, was_stripped = strip_preamble(neutral)
        if was_stripped:
            stripped_count += 1

        # Guard: if stripping left almost nothing, the text WAS the preamble —
        # treat as broken and quarantine.
        if len(cleaned.strip()) < 40:
            rejected.append({**p, "_reject_reason": "empty after strip"})
            continue

        clean.append({**p, "neutral_text": cleaned})

    # Write outputs (never overwrite original)
    out_dir = path.parent
    clean_path = out_dir / "neutral_pairs_clean.json"
    reject_path = out_dir / f"neutral_pairs_rejected.json"
    with open(clean_path, "w") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    with open(reject_path, "w") as f:
        json.dump(rejected, f, indent=2, ensure_ascii=False)

    return {
        "author": author,
        "total": len(pairs),
        "clean": len(clean),
        "rejected": len(rejected),
        "stripped": stripped_count,
        "clean_pairs": clean,
        "clean_path": str(clean_path),
        "reject_path": str(reject_path),
    }


def print_sample(author: str, clean_pairs: list, n: int):
    sample = random.sample(clean_pairs, min(n, len(clean_pairs)))
    print(f"\n{'#'*70}")
    print(f"# SAMPLE FOR REVIEW — {author}  ({len(sample)} pairs)")
    print(
        f"# Score each: (1) is STOIC half real philosophy?  "
        f"(2) does NEUTRAL half reason NON-Stoically, or just restate in plainer words?"
    )
    print(f"{'#'*70}")
    for p in sample:
        print(f"\n--- id {p.get('id','?')} " + "-" * 50)
        print(f"STOIC:\n{p.get('stoic_text','')[:400]}")
        print(f"\nNEUTRAL (cleaned):\n{p.get('neutral_text','')[:400]}")


def main():
    print(f"\n{'='*70}\nPAIR CLEANING REPORT\n{'='*70}")
    summary = []
    for author in AUTHORS:
        r = process_author(author)
        if not r:
            continue
        print(f"\n{author}:")
        print(f"  total       : {r['total']}")
        print(f"  preamble stripped : {r['stripped']}")
        print(f"  rejected (refusal/empty): {r['rejected']}  → {r['reject_path']}")
        print(f"  clean kept  : {r['clean']}  → {r['clean_path']}")
        summary.append(r)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"  {'author':<18}{'total':>6}{'stripped':>10}{'rejected':>10}{'clean':>7}")
    for r in summary:
        print(
            f"  {r['author']:<18}{r['total']:>6}{r['stripped']:>10}"
            f"{r['rejected']:>10}{r['clean']:>7}"
        )

    # Print samples last so they're easy to scroll to
    for r in summary:
        print_sample(r["author"], r["clean_pairs"], SAMPLE_N)


if __name__ == "__main__":
    main()
