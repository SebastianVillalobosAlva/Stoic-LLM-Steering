import json
import re
from pathlib import Path
from stoic_llm.config import PROCESSED_DIR  # adjust if your pairs live elsewhere

# Markers grouped by failure type (lowercased matching)
PREAMBLE = [
    "here's the modern",
    "here is the modern",
    "here's a modern",
    "here's the rewrite",
    "modern rewrite:",
    "here's my",
    "sure,",
    "look,",
    "okay,",
    "alright,",
]

REFUSAL_META = [
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
]

AUTHORS = ["marcus_aurelius", "seneca", "epictetus"]


def find_markers(text: str, markers: list) -> list:
    """Return which markers appear in text (lowercased substring match)."""
    t = text.lower()
    return [m for m in markers if m in t]


def check_author(author: str) -> dict:
    path = PROCESSED_DIR / author / "neutral_pairs.json"
    if not path.exists():
        # try flat layout: processed/{author}_neutral_pairs.json
        path = PROCESSED_DIR / author / "neutral_pairs.json"
    if not path.exists():
        print(f"⚠ {author}: no neutral_pairs.json found (looked in {path})")
        return {}

    with open(path) as f:
        pairs = json.load(f).get("pairs", [])

    total = len(pairs)
    preamble_hits = []
    refusal_hits = []
    clean = 0

    for p in pairs:
        neutral = p.get("neutral_text", "")
        pre = find_markers(neutral, PREAMBLE)
        ref = find_markers(neutral, REFUSAL_META)
        if pre:
            preamble_hits.append((p.get("id", "?"), pre, neutral[:80]))
        if ref:
            refusal_hits.append((p.get("id", "?"), ref, neutral[:80]))
        if not pre and not ref:
            clean += 1

    return {
        "author": author,
        "total": total,
        "preamble": preamble_hits,
        "refusal": refusal_hits,
        "clean": clean,
        "path": str(path),
    }


def main():
    print(f"\n{'='*70}")
    print("CONTRASTIVE PAIR CONTAMINATION REPORT")
    print(f"{'='*70}")

    summary = []
    for author in AUTHORS:
        r = check_author(author)
        if not r:
            continue

        total = r["total"]
        n_pre = len(r["preamble"])
        n_ref = len(r["refusal"])
        # a pair can hit both; count distinct contaminated
        contaminated_ids = {x[0] for x in r["preamble"]} | {x[0] for x in r["refusal"]}
        n_contam = len(contaminated_ids)

        print(f"\n{'-'*70}")
        print(f"{author}  ({total} pairs)  —  {r['path']}")
        print(f"  preamble leakage : {n_pre:>3}  ({100*n_pre/total:.0f}%)")
        print(f"  refusal/meta     : {n_ref:>3}  ({100*n_ref/total:.0f}%)")
        print(
            f"  contaminated     : {n_contam:>3}  ({100*n_contam/total:.0f}%)  ← distinct pairs"
        )
        print(f"  clean            : {r['clean']:>3}  ({100*r['clean']/total:.0f}%)")

        # show a few worst examples
        if r["refusal"]:
            print(f"\n  sample refusals/meta:")
            for pid, markers, preview in r["refusal"][:3]:
                print(f"    id {pid}: {markers} — {preview!r}")
        if r["preamble"]:
            print(f"\n  sample preamble:")
            for pid, markers, preview in r["preamble"][:3]:
                print(f"    id {pid}: {markers} — {preview!r}")

        summary.append((author, total, n_contam, 100 * n_contam / total))

    # cross-philosopher summary — the key table
    print(f"\n{'='*70}")
    print("SUMMARY — contamination rate by philosopher")
    print(f"{'='*70}")
    print(f"  {'author':<18} {'pairs':>6} {'contaminated':>13} {'rate':>7}")
    for author, total, n_contam, rate in summary:
        print(f"  {author:<18} {total:>6} {n_contam:>13} {rate:>6.0f}%")
    print()


if __name__ == "__main__":
    main()
