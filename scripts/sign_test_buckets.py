"""Per-concept and per-stance sign tests on dilemma deltas.

Usage:
    python scripts/sign_test_buckets.py \
        results/dilemmas-v2/lora/dilemma_eval_<timestamp>.json \
        data/config/dilemmas_v2.json
"""

import json
import sys
from math import comb


def sign_test(deltas: list[float], tol: float = 1e-9) -> dict:
    pos = sum(1 for d in deltas if d > tol)
    neg = sum(1 for d in deltas if d < -tol)
    n = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "n": 0, "p": 1.0}
    k = max(pos, neg)
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2**n * 2
    return {"pos": pos, "neg": neg, "n": n, "p": min(p, 1.0)}


def main(results_path: str, dilemmas_path: str) -> None:
    with open(results_path) as f:
        results = json.load(f)
    with open(dilemmas_path) as f:
        dilemmas = {d["id"]: d for d in json.load(f)["dilemmas"]}

    for name, r in results["philosophers"].items():
        deltas = r.get("per_item_delta_logit", r["per_item_delta"])
        print(f"\n=== {name} ===")
        for key in ("concept", "stoic_stance"):
            print(f"  by {key}:")
            buckets: dict[str, list[float]] = {}
            for item_id, d in deltas.items():
                buckets.setdefault(dilemmas[item_id][key], []).append(d)
            for b, ds in sorted(buckets.items()):
                s = sign_test(ds)
                mean = sum(ds) / len(ds)
                print(
                    f"    {b:<26} {s['pos']:>2}+/{s['neg']:<2}- of {s['n']:<2}"
                    f"  mean Δlo {mean:+.3f}  p={s['p']:.3f}"
                )
    print(
        "\nNote: concept buckets are n=3-6 items — read direction and mean, "
        "not p-values; nothing this small clears significance alone."
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
