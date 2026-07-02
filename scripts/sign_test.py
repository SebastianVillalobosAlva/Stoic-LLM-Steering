"""Sign test on per-item dilemma deltas. Reads the saved Exp 11 results JSON.

Usage: python scripts/sign_test.py results/dilemmas-v2/lora/dilemma_eval_<timestamp>.json
"""

import json
import sys
from math import comb


def sign_test(deltas: dict[str, float], tol: float = 1e-9) -> dict:
    pos = sum(1 for d in deltas.values() if d > tol)
    neg = sum(1 for d in deltas.values() if d < -tol)
    ties = len(deltas) - pos - neg
    n = pos + neg  # ties dropped, standard practice
    # exact two-sided binomial p under H0: P(item moves +) = 0.5
    k = max(pos, neg)
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2**n * 2
    return {"pos": pos, "neg": neg, "ties": ties, "n": n, "p_two_sided": min(p, 1.0)}


def main(path: str) -> None:
    with open(path) as f:
        results = json.load(f)

    print(f"{'philosopher':<12} {'+':>4} {'-':>4} {'ties':>5} {'p (2-sided)':>12}")
    for name, r in results["philosophers"].items():
        # log-odds deltas if present, else P-space (sign is identical)
        deltas = r.get("per_item_delta_logit", r["per_item_delta"])
        s = sign_test(deltas)
        print(
            f"{name:<12} {s['pos']:>4} {s['neg']:>4} {s['ties']:>5} {s['p_two_sided']:>12.4f}"
        )

    print("\nRead: for n=40 with no ties, ~27+/40 in one direction gives p<0.05.")
    print(
        "Sign test ignores magnitude — it asks only 'did most items move the same way'."
    )


if __name__ == "__main__":
    main(sys.argv[1])
