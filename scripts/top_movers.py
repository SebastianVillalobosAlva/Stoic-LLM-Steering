"""Print Seneca's biggest-moving dilemmas, largest first.

Usage:
    python scripts/top_movers.py results/dilemmas-v2/lora/dilemma_eval_<timestamp>.json data/config/dilemmas_v2.json
"""

import json
import sys

results_path, dilemmas_path = sys.argv[1], sys.argv[2]

with open(results_path) as f:
    results = json.load(f)
with open(dilemmas_path) as f:
    dilemmas = {d["id"]: d for d in json.load(f)["dilemmas"]}

deltas = results["philosophers"]["seneca"]["per_item_delta_logit"]
top = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]

for item_id, d in top:
    dil = dilemmas[item_id]
    print(f"{item_id}  Δlo {d:+.3f}  [{dil['concept']} / {dil['stoic_stance']}]")
    print(f"  situation: {dil['situation']}")
    print(f"  stoic:     {dil['stoic']}")
    print(f"  nonstoic:  {dil['nonstoic']}\n")
