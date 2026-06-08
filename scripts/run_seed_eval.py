import json
from stoic_llm.model import ModelLoader
from stoic_llm.eval.sweep import SteeringSweep
from stoic_llm.config import VECTORS_DIR, SWEEPS_DIR

loader = ModelLoader("3B")
model, tokenizer = loader.load()


def make_sweep(author):
    return SteeringSweep(
        model=model,
        tokenizer=tokenizer,
        vector_path=str(VECTORS_DIR / f"{author}_steering_3B.pt"),
    )


# Per-philosopher top content layers from the sweeps (coeff fixed at 0.11)
CONFIGS = {
    "marcus_aurelius": [26, 16, 20],
    "seneca": [20, 8, 4],
    "epictetus": [20, 4, 26],
}

COEFFICIENT = 0.11
N_SEEDS = 5

all_results = {}
for author, layers in CONFIGS.items():
    candidates = [{"layer": L, "coefficient": COEFFICIENT} for L in layers]

    result = make_sweep(author).seed_eval_candidates(
        candidates=candidates,
        author=author,
        n_seeds=N_SEEDS,
        vary="judge",
    )
    all_results[author] = result

    # Crash-safe: save each philosopher's result as it completes,
    # so a failure on a later author doesn't waste paid runs.
    out_path = SWEEPS_DIR / f"seed_{author}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"✓ Saved {author} → {out_path}")

print("\nAll philosophers complete.")
