"""
scripts/run_sweep.py — Full layer + coefficient sweep

Runs a two-stage sweep for each philosopher:
  1. Layer sweep at a fixed coefficient to find the best layer
  2. Coefficient sweep at the best layer to find optimal strength

Usage:
  python scripts/run_sweep.py          # defaults to 1B
  python scripts/run_sweep.py 3B       # uses 3B
"""

import sys
from stoic_llm.model import ModelLoader
from stoic_llm.eval.sweep import SteeringSweep, summarize_sweep
from stoic_llm.config import VECTORS_DIR

model_size = sys.argv[1] if len(sys.argv) > 1 else "1B"

loader = ModelLoader(model_size)
model, tokenizer = loader.load()

# Scale layer range based on model size
layer_range = {
    "1B": [4, 6, 8, 10, 12, 14],
    "3B": [4, 8, 12, 16, 20, 24, 26],
}

authors = {
    "marcus_aurelius": VECTORS_DIR / f"marcus_aurelius_steering_{model_size}.pt",
    "seneca": VECTORS_DIR / f"seneca_steering_{model_size}.pt",
    "epictetus": VECTORS_DIR / f"epictetus_steering_{model_size}.pt",
}

for author, vector_path in authors.items():
    print(f"\n{'='*60}")
    print(f"SWEEPING: {author} ({model_size})")
    print(f"{'='*60}")

    sweep = SteeringSweep(
        model=model,
        tokenizer=tokenizer,
        vector_path=str(vector_path),
    )

    results = sweep.full_sweep(
        layers=layer_range.get(model_size, [4, 8, 12, 16, 20, 24]),
        author=author,
    )
    print(summarize_sweep(results))
    sweep.save_results(results)

print(f"\n{'='*60}")
print("Done! All sweeps complete.")
print(f"{'='*60}")
