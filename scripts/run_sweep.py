from stoic_llm.model import ModelLoader
from stoic_llm.eval.sweep import SteeringSweep, summarize_sweep
from stoic_llm.config import VECTORS_DIR

loader = ModelLoader()
model, tokenizer = loader.load()

authors = {
    "marcus_aurelius": VECTORS_DIR / "marcus_aurelius_steering.pt",
    "seneca": VECTORS_DIR / "seneca_steering.pt",
    "epictetus": VECTORS_DIR / "epictetus_steering.pt",
}

for author, vector_path in authors.items():
    print(f"\n{'='*60}")
    print(f"SWEEPING: {author}")
    print(f"{'='*60}")

    sweep = SteeringSweep(
        model=model,
        tokenizer=tokenizer,
        vector_path=str(vector_path),
    )

    results = sweep.full_sweep(author=author)
    print(summarize_sweep(results))
    sweep.save_results(results)

print(f"\n{'='*60}")
print("Done! All sweeps complete.")
print(f"{'='*60}")
