import json
from stoic_llm.model import ModelLoader
from stoic_llm.eval.sweep import SteeringSweep
from stoic_llm.eval.judge import StoicJudge
from stoic_llm.config import VECTORS_DIR, SWEEPS_DIR

loader = ModelLoader("3B")
model, tokenizer = loader.load()
gemini_judge = StoicJudge(provider="gemini")


def make_sweep(author):
    return SteeringSweep(
        model=model,
        tokenizer=tokenizer,
        vector_path=str(VECTORS_DIR / f"{author}_steering_3B.pt"),
        judge=gemini_judge,
    )


CANDIDATES = {
    "marcus_aurelius": [26, 12, 16],
    "seneca": [4, 26, 12],
    "epictetus": [8, 24, 20],
}

for author, layers in CANDIDATES.items():
    cands = [{"layer": L, "coefficient": 0.11} for L in layers]
    result = make_sweep(author).seed_eval_candidates(
        candidates=cands,
        author=author,
        n_seeds=5,
        vary="judge",
    )
    with open(SWEEPS_DIR / f"seed_gemini_clean_{author}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"✓ Saved {author}")
