"""
scripts/run_eval.py — Stoic LLM evaluation runner

Tests optimal configs for steered vs unsteered outputs
using LLM-as-judge scoring.

Usage:
  python scripts/run_eval.py          # defaults to 1B
  python scripts/run_eval.py 3B       # uses 3B
"""

import sys
from stoic_llm.model import ModelLoader
from stoic_llm.eval.judge import StoicJudge, summarize_eval
from stoic_llm.steering.runner import SteeringRunner
from stoic_llm.config import DEFAULT_PROMPTS, VECTORS_DIR

model_size = sys.argv[1] if len(sys.argv) > 1 else "1B"

loader = ModelLoader(model_size)
model, tokenizer = loader.load()
judge = StoicJudge()

# Optimal configs per model size
configs = {
    "1B": {
        "marcus_aurelius": {"layer": 10, "coefficient": 0.08},
        "epictetus": {"layer": 12, "coefficient": 0.10},
        "seneca": {"layer": 14, "coefficient": 0.10},
    },
    "3B": {
        "marcus_aurelius": {"layer": 8, "coefficient": 0.08},
        "epictetus": {"layer": 8, "coefficient": 0.15},
        "seneca": {"layer": 8, "coefficient": 0.11},
    },
}

model_configs = configs[model_size]

for author, cfg in model_configs.items():
    if cfg["layer"] is None:
        print(
            f"\n⚠ No optimal config for {author} on {model_size} — run run_sweep.py {model_size} first"
        )
        continue

    print(f"\n{'='*60}")
    print(f"{author} — {model_size} — layer={cfg['layer']}, coeff={cfg['coefficient']}")
    print(f"{'='*60}")

    vector_path = VECTORS_DIR / f"{author}_steering_{model_size}.pt"

    runner = SteeringRunner(
        file_path=str(vector_path),
        model=model,
        tokenizer=tokenizer,
        layer=cfg["layer"],
        coefficient=cfg["coefficient"],
        prompts=DEFAULT_PROMPTS,
    )

    steered = runner.run_model_with_hook(
        return_output=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    runner.cleanup()

    if steered is None:
        print(f"WARNING: No steered output for {author}, skipping.")
        continue

    unsteered = []
    for prompt in DEFAULT_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        unsteered.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    results = judge.evaluate_steering(
        prompts=DEFAULT_PROMPTS,
        steered_outputs=steered,
        unsteered_outputs=unsteered,
        author=author,
        metadata={
            "model_size": model_size,
            "layer": cfg["layer"],
            "coefficient": cfg["coefficient"],
            "pairs": "100+",
        },
    )

    print(summarize_eval(results))
    judge.save_results(results)
