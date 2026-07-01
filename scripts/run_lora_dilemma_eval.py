"""Run the LoRA forced-choice dilemma eval.

Compares three clean-data LoRA adapters against the SAME base-model
baseline used by the CAA run. Read the Δlog-odds column: does LoRA reach
the decision where CAA was flat?

Run from the repo root so the relative model paths resolve:
    python scripts/run_lora_dilemma_eval.py
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from stoic_llm.eval.dilemma import LoRADilemmaEval, RESULTS_DIR
from stoic_llm.model import ModelLoader

# Point these at wherever you unzipped the adapters (repo_root/models/ here).
ADAPTERS = {
    "marcus": "models/lora_marcus_clean",
    "seneca": "models/lora_seneca_clean",
    "epictetus": "models/lora_epictetus_clean",
}


def main() -> None:
    base, tok = ModelLoader("3B").load()
    ev = LoRADilemmaEval(base, tok)

    results = ev.run_all_lora(ADAPTERS)

    # --- base-integrity check (works even without the in-class guard) ---
    # If merge_and_unload mutated the base in place, the base now carries
    # merged adapters; recomputing the baseline will drift from the one
    # reported at the start of the run.
    start = results["baseline_mean"]
    end = sum(ev.eval_condition().values()) / len(ev.dilemmas)
    drift = abs(end - start)
    print(f"\nbase integrity: start {start:.3f} | end {end:.3f} | drift {drift:.4f}")
    if drift >= 0.005:
        print("  ** BASE MUTATED across authors -> results contaminated.")
        print("     Switch _merged to the fresh-base version and rerun.")
    else:
        print("  OK base intact; reuse-the-base merge is safe.")

    ev.save_results(results, out_dir=RESULTS_DIR / "lora")
    print()
    print(ev.summarize(results))


if __name__ == "__main__":
    main()
