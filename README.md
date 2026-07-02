# Stoic LLM Steering

**Can philosophical (Stoic) reasoning be steered into an LLM — and at what depth does each method actually operate?** This project compares two interventions — **CAA activation steering** vs **LoRA fine-tuning** — on Llama-3.2-3B, and measures their effects at three levels: **style** (how the model writes), **content** (how it reasons in prose), and **decision** (what it actually chooses). The companion project [**ModelLens**](https://github.com/SebastianVillalobosAlva/ModelLens) supplies the circuit analysis that explains the behavioral split.

> Full experiment-by-experiment record: [`Stoic LLM - Results Tracker.md`](Stoic%20LLM%20-%20Results%20Tracker.md).

## Headline findings

| Level | CAA (activation steering) | LoRA (weight adaptation, clean data) |
|---|---|---|
| **Style / register** | Robust (+1.0–1.6) | Robust, stronger (+1.6–2.8) |
| **Content / prose reasoning** | Strong with clean pairs (+0.408 to +0.767, independent judge) | Not re-scored on clean adapters |
| **Decision / choice** | **None**, at any coefficient up to 1.5 | **Structured effects** (Seneca shift, Marcus passivity prior, Epictetus null) |

Three results anchor the project:

1. **Pair quality is the determining variable for CAA content** (Exp 9). Holding model, method, layer, judge, and N fixed, swapping contaminated contrastive pairs for clean reasoning-isolating ones flipped nulls into strong positives — Epictetus L8: **−0.100 → +0.767** under an independent Gemini judge. Single-variable, controlled.
2. **CAA is a register direction, not a decision direction** (Exp 10). On a calibrated judge-free forced-choice instrument (baseline P(stoic) = 0.542), CAA is flat at every coefficient up to the edge of incoherence. It changes how the model *talks*, weakly shifts how it *reasons* in prose, and does not change what it *chooses*.
3. **LoRA reaches the decision layer where CAA does not** (Exp 11) — and the circuit topology predicted it. But what LoRA installs is *not* uniform Stoic decision-making: **Marcus** = broad passivity prior (accepting items 18+/4−, p=0.004; active flat), **Seneca** = heavy-tailed minority-of-items effect (t=2.39, sign test n.s.), **Epictetus** = null (smallest corpus).

**Open confounds, deliberately carried:** CAA's contrastive objective vs LoRA's continued-pretraining objective (method + objective jointly); corpus size perfectly confounded with philosopher identity; a possible Senecan-idiom lexical echo in the dilemma option wording (next experiment).

## The publishable anchor: CAA vs LoRA circuit topology

The most defensible result is a **behavioral-outcome-controlled circuit comparison** (Experiments 5 & 6, via ModelLens): same Stoic-flavored output, different internal path. CAA injects into mid-MLPs and is partially compensated away before the decision forms; LoRA's changes are distributed with dominant output-layer effects that reach it. This is exactly why the decision-level split (Exp 10/11) falls out the way it does — mechanism → prediction → behavioral confirmation.

| CAA circuit (Exp 5) | LoRA circuit (Exp 6) |
|---|---|
| ![CAA circuit](CAA.png) | ![LoRA circuit](LoRA.png) |

## Methods

**CAA (Contrastive Activation Addition).** Extract a per-layer steering vector as the mean activation difference between Stoic and neutral contrastive pairs, captured at `model.layers[L].mlp`; at inference, add `coefficient * vector` back at the **same** layer (same-layer extract/inject — the pre-Exp-8 code extracted at the final layer, a bug). Zero training cost, adjustable at inference.

**LoRA.** Parameter-efficient continued pretraining on cleaned Stoic text (r=8, α=32, `q_proj`+`v_proj`, 3 epochs), one adapter per philosopher, merged into the base for evaluation.

**Decision-level instrument (the decisive test).** A judge-free, generation-free forced choice: present a situation with two options labeled A/B (one Stoic, one a genuinely defensible competing-values alternative), take one forward pass, and read `P(stoic) = softmax over {A,B}` on the next token. Every item runs in both label orders and is averaged, which cancels position/label bias exactly. No judge, no dependence on style.

## Canonical configuration

- **Base:** `meta-llama/Llama-3.2-3B`, float16, CPU (M4, 16 GB). The 1B path is legacy.
- **Clean CAA vectors:** per-layer `{layer: tensor}` dicts — Marcus **L26**, Seneca **L4**, Epictetus **L8**, coeff **0.11**.
- **Clean LoRA adapters:** `models/lora_{author}_clean/` (not in git — trained on Colab T4, kept in Drive + local).
- **Dilemma set:** `data/config/dilemmas_v2.json` is **canonical** (40 items, baseline 0.542). `dilemmas_v1.json` is retained only as the failed-calibration record (baseline 0.881, ceiling).

## Reproducing the experiments

Setup:

```bash
pip install -e .
export ANTHROPIC_API_KEY=...   # judge (Exp 8); GEMINI_API_KEY for the Exp 9 judge
```

Heavy artifacts (`models/`, `data/raw/`, `data/steering_vectors/*.pt`, processed corpora) are gitignored; the corpora are reproducible via the downloader, and LoRA adapters live in Drive.

| Experiment | Script | Data | Notes / runtime |
|---|---|---|---|
| Corpus + contrastive pairs | `scripts/file_downloader.py`, `chunk_generator.py`, `generate_pairs.py`, `clean_pairs.py` | Gutenberg → `data/chunked/`, `data/processed/*/neutral_pairs_clean.json` | Cleaning pipeline is what fixed Exp 9 |
| CAA vector extraction | `scripts/extract_vectors.py 3B` | clean pairs → `data/steering_vectors/{author}_steering_3B.pt` | seconds/philosopher, local |
| **Exp 8/9** — content (judge) | `scripts/run_seed_eval.py` | vectors + judge | Exp 9 uses the Gemini judge → `results/sweeps/seed_gemini_clean_*` |
| **Exp 5/6** — circuits (ModelLens) | `scripts/bridge_analysis_caa.py`, `bridge_analysis_lora.py` | vectors / adapters | local compute, no API → `results/bridge/` |
| **Exp 10** — decision, CAA | `scripts/run_dilemma_eval.py --model 3B` | `dilemmas_v2.json` + vectors | ~29 min CPU, $0 → `results/dilemmas-v2/` |
| **Exp 11** — decision, LoRA | `scripts/run_lora_dilemma_eval.py --model 3B` | `dilemmas_v2.json` + adapters | ~33 min CPU, $0 → `results/dilemmas-v2/lora/` |
| Significance follow-ups | `scripts/sign_test.py`, `sign_test_buckets.py`, `top_movers.py` | a saved dilemma-eval JSON | distribution-free, zero cost |

See [`results/README.md`](results/README.md) for the full filename → experiment map (including superseded pre-June-8 buggy-extraction runs, which are retained as the cited record).

## Project structure

```
stoic-llm/
├── stoic_llm/
│   ├── data/                # download, chunk, pair generation + cleaning
│   ├── steering/            # CAA: vector extraction + inference hook
│   ├── lora/                # LoRA fine-tuning
│   ├── eval/                # judge/sweep + dilemma.py (decision-level eval)
│   ├── model.py             # ModelLoader (1B legacy / 3B canonical)
│   └── config.py            # consolidated paths/config
├── data/
│   ├── config/              # dilemmas_v1.json (v1), dilemmas_v2.json (canonical)
│   ├── chunked/             # clean chunked corpora (LoRA training source)
│   └── processed/           # contrastive pairs (clean + contaminated baselines)
├── results/                 # evidence — see results/README.md
├── scripts/                 # entry points
└── Stoic LLM - Results Tracker.md   # full experiment log
```

## Relationship to ModelLens

[**ModelLens**](https://github.com/SebastianVillalobosAlva/ModelLens) is an architecture-agnostic interpretability toolkit used here for the circuit analysis (Experiments 5 & 6). The Stoic LLM project supplies a controlled behavioral target — the same Stoic-flavored output produced two different ways — and ModelLens supplies the logit-lens, residual-stream, and activation-patching machinery that shows the two methods take different internal paths. That circuit comparison is what turns the Exp 10/11 behavioral split from an observation into a prediction.

## Author

**Sebastian Villalobos**
- GitHub: [@SebastianVillalobosAlva](https://github.com/SebastianVillalobosAlva)
- LinkedIn: [Sebastian Villalobos Alva](https://www.linkedin.com/in/sebastian-villalobos-alva/)
