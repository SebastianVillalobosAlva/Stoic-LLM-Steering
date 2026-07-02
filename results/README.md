# results/ — filename → experiment map

Every JSON here is evidence and is kept, **including superseded runs** (they are
cited in `Stoic LLM - Results Tracker.md` as the record that was later corrected —
do not delete them). This file maps filename patterns to experiment numbers so the
old and corrected runs can be told apart.

The **buggy-extraction boundary** is the single most important line below: steering
vectors extracted before the Experiment 8 fix (≈ June 8, 2026) were captured at the
final layer and injected at layer L (a bug). Runs dated **before June 8** are the
superseded record; runs dated **June 8 onward** use corrected same-layer extraction.

## `sweeps/` — layer × coefficient sweeps and seed-averaged evals

| Pattern | Experiment | Notes |
|---|---|---|
| `full_{author}_202605{17,24,25,30}_*.json` | Exp 1 / 3 era | **Buggy extraction (pre-June-8), superseded.** Full grid sweeps; May 17 ≈ 1B/early, May 24–25 = Exp 3 (3B), May 30 = intermediate. Retained as the cited superseded record. |
| `full_{author}_2026061{0,1}_*.json` | Exp 8 | Corrected same-layer extraction, re-sweep (June 10–11). |
| `seed_{author}.json` | Exp 8 | Seed-averaged content eval, **Sonnet** judge, contaminated pairs. |
| `seed_gemini_{author}.json` | Exp 8 (Gemini column) | Seed-averaged, **Gemini** judge, still **contaminated** pairs — the "Contaminated (Exp 8, Gemini)" column in the Exp 9 table. |
| `seed_gemini_clean_{author}.json` | **Exp 9** | Seed-averaged, **Gemini** judge, **clean** pairs — the headline clean-vs-contaminated result. |

## `judges/` — per-run generation + judge outputs

| Pattern | Experiment |
|---|---|
| `eval_{author}_20260520–27_*.json` | Exp 2 / 3 / 3b / 8 — individual judge eval runs (raw generations + rubric scores). |

## `bridge/` — ModelLens circuit analyses (local compute, no API)

| Pattern | Experiment |
|---|---|
| `bridge_3B_*.json` | **Exp 5** — CAA × ModelLens (activation-steering circuit). |
| `bridge_lora_3B_*.json` | **Exp 6** — LoRA × ModelLens (fine-tuning circuit). |

## `dilemmas/` and `dilemmas-v2/` — forced-choice decision-level evals

The `v1` / `v2` directory split parallels `data/config/dilemmas_v1.json` /
`dilemmas_v2.json`. **v2 is canonical**; v1 is the failed-calibration record.
Code writes here via `RESULTS_DIR` (`stoic_llm/eval/dilemma.py`).

| File | Experiment | Baseline P(stoic) |
|---|---|---|
| `dilemmas/dilemma_eval_20260612_*.json` | **Exp 10a** — v1 set, CAA | 0.881 (ceiling — failed calibration) |
| `dilemmas-v2/dilemma_eval_20260614_*.json` | **Exp 10b/c** — v2 set, CAA (+ coeff sweep) | 0.542 (calibrated) |
| `dilemmas-v2/lora/dilemma_eval_20260701_*.json` | **Exp 11** — v2 set, LoRA (merged adapters) | 0.542 (same base baseline) |

## `comparisons/`

Reserved (`COMPARISONS_DIR` in `stoic_llm/config.py`); currently empty. CAA-vs-LoRA
synthesis lives in the tracker, not as a saved JSON.
