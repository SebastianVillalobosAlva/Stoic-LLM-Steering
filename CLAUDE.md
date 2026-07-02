# CLAUDE.md — Stoic LLM Repo: Context & Cleanup Brief

## What this project is

Research codebase testing whether philosophical (Stoic) reasoning can be steered into LLMs, comparing two methods — CAA activation steering vs LoRA fine-tuning — and measuring effects at three levels: **style** (LLM-judge rubric), **content** (judge-scored reasoning in prose), and **decision** (judge-free forced-choice dilemmas). Companion project: **ModelLens**, an architecture-agnostic interpretability toolkit used for circuit analysis. Target: Anthropic Fellows application; the publishable anchor is the CAA-vs-LoRA circuit-topology comparison (Exp 5/6) plus the decision-level behavioral split (Exp 10/11).

## Current headline findings (do not contradict these when editing docs)

1. **CAA is a register direction, not a decision direction.** Moves style robustly (Exp 3b), moves judge-scored content only with clean pairs (Exp 9: Marcus L26 +0.408, Seneca L4 +0.583, Epictetus L8 +0.767, Gemini judge, N=5), moves decisions **not at all** at any coefficient up to 1.5 (Exp 10, calibrated baseline 0.542).
2. **LoRA reaches the decision layer where CAA doesn't** (Exp 11) — but with structure: Marcus = broad passivity prior (accepting 18+/4−, p=0.004; active flat), Seneca = heavy-tailed magnitude effect (t=2.39 but sign test n.s. 25/40), Epictetus = null (smallest corpus, 123 chunks, Enchiridion only).
3. **Pair quality was the determining variable for CAA content** (Exp 9): contamination fix flipped nulls to strong positives, single-variable controlled.
4. **Known confounds, deliberately carried:** CAA=contrastive objective vs LoRA=continued pretraining (method+objective jointly); corpus size ⊥ philosopher identity (perfectly confounded); Senecan-idiom lexical echo in dilemma options (untested — next experiment).

## Model / configs (canonical)

- Base: `meta-llama/Llama-3.2-3B`, float16, CPU (M4 MacBook, 16GB). 1B is legacy.
- Clean CAA vectors: per-layer `{layer: tensor}` dicts — Marcus L26, Seneca L4, Epictetus L8, coeff 0.11.
- Clean LoRA adapters: `lora_{author}_clean/` (r=8, alpha=32, q_proj+v_proj, 3 epochs, trained on clean chunked text, Colab T4). NOT in git — in Drive + local.
- Dilemma eval: **v2 set is canonical** (40 items, baseline 0.542). v1 is retained only as the failed-calibration record (baseline 0.881, ceiling).

## Cleanup tasks (prioritized)

### 1. Single source of truth for the tracker
- Merge `exp10-tracker-entry.md` and `exp11-tracker-entry.md` INTO `Stoic LLM - Results Tracker.md` (chronological order, after Exp 9). Then DELETE the two standalone files.
- Bump tracker "Last updated" to the current date.
- The tracker has a stray section header `# New Steering Vectors` before Exp 9 — remove or demote it; Exp 9 should follow Exp 8 under the same document structure.

### 2. Dilemma file naming (ambiguity hazard)
- `data/config/` contains both `dilemmas.json` and `dilemma-set-v2.json`. Determine which one the harness (`DILEMMAS_PATH` in `stoic_llm/eval/dilemma.py`) actually reads.
- Rename to `dilemmas_v1.json` and `dilemmas_v2.json`; point `DILEMMAS_PATH` explicitly at v2; grep for any other references and update.

### 3. Gitignore hygiene (partially done — verify)
- Should be ignored and untracked: `models/`, `data/raw/`, `data/steering_vectors/*.pt`, processed `.txt` corpora, `__pycache__/`, `.env`.
- Previously tracked files need `git rm --cached` (gitignore alone doesn't untrack). Verify none of: `.pt`, adapter binaries, raw `.txt` remain tracked (`git ls-files | grep -E '\.(pt|safetensors|txt)$'`).
- `data/steering_vectors/temp_combined.pt` is scratch — delete from disk.
- Old per-author adapter dirs (`models/epictetus/`, `models/marcus_aurelius/`, `models/seneca/` + their `checkpoint-45/`) are superseded contaminated-data adapters, deletions already staged — confirm committed.
- Check repo weight: `du -sh .git`. If bloated from previously committed binaries, flag it (do NOT run filter-repo without asking).

### 4. Dead / stale data files (verify then delete)
- `data/processed/*/neutral_pairs_30.json`, `_63.json`, `_backup.json` — legacy pair sets from Exp 1/2 era. Keep `neutral_pairs_clean.json` and `neutral_pairs_rejected.json` (Exp 9 artifacts) and the original `neutral_pairs.json` (Exp 8 contaminated baseline — needed for the before/after story). Delete `_backup` if it duplicates; keep `_30`/`_63` only if referenced in tracker Exp 1/2 (they are — so either keep or note in tracker they were pruned).
- `data/lora_training/*.jsonl` — OLD pre-clean LoRA training data (Exp 4). Superseded by clean chunked JSON. Keep for provenance or delete with a tracker note; do not let anyone train from them again (add a README line in `data/lora_training/` saying "STALE — Exp 4 era, pre-cleaning; clean LoRA trains on data/chunked/").

### 5. Code organization
- Confirm layout: logic in `stoic_llm/eval/dilemma.py` (contains `DilemmaEval` + `LoRADilemmaEval` + `_logit`), thin scripts in `scripts/` (`run_dilemma_eval.py`, `run_lora_dilemma_eval.py` with `--model` argparse, `sign_test.py`, `sign_test_buckets.py`, `top_movers.py`).
- `LoRADilemmaEval._merged` MUST be the fresh-base-per-adapter version (reuse-the-base was caught stacking adapters via PEFT `peft_config` warning). Verify; the in-loop `del merged` should be followed by `gc.collect()` (CPU — `torch.cuda.empty_cache()` is a no-op here).
- Extraction: `extract_vectors.py` must extract per-layer `{layer: tensor}` dicts (same-layer extract/inject). The old final-layer extraction was a bug (pre-Exp 8).
- Known conventions: type hints on public methods, hook cleanup (`_remove_hook` before `_register_hook`), `mkdir(parents=True, exist_ok=True)` before writes, try/except around API calls in batch ops.

### 6. Results directory
- `results/dilemmas/` (CAA) and `results/dilemmas/lora/` (LoRA) — keep all JSONs, they're evidence.
- `results/sweeps/` contains both buggy-extraction sweeps (May, pre-Exp-8-fix) and corrected ones — do NOT delete the buggy ones (they're cited in the tracker as the superseded record), but consider a `results/README.md` mapping filename patterns → experiment numbers (e.g. `full_*_202605*` = Exp 3 buggy extraction; `seed_gemini_clean_*` = Exp 9; `dilemma_eval_20260612/0614` = Exp 10 v1/v2; `dilemma_eval_20260701` = Exp 11).

### 7. Repo README (if time permits)
- Top-level README should state: project one-liner, the three-level findings table (style/content/decision × CAA/LoRA), how to reproduce each experiment (script + data + expected runtime), and the ModelLens relationship. The headline figure is Exp 9's before/after and Exp 10/11's CAA-flat vs LoRA-moves split.

## Do NOT

- Do not modify any results JSON.
- Do not delete `neutral_pairs.json` (contaminated originals) — they are the Exp 8/9 controlled comparison.
- Do not "fix" the objective confound by changing the LoRA recipe — it's held identical to Exp 4 on purpose.
- Do not rewrite tracker experiment conclusions — supersession notes are added, history is never scrubbed (house style: strikethrough-by-annotation, e.g. Exp 3's supersession blockquote).
- Do not commit anything in `models/` or any `.pt`/`.safetensors`.

## Next experiments queued (context for why files exist)

1. Senecan-idiom scoring of the 40 v2 Stoic options (judge pass) → correlate with Seneca per-item Δlog-odds → tests lexical-echo confound.
2. v3 dilemma set: 2×2 (Letters-core topic vs off-topic × plain vs Stoic-idiom option phrasing).
3. Epictetus full-corpus retrain (Enchiridion + Discourses) → corpus-size hypothesis.
4. Expand dilemma set 40 → 80+ uniformly.
