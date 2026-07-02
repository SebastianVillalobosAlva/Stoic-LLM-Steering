# AUDIT.md — Full Repo Revision Brief for Claude Code

**Task:** Audit the entire Stoic-LLM-Steering repo — code, docs, data, results — against the canonical state defined here. Produce a report of discrepancies, then fix the mechanical ones. Do NOT change experimental logic, results JSONs, or tracker conclusions. When a finding is ambiguous, report it instead of fixing it.

Read `CLAUDE.md` first for project context. This file is the audit checklist.

---

## Canonical state (the source of truth for every check)

- **Model:** `meta-llama/Llama-3.2-3B`, float16, CPU. 1B = legacy only.
- **CAA configs:** Marcus **L26**, Seneca **L4**, Epictetus **L8**, coefficient **0.11**. Vectors are per-layer `{layer: tensor}` dicts from **same-layer extraction** (extract residual stream at L, inject at MLP output of L).
- **LoRA:** clean adapters `lora_{author}_clean` (r=8, α=32, q_proj+v_proj, 3 epochs), trained on clean chunked text. Old adapters (`models/epictetus/`, `models/marcus_aurelius/`, `models/seneca/`) = superseded, deleted, never referenced as current.
- **Clean pairs:** N=53 per philosopher, `neutral_pairs_clean.json`. Old counts (100/100/58, 30, 63) = historical only.
- **Dilemma sets:** `data/config/dilemmas_v2.json` canonical (40 items, baseline 0.542); `dilemmas_v1.json` = failed-calibration record (0.881).
- **Superseded claims (must never appear as current):** layer-8 convergence; final-layer vector extraction; old per-philosopher configs (M:L10/0.08, E:L12/0.10-0.15, S:L14/0.10-0.11); style-inflated CAA aggregate deltas as content evidence; "LoRA shifts decisions" without the Amendment-1 revision (passivity prior / heavy-tailed / null structure).
- **Dilemma eval invariants:** both-label-orders averaging; P(stoic) = softmax over the two label tokens only; Δlog-odds reported alongside ΔP; stance + concept bucketing; LoRA path = fresh base per adapter.

---

## Part A — Documentation audit

1. **Grep every `.md` for stale facts.** Search for: `layer 8` / `L8` (valid ONLY for Epictetus-current or as flagged-superseded history), `layer 10|12|14` as current configs, `converge`, `100 pairs|58 pairs|63 pairs` presented as current, `0.881` without the ceiling-effect framing, old adapter paths. Report each hit with file:line and whether it carries a supersession note.
2. **README fixes (from external review — apply these):**
   - In "The publishable anchor" section, after "mechanism → prediction → behavioral confirmation", ADD: *"Caveat: the circuit analyses were run on the earlier configurations (pre-cleaning vectors at L8; original adapters); the prediction was registered before Exp 11 ran, but the paper-grade circuit re-run on the current clean configs (content-relevant patching metric) is pending."*
   - In Methods → CAA, verify against `extractor.py` and correct the extraction-site sentence. Expected true statement: extract from **residual stream / hidden states at layer L**, inject at **MLP output of layer L**. Fix whichever of README/code-docstrings is wrong.
   - Verify `CAA.png` and `LoRA.png` exist at repo root and render. If missing → report.
   - Add an MIT `LICENSE` file (copyright 2026 Sebastian Villalobos Alva) and a license line at the README bottom.
3. **Cross-reference integrity.** Every experiment number cited in any md must exist in the tracker; every relative link (tracker, results/README.md, ModelLens URL) must resolve; every "Next Step(s)" in the tracker must carry a status annotation (DONE/PENDING — per the editorial patch). Report any dangling forward reference.
4. **Tracker editorial patch verification.** Confirm all 9 edits from the editorial patch were applied: Current State section at top; API table at end with the two $0 rows; status notes on Exp 8/9/10 next-steps; Exp 10 synthesis "extended by Exp 11" note; Amendment 2 present; headers normalized to `##` per experiment; `-------` → `---`; Exp 7 status note.

## Part B — Code audit (report + fix mechanical issues; flag logic issues)

5. **Path consistency (high priority — the reproduction table depends on it).**
   - README says results go to `results/dilemmas-v2/` and `results/dilemmas-v2/lora/`. Check `RESULTS_DIR` / `save_results` defaults in `stoic_llm/eval/dilemma.py` and both runner scripts. Make code and README agree (prefer whatever the actual results dirs on disk are named).
   - `DILEMMAS_PATH` must point at `dilemmas_v2.json` explicitly. Grep for any remaining reference to old names (`dilemmas.json`, `dilemma-set-v2.json`).
   - Vector filenames: README says `{author}_steering_3B.pt`; verify `extract_vectors.py` writes and `run_dilemma_eval.py` loads the same names.
   - Adapter paths in `run_lora_dilemma_eval.py` (`models/lora_{author}_clean`) — verify against actual dir names.
6. **CLI consistency.** `extract_vectors.py 3B` (positional) vs `run_dilemma_eval.py --model 3B` (flag): verify each script's actual argparse matches the README table. Unify to `--model` flags if cheap; otherwise fix the README.
7. **Known-correct invariants to verify in code (do not "improve", just confirm):**
   - `dilemma.py`: label tokens asserted single-token; both orders averaged; `_logit` clamps with eps; paired stats include log-odds variant; `summarize` prints Δlog-odds + calibration-gate footer.
   - `LoRADilemmaEval._merged` = **fresh base per adapter** (loads `ModelLoader(self._model_size)`), lazy imports inside the method; `run_all_lora` does `del merged` + `gc.collect()` (CPU — `torch.cuda.empty_cache()` alone is insufficient); base-integrity check present in the runner script.
   - `extractor.py`: same-layer per-layer dict extraction; no final-layer default remains anywhere (grep `layer_idx=-1`).
   - `SteeringRunner`: `_remove_hook()` before `_register_hook()`; `cleanup()` path intact.
   - Matched greedy decoding (`GEN_KWARGS`) shared across steered/baseline in all generation evals.
   - `LoRATrainer.train_all_authors` double-training bug: confirm the repo version trains each author once (or is superseded by the Colab notebook and marked legacy).
8. **Dead code / stale scripts.** Identify scripts that only serve superseded experiments (old bridge scripts using default max-logit metric, old eval runners with unmatched decoding). Do NOT delete — add a one-line header comment: `# LEGACY (Exp N): superseded by <script>; retained for the record.` Report the list.
9. **Config drift.** `config.py`: `LAYER_IDX = 12`, `COEFFICIENT = 0.11` defaults predate everything. Defaults should be harmless or clearly labeled; recommend `DEFAULT_CONFIGS = {"marcus_aurelius": (26, 0.11), "seneca": (4, 0.11), "epictetus": (8, 0.11)}` as the canonical constant, with per-script overrides removed if they duplicate it. Report where configs are hardcoded in scripts vs read from config.

## Part C — Data & results audit

10. **Gitignore enforcement:** `git ls-files | grep -E '\.(pt|safetensors|pth|bin)$'` must return nothing; `models/`, `data/raw/`, `_ckpt/` untracked; report `du -sh .git` (flag if > ~200 MB; do NOT run history rewrites).
11. **Results map:** `results/README.md` must map every filename pattern present on disk to an experiment number, including superseded runs. List any unmapped files.
12. **Data provenance:** `data/lora_training/*.jsonl` must carry the STALE readme note; `neutral_pairs.json` (contaminated) must exist alongside `neutral_pairs_clean.json` (both are needed for the Exp 8→9 controlled comparison — flag if either is missing).

## Part D — Output

Produce `AUDIT_REPORT.md` at repo root with three sections: **Fixed** (mechanical changes made, with diffs summarized), **Flagged** (ambiguities and logic-adjacent findings needing human decision), **Verified clean** (checks that passed). Commit fixes as small, single-purpose commits with messages prefixed `audit:`.

## Do NOT (inherited from CLAUDE.md, restated)

- Never modify `results/**/*.json`, `neutral_pairs*.json`, or tracker experimental conclusions (annotate, never rewrite).
- Never change the LoRA recipe, the dilemma prompt template, or eval math.
- Never commit binaries. Never run `git filter-repo` or force-push.
- Never delete legacy scripts or superseded data — label them.
