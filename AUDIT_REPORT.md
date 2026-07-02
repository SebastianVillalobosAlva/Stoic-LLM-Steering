# AUDIT_REPORT.md — Full Repo Audit Findings

**Scope:** executed the checklist in `AUDIT.md` against the canonical state it defines. Read `CLAUDE.md` for context first, as instructed. No `results/**/*.json`, `neutral_pairs*.json`, or tracker experimental conclusions were modified. No binaries committed, no `git filter-repo`, no force-push.

Commits (branch `audit-pass`, prefixed `audit:`):
- `1dea510` — add MIT LICENSE and README external-review fixes
- `05c0f3c` — fix stale vector-filename docstrings in `dilemma.py`
- `3f7eb02` — label superseded scripts as LEGACY (not deleted)

---

## Fixed

### 1. LICENSE missing (AUDIT.md A2, item 4)
Added `LICENSE` (MIT, copyright 2026 Sebastian Villalobos Alva) and a `## License` section linking it at the bottom of `README.md`.

### 2. README circuit-analysis caveat missing (AUDIT.md A2, item 2)
Added the exact caveat sentence specified by the brief to the "publishable anchor" section, after "mechanism → prediction → behavioral confirmation": the Exp 5/6 circuit runs used the pre-cleaning L8 vectors and original adapters, not the current clean configs; the paper-grade re-run with a content-relevant patching metric is pending.

### 3. Stale vector-filename docstrings in `stoic_llm/eval/dilemma.py` (AUDIT.md B5)
Both the module docstring (`{author}_clean.pt`) and `run_all`'s docstring example (`marcus_clean.pt`, `seneca_clean.pt`, `epictetus_clean.pt`) referenced a naming convention that `extract_vectors.py` never writes and no runner script ever loads. Corrected both to `{author}_steering_{model_size}.pt` — the convention `extract_vectors.py` actually produces (`data/steering_vectors/epictetus_steering_3B.pt` etc.) and what `run_dilemma_eval.py` actually loads.

### 4. Five scripts mislabeled as current (AUDIT.md B8)
Added one-line `# LEGACY` header comments (no deletions, no logic changes):

| Script | Why it's dead | Superseded by |
|---|---|---|
| `run_eval.py` | Hardcodes the exact superseded configs (all-layer-8 on 3B; L10/L12/L14 on 1B) that Experiment 8 invalidated | `run_seed_eval.py` (content), `run_dilemma_eval.py` (decision) |
| `run_lora_eval.py` | `LoRARunner` resolves adapters at `models/{size}/{author}/`, which holds the **old contaminated-data adapters** (dated May 26) — not the current `models/lora_{author}_clean/` (dated June 30) | `run_lora_dilemma_eval.py` |
| `pair_generator_2.py` | 5-pair Marcus-only smoke test | `generate_pairs.py` |
| `test_2.py` (`trim_pairs.py`) | One-off migration that produced the current N=53 `neutral_pairs.json`; task complete | n/a — retained for provenance |
| `read_json.py` | Scratch inspector hardcoded to a pre-Exp-8 buggy-extraction sweep file | n/a — retained for provenance |

`run_lora_eval.py`'s finding is the highest-value catch in this audit: it is not just stylistically outdated, it silently evaluates the **wrong (contaminated) adapters** if run today, with no error or warning.

---

## Flagged (ambiguous — needs a human decision)

### 1. AUDIT.md's prescribed CAA extraction-site statement conflicts with the code
AUDIT.md's Part A2 instructs: *"Expected true statement: extract from **residual stream / hidden states at layer L**, inject at **MLP output of layer L**."* This does not match the actual code. `extractor.py`'s `_mlp_module` hooks `model.model.layers[layer_idx].mlp` and captures its **output** (`out.detach()` in `extract_activations_multi`) — not the residual stream. `runner.py`'s `SteeringRunner` injects at the same site (`layers[L].mlp.register_forward_hook`). The README's Methods section already correctly says "captured at `model.layers[L].mlp`" — same site for extraction and injection, which matches the code and matches the "same-layer extraction" framing throughout the tracker (Exp 8's fix was same-layer extract/inject, replacing final-layer extraction; residual stream was never the site).

**I did not change the README to match AUDIT.md's prescribed statement**, since doing so would make the docs contradict the code. Flagging so a human can confirm: either AUDIT.md's brief is describing an intended future change not yet implemented, or the brief's author misremembered the site and the code/README are correct as they stand.

### 2. `config.py`'s `LAYER_IDX = 12` / `COEFFICIENT = 0.11` are stale, live defaults
These are used as function-signature defaults in `stoic_llm/steering/runner.py` and `stoic_llm/eval/sweep.py`. `LAYER_IDX = 12` doesn't match any current canonical per-philosopher config (Marcus L26, Seneca L4, Epictetus L8) — it's a holdover from Epictetus's old 1B Experiment-1 config. AUDIT.md recommends introducing `DEFAULT_CONFIGS = {"marcus_aurelius": (26, 0.11), "seneca": (4, 0.11), "epictetus": (8, 0.11)}` as the canonical constant.

I did not apply this: a single scalar default cannot represent three different per-philosopher configs, so the fix requires refactoring the call sites in `runner.py` and `sweep.py` to accept a per-philosopher lookup rather than a positional/keyword default — a behavior change beyond a docstring or rename fix, and adjacent to "eval math" the brief says never to change. Recommend a human decide whether to (a) refactor call sites to require an explicit `(layer, coeff)` pair with no silent default, or (b) leave as-is with a comment marking the default as arbitrary/non-canonical.

### 3. `bridge_analysis_caa.py` / `bridge_analysis_lora.py` still use the default max-logit patching metric
Confirmed: `run_patching_bridge` (CAA) and `run_patching_analysis` (LoRA) both compute `clean_metric = clean_logits.max(dim=-1).values.mean().item()` — the default max-logit metric. This is exactly what the tracker (Exp 6 note, "Layer Evolution" section) already flags as needing a content-relevant `metric_fn` for the paper-grade re-run, and is now also captured by the new README caveat (Fixed #2). **No LEGACY label applies** — there is no superseding script for AUDIT.md's `# LEGACY (Exp N): superseded by <script>` template to point at; these are still the only/current bridge scripts, just carrying a known, already-documented pending upgrade. No action taken beyond confirming the finding is real and already tracked.

### 4. `train_lora.py` — unclear if still a live path
`train_lora.py` (local CPU/MPS trainer via `LoRATrainer.train_all_authors`) and `lora-train-colab.py` (the Colab T4 notebook CLAUDE.md identifies as what actually produced the current clean adapters) both exist and use compatible hyperparameters (r=8, α=32, 3 epochs). I could not confirm from the repo alone whether `train_lora.py` is a still-valid local alternative or a fully-replaced path — unlike `run_lora_eval.py`, it doesn't point at demonstrably stale data, it's just unclear which path is "current." Left unlabeled; flagging for a human call.

---

## Verified clean

**Part A — Documentation**
- **A1 (stale-fact greps).** Searched all `.md` files for `layer 8`/`L8`, `layer 10|12|14`, `converge`, `100/58/63 pairs`, `0.881`, and old adapter paths. Every hit is either a currently-valid fact (Epictetus L8 is genuinely canonical; 0.542 dilemma baseline) or sits inside a section that already carries an explicit supersession blockquote (Exp 3/3b's "⚠ PARTIALLY SUPERSEDED" notes, Exp 1/2's historical pair counts). No unflagged stale claims found.
- **A3 (cross-reference integrity).** Every experiment number cited resolves in the tracker. All relative links checked (`CAA.png`, `LoRA.png`, the tracker file, `results/README.md`) resolve to real files. All four `#### Next Step(s)` sections in the tracker (Exp 8, 9, 10, 11) carry status annotations from the editorial patch.
- **A4 (editorial patch verification).** All 9 edits confirmed present: Current State section at top; API-cost table at end with the two `$0` Exp 10/11 rows; status notes on Exp 8/9/10 next-steps; Exp 10 synthesis "Extended by Exp 11" note; Amendment 2 present; all experiment headers normalized to `##`; no remaining `-------` separators; Exp 7 status-update note present.

**Part B — Code**
- **B5 (path consistency, remainder).** `RESULTS_DIR` in `dilemma.py` = `results/dilemmas-v2`, matches `README.md`'s reproduction table and the actual directories on disk. `DILEMMAS_PATH` points explicitly at `dilemmas_v2.json`; grepped all `.py` files for the old names (`dilemmas.json`, `dilemma-set-v2.json`) — zero hits. Adapter paths in `run_lora_dilemma_eval.py` (`models/lora_{author}_clean`) match the actual directory names on disk exactly.
- **B6 (CLI consistency).** `extract_vectors.py` takes a positional arg; `run_dilemma_eval.py`/`run_lora_dilemma_eval.py` take `--model`. README's reproduce table already documents each script's actual invocation correctly (`extract_vectors.py 3B` vs `run_dilemma_eval.py --model 3B`) — no doc fix needed, and unifying the CLIs wasn't judged cheap enough to justify touching `extract_vectors.py`'s stable, working interface.
- **B7 (known-correct invariants).** All confirmed present and unchanged:
  - `dilemma.py`: `_single_token_id` raises loudly if a label isn't a single token; `p_stoic` averages both label orders; `_logit` clamps with `eps=1e-6`; `_paired_stats` reports both P-space and log-odds; `summarize` prints Δlog-odds plus the calibration-gate footer.
  - `LoRADilemmaEval._merged`: fresh `ModelLoader(self._model_size)` base per adapter (verified no adapter stacking); `PeftModel`/`ModelLoader` imports are lazy, inside the method; `run_all_lora` does `del merged` followed by `gc.collect()` (fixed in a prior pass — confirmed still present); `run_lora_dilemma_eval.py` retains its base-integrity check (`start`/`end`/`drift` printout).
  - `extractor.py`: same-layer, per-layer `{layer: tensor}` extraction confirmed (`extract_activations_multi` hooks `layers[L].mlp` per requested layer in one forward pass); grepped for `layer_idx=-1` / `hidden_states[-1]` as an active default — the only hit is a docstring describing the *old, fixed* bug, not live code.
  - `SteeringRunner`: `_register_hook` calls `_remove_hook()` first; `cleanup()` calls `_remove_hook()`.
  - Matched greedy decoding: `GEN_KWARGS` (`do_sample=False`, fixed `repetition_penalty`/`no_repeat_ngram_size`) is shared across baseline and steered generation in `sweep.py`, `run_eval.py`, and `run_lora_eval.py` — decoding itself is matched even in the two now-LEGACY-labeled scripts (their staleness is about which *data* they load, not decoding mismatch).
  - `LoRATrainer.train_all_authors`: loops each author exactly once (`train_author` called once per author, no duplicate loop) — no double-training bug present in the current repo version.

**Part C — Data & results**
- **C10 (gitignore enforcement).** `git ls-files | grep -E '\.(pt|safetensors|pth|bin)$'` returns nothing. `models/`, `data/raw/` untracked and `check-ignore`-verified. `du -sh .git` = **19 MB** — well under the 200 MB flag threshold. (Note: a prior audit pass already identified a single ~17 MB historical blob, `models/epictetus/checkpoint-45/tokenizer.json`, a stock/reproducible Llama tokenizer file from a since-deleted contaminated checkpoint. Consistent with AUDIT.md's instruction, **no history rewrite was run or attempted.**)
- **C11 (results map completeness).** Cross-checked every file under `results/` against `results/README.md`'s pattern table — `bridge/`, `judges/`, `sweeps/`, `dilemmas/`, `dilemmas-v2/`, `dilemmas-v2/lora/`. Every file on disk matches a documented pattern, including the superseded pre-June-8 buggy-extraction sweeps. `results/comparisons/` is empty on disk and is documented as reserved/unused. No unmapped files.
- **C12 (data provenance).** `data/lora_training/README.md` (STALE-data warning) is present. For all three philosophers, both `neutral_pairs.json` (contaminated, Exp 8 baseline) and `neutral_pairs_clean.json` (Exp 9) exist side by side — the Exp 8→9 controlled comparison is intact.
