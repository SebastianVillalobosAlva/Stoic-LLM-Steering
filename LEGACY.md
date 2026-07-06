# LEGACY.md — frozen provenance record

This repository is **frozen**. It is the evidence trail for Experiments 1–11.
Nothing here is edited to "fix" old bugs — doing so would destroy its value as a
record. This file documents a single root-cause defect discovered during the
clean rebuild (see `stoic-llm` / `llm-stoic`, Pass A Stage 3) and maps every
experiment it contaminated.

---

## Root cause: the CAA content artifact was an asymmetric-decoding bug

**One bug, one mechanism, several contaminated experiments.**

Every judge-scored **CAA content** result compared a **steered** generation that
was silently **sampled and truncated to ~13 tokens** against a **baseline** that
was **greedy and 100 tokens long**. The judge was reacting to the decoding
difference, not to the steering vector. Under genuinely matched decoding the CAA
content effect collapses to ≈ 0 (see "Empirical confirmation" below).

### Where the asymmetry comes from (exact lines)

The steered generation for every CAA content eval routes through
`SteeringRunner.run_model_with_hook`, which forwards **only** the keyword
arguments the caller passes and nothing else:

`stoic_llm/steering/runner.py:102-116`
```python
def run_model_with_hook(self, return_output=False, **generate_kwargs):
    ...
    outputs = self.model.generate(**inputs, **generate_kwargs)   # line 115
```
It never uses `self.do_sample`, `self.temperature`, or `self.max_tokens` — so the
`do_sample=False, temperature=0.0` passed to the `SteeringRunner` constructor are
**dead arguments**. Every CAA content caller passes only the repetition controls:

- `scripts/run_eval.py:68-72` (Exp 3 / 3b, and the CAA arm of Exp 4)
  ```python
  steered = runner.run_model_with_hook(
      return_output=True,
      repetition_penalty=1.3,
      no_repeat_ngram_size=3,
  )   # no do_sample, no max_new_tokens
  ```
- `stoic_llm/eval/sweep.py:70-74` (`_run_steered`, used by Exp 8 seed eval, Exp 9,
  and all layer/coeff sweeps in `run_sweep.py`)
  ```python
  outputs = runner.run_model_with_hook(
      return_output=True,
      repetition_penalty=1.3,
      no_repeat_ngram_size=3,
  )   # same — no do_sample, no max_new_tokens
  ```

With `do_sample` and `max_new_tokens` absent, `model.generate` falls back to the
model's `generation_config`. For `meta-llama/Llama-3.2-3B` that config is
(verified empirically on the frozen weights):

```
do_sample = True,  temperature = 0.6,  top_p = 0.9,  max_length = 20
```

So **steered generation was sampled (temp 0.6, top-p 0.9) and capped at
`max_length=20` (~13 new tokens past a ~7-token prompt).**

The **baseline** in those same scripts is explicitly greedy and long:

- `scripts/run_eval.py:82-89`
  ```python
  outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                           temperature=0.0, repetition_penalty=1.3,
                           no_repeat_ngram_size=3)
  ```
- `stoic_llm/eval/sweep.py:50` → `self.model.generate(**inputs, **GEN_KWARGS)`,
  where `GEN_KWARGS` (`stoic_llm/config.py:8-13`) is
  `max_new_tokens=100, do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3`.

### The asymmetry, side by side

| | Steered / adapted | Baseline |
|---|---|---|
| **CAA content** (run_eval, sweep, seed_eval) | **sampled** (temp 0.6, top-p 0.9), **max_length 20** (~13 tok) | **greedy**, **100 tok** |

The delta the judge scored is dominated by "short sampled snippet vs long greedy
web-text," not by the steering direction.

---

## Empirical confirmation (clean rebuild, Pass A Stage 3)

The clean rebuild enforces one decoding config for both sides
(`generate()` defined once). Re-running the Exp 9 content eval with **matched
greedy** decoding on both steered and baseline, same frozen vectors (cosine 1.0000
vs the reference `.pt`), same Gemini judge, n_seeds=5:

| Author | Layer | Exp 9 (asymmetric) | Clean rebuild (matched greedy) |
|---|---|---|---|
| Marcus | 26 | **+0.408 ± 0.136** | **−0.175 ± 0.054** |
| Seneca | 4 | **+0.583 ± 0.121** | **−0.117 ± 0.054** |
| Epictetus | 8 | **+0.767 ± 0.076** | **−0.117 ± 0.080** |

Under matched greedy the steered text is **byte-identical to baseline** at coeff
0.11 (a coefficient sweep shows greedy output only visibly changes at coeff ≥ 1.0),
so the residual signal is pure judge noise. The headline CAA content effect does
not survive removal of the decoding asymmetry.

A matched-**sampled** test (both sides sampled at temp 0.6 / top-p 0.9 / 100
tokens, paired by seed, n=5 seeds, same judge) also returned null — the effect is
absent at the distribution level too, not just at the greedy argmax:

| Author | Layer | Matched sampled |
|---|---|---|
| Marcus | 26 | +0.075 ± 0.165 (n.s.) |
| Seneca | 4 | −0.042 ± 0.204 (n.s.) |
| Epictetus | 8 | −0.192 ± 0.329 (n.s.) |

All ±1σ intervals include zero. **Verdict: the CAA content effect at coeff 0.11
is a decoding-asymmetry artifact, full stop** — it does not exist under matched
greedy or matched sampled decoding.

### The style/register claim collapses too

Exp 3b's stylistic-authenticity deltas — flagged in the tracker as "the robust,
valid result" — were re-tested the same way (canonical clean configs, matched
decoding both regimes, n=5 seeds, pre-registered survives-if->2σ rule):

| Author | Exp 3b style (asymmetric) | Matched greedy | Matched sampled |
|---|---|---|---|
| Marcus | +1.00 | −0.150 ± 0.181 | −0.033 ± 0.139 |
| Seneca | +1.42 | −0.100 ± 0.091 | −0.017 ± 0.260 |
| Epictetus | +1.58 | +0.050 ± 0.126 | −0.017 ± 0.124 |

Nothing survives. The "style moves robustly" signal was the judge scoring short
sampled snippets as more aphoristic than 100-token greedy prose — the same
artifact. **Under fair measurement, CAA at the canonical coefficient moves
nothing at any of the three levels (style, content, decisions).** Caveat: tested
at c=0.11 (canonical); the tracker's Exp 3b configs (all-L8, c=0.08–0.15) and
higher coefficients were not re-swept under matched decoding.
(Record: `stoic-llm` repo, `results/style_validation/style_20260705_212411.json`.)

---

## Experiments contaminated by this one bug

All are **CAA judge-scored content** evals that generate steered text through
`run_model_with_hook`:

| Exp | What | Script / path | Status |
|---|---|---|---|
| 1, 2 | 1B layer/coeff content sweeps | `run_sweep.py` → `sweep.py` (`_run_steered`) | superseded — decoding-asymmetric (also pre-Exp-8 extraction bug) |
| 3, 3b | 3B content at "optimal" configs | `run_eval.py` | superseded — decoding-asymmetric |
| 4 (CAA arm only) | CAA-vs-LoRA head-to-head, CAA deltas (+0.31/+0.46/+0.67) | `run_eval.py` | CAA side inflated — tracker already flags "style-inflated on the CAA side" |
| 8 | Seed-averaged content validation | `run_seed_eval.py` → `sweep.py` | superseded — **claimed matched decoding but wasn't** (see note) |
| 9 | Clean-pairs content re-test (+0.408/+0.583/+0.767) | `run_seed_eval.py` → `sweep.py` | superseded — decoding-asymmetric; the +0.767 headline is the artifact |

### Note on Exp 8's "matched decoding" claim

The tracker's Exp 8 entry states the seed eval generates "once (greedy, matched
decoding)" and that Exp 8 fixed a prior mismatch ("Previously baseline sampled at
temp 0.7 while steered/eval used greedy"). **The frozen code contradicts this.**
Exp 8 corrected the *baseline* side to greedy `GEN_KWARGS`, but the *steered* side
still runs through `run_model_with_hook`, which silently reverts to the sampled
`generation_config` defaults. The fix moved the asymmetry instead of removing it,
so Exp 8 and Exp 9 — the experiments that advertised matched decoding — are
themselves contaminated. (There appear to have been two decoding bugs across the
code's history; what is frozen now is steered-sampled / baseline-greedy.)

---

## NOT contaminated by this bug

- **Exp 4 LoRA arm** (`run_lora_eval.py`). This script passes `do_sample=False`
  through to `LoRARunner.generate`, whose `gen.update(generate_kwargs)`
  (`stoic_llm/lora/runner.py:38-57`) lets that override win → **greedy, 100 tok on
  both sides** (`run_lora_eval.py:26-49`, comment line 25: "Identical decoding on
  both sides"). The LoRA content deltas (+1.04/+1.42/+1.38) are from symmetric
  decoding. (Separate caveat, per the script header: it resolves adapters at
  `models/{size}/{author}/`, the OLD contaminated-data adapters, not the clean
  ones — a data-provenance issue, not a decoding one.)
- **Decision-level dilemma evals, Exp 10 & 11** (`run_dilemma_eval.py`,
  `run_lora_dilemma_eval.py`). These take a **single forward pass** and compare
  next-token logits over `{A, B}` — **no generation at all** — so decoding settings
  cannot affect them. The CAA-flat / LoRA-moves decision split is immune to this
  bug and stands.
- **Bridge / circuit-topology results, Exp 5 & 6.** Interpretability over
  activations, not judge-scored generation; independent of this artifact.

---

## Bottom line

The Exp 3/3b/8/9 CAA judge-scored effects — **both content and style** — and the
CAA arm of Exp 4 are a single decoding-asymmetry artifact (sampled+truncated
steered vs greedy+long baseline), introduced by `run_model_with_hook` dropping
`do_sample`/`max_new_tokens`. Under matched decoding, CAA at the canonical
coefficient moves nothing at any level. The decision-level results (Exp 10/11),
the LoRA content deltas (Exp 4 LoRA arm), and the circuit-topology comparison
(Exp 5/6) are **not** affected by it.
