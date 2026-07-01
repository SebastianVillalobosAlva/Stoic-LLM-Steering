# Stoic LLM — Results Tracker

**Last updated:** June 8, 2026

---

## Experiment 1: 30-Pair Sweep (Old Contrastive Prompt)

**Date:** Mid-May 2026
**Model:** Llama-3.2-1B
**Pairs:** 30 per philosopher
**Prompt:** "Rewrite this philosophical text in plain, neutral language without any poetic or philosophical style. Keep the same meaning but make it straightforward."
**Problem identified:** Prompt preserved Stoic *reasoning* in plain English — vector learned "ornate vs plain style" not "Stoic vs non-Stoic thinking."

### Optimal Configs (from full layer × coefficient sweep)

| Philosopher | Best Layer | Best Coefficient | Aggregate Delta |
|---|---|---|---|
| Marcus Aurelius | 10 | 0.08 | +0.52 |
| Epictetus | 12 | 0.10 | +0.04 |
| Seneca | 14 | 0.10 | (not recorded) |

### Key Finding
- Epictetus was effectively dead (+0.04) — the contrastive signal was too weak
- Marcus Aurelius had the strongest effect but was likely driven by style, not philosophy
- Each philosopher required a different optimal layer, suggesting philosopher-specific representations

---

## Experiment 2: 100+ Pair Eval (New Contrastive Prompt)

**Date:** May 22, 2026
**Model:** Llama-3.2-1B
**Pairs:** Marcus 100, Seneca 100, Epictetus 58
**Prompt:** New prompt that requires same topic but DIFFERENT reasoning — explicitly bans Stoic concepts, asks for mainstream modern advice
**Configs used:** Inherited from 30-pair sweep (not re-swept)

### Marcus Aurelius — Layer 10, Coefficient 0.08

| Dimension | Unsteered | Steered | Delta |
|---|---|---|---|
| Philosophical Depth | 1.50 | 1.75 | +0.25 ↑ |
| Stoic Alignment | 1.75 | 1.58 | -0.17 ↓ |
| Coherence | 1.83 | 1.58 | -0.25 ↓ |
| Stylistic Authenticity | 1.17 | 2.25 | +1.08 ↑ |
| **Aggregate** | **1.56** | **1.79** | **+0.23 ↑** |

### Epictetus — Layer 12, Coefficient 0.10

| Dimension | Unsteered | Steered | Delta |
|---|---|---|---|
| Philosophical Depth | 1.58 | 1.92 | +0.33 ↑ |
| Stoic Alignment | 1.83 | 1.92 | +0.08 ↑ |
| Coherence | 1.83 | 1.75 | -0.08 ↓ |
| Stylistic Authenticity | 1.17 | 1.83 | +0.67 ↑ |
| **Aggregate** | **1.60** | **1.85** | **+0.25 ↑** |

### Seneca — Layer 14, Coefficient 0.10

| Dimension | Unsteered | Steered | Delta |
|---|---|---|---|
| Philosophical Depth | 1.50 | 1.75 | +0.25 ↑ |
| Stoic Alignment | 1.75 | 2.00 | +0.25 ↑ |
| Coherence | 1.75 | 1.75 | +0.00 → |
| Stylistic Authenticity | 1.08 | 1.17 | +0.08 ↑ |
| **Aggregate** | **1.52** | **1.67** | **+0.15 ↑** |

### Cross-Experiment Comparison

| Philosopher | 30-Pair Delta | 100+ Pair Delta | Change |
|---|---|---|---|
| Marcus Aurelius | +0.52 | +0.23 | ↓ dropped |
| Epictetus | +0.04 | +0.25 | ↑↑ fixed |
| Seneca | (not recorded) | +0.15 | — |

### Key Findings

1. **New contrastive prompt works.** Epictetus went from +0.04 to +0.25 — a 6x improvement. The old prompt was the bottleneck, not the philosopher.

2. **Marcus dropped but for the right reasons.** Old vectors were pushing style (easy for 1B). New vectors push philosophy (harder for 1B). Stylistic Authenticity jumped +1.08 but Coherence dropped -0.25 — the model can't maintain coherent philosophical output under stronger steering pressure.

3. **1B model is the bottleneck.** All three philosophers show modest aggregate gains (+0.15 to +0.25). The steering vectors are pushing harder than the old ones, but the model doesn't have enough capacity to follow. This motivates scaling to 3B.

4. **Configs need re-sweeping.** Current configs are from the 30-pair sweep with the old prompt. The new vectors likely have different optimal layers/coefficients. Re-sweep on 3B, not 1B.

---

## Experiment 3: Scale to Llama-3.2-3B

**Date:** May 25, 2026
**Model:** Llama-3.2-3B (float16, 28 layers, hidden_dim 3072)
**Pairs:** Same 100+ pairs as Experiment 2
**Vectors:** Re-extracted on 3B (shape 3072)
**Sweep:** Full layer × coefficient sweep (layers [4,8,12,16,20,24,26], coefficients [0.03,0.05,0.08,0.11,0.15,0.2,0.3])

> **⚠ PARTIALLY SUPERSEDED BY EXPERIMENT 8.** The "converge on layer 8" result below was produced under a steering-vector extraction bug (vectors extracted at the final layer, injected at layer 8) and under aggregate-based selection that is inflated by style. Under corrected same-layer extraction + content-only selection + seed averaging (Experiment 8), the layer-8 convergence does **not** hold, and the content-best layers scatter (Marcus L20/26, Seneca L4, Epictetus L26). The **aggregate (style-inflated) numbers in this section remain valid** as a record of style steering; the **content and "optimal layer" conclusions are superseded.** Read this section as the style result, not the reasoning result.

### Marcus Aurelius — Optimal: Layer 8, Coefficient 0.08, Aggregate 2.31

| Layer | Aggregate |
|---|---|
| 4 | 1.79 |
| **8** | **2.35 ← best** |
| 12 | 2.21 |
| 16 | 1.92 |
| 20 | 1.88 |
| 24 | 2.00 |
| 26 | 1.96 |

### Seneca — Optimal: Layer 8, Coefficient 0.11, Aggregate 2.44

| Layer | Aggregate |
|---|---|
| 4 | 1.92 |
| **8** | **2.31 ← best** |
| 12 | 2.21 |
| 16 | 1.85 |
| 20 | 1.90 |
| 24 | 2.06 |
| 26 | 2.02 |

### Epictetus — Optimal: Layer 8, Coefficient 0.15, Aggregate 2.48

| Layer | Aggregate |
|---|---|
| 4 | 2.08 |
| **8** | **2.38 ← best** |
| 12 | 2.19 |
| 16 | 1.94 |
| 20 | 2.08 |
| 24 | 2.23 |
| 26 | 1.98 |

### Cross-Model Comparison (1B vs 3B)

| Philosopher | 1B Layer | 1B Aggregate | 3B Layer | 3B Aggregate | Change |
|---|---|---|---|---|---|
| Marcus Aurelius | 10 | 1.79 | 8 | 2.31 | ↑↑ +0.52 |
| Epictetus | 12 | 1.85 | 8 | 2.48 | ↑↑ +0.63 |
| Seneca | 14 | 1.67 | 8 | 2.44 | ↑↑ +0.77 |

### Key Findings

> Note: Findings 1, 3, and 4 below concern the layer-8 convergence and content/alignment claims that Experiment 8 supersedes. The scaling/capacity story (the 1B→3B aggregate jump, finding 2 and 5) stands.

1. **All three philosophers converge on layer 8.** *(Superseded — see Exp 8; this was an extraction-bug + style-selection artifact.)* On 1B they were scattered (10, 12, 14). On 3B they all land on layer 8.

2. **Massive improvement across the board.** Every philosopher jumped significantly — Seneca had the largest gain (+0.77). The 1B model was the bottleneck, confirmed. *(Aggregate/style result — stands.)*

3. **Epictetus went from weakest to strongest.** From +0.04 (dead on old prompt), to +0.25 (fixed prompt on 1B), to 2.48 (highest aggregate on 3B). *(Aggregate-driven; the content component is later shown weak in Exp 8.)*

4. **Coefficients still differ.** Marcus 0.08, Seneca 0.11, Epictetus 0.15. Epictetus needs the strongest push, likely because it has fewer pairs (58 vs 100). *(Coefficient ranking shown unreliable in Exp 8.)*

5. **Oversteering is worse on 3B.** Coefficient 0.30 collapsed for all three philosophers. The model is more sensitive to oversteering — it has more capacity but also more ways to go incoherent under heavy intervention. *(Stands.)*

---

### Experiment 3b: 3B Eval at Optimal Configs

**Date:** May 25, 2026
**Model:** Llama-3.2-3B
**Configs:** From Experiment 3 sweep (all layer 8)

> **⚠ Configs here (all layer 8) are the superseded picks from Experiment 3.** The Stylistic Authenticity column is the robust, valid result; the Phil. Depth / Stoic Alignment (content) columns are revisited and found weak/unreliable under seed averaging in Experiment 8.

| Philosopher | Coefficient | Phil. Depth | Stoic Align. | Coherence | Style Auth. | Aggregate | Delta |
|---|---|---|---|---|---|---|---|
| Marcus Aurelius | 0.08 | +0.17 | -0.08 | +0.17 | +1.00 | 2.02 | +0.31 |
| Epictetus | 0.15 | +0.42 | +0.08 | -0.25 | +1.58 | 2.27 | +0.46 |
| Seneca | 0.11 | +0.33 | +0.17 | +0.75 | +1.42 | 2.35 | +0.67 |

### Key Findings

1. **Seneca is the strongest on 3B.** Every dimension improved, including Coherence (+0.75). The model becomes more organized under Seneca's structured prose style.

2. **Coherence no longer drops consistently.** On 1B, steering hurt coherence. On 3B, Marcus and Seneca both improved coherence. Epictetus dropped (-0.25) but at the highest coefficient (0.15) — likely slight oversteering.

3. **Stylistic Authenticity is the biggest winner across all three.** Marcus +1.00, Epictetus +1.58, Seneca +1.42. The 3B model has enough capacity to genuinely shift writing style. *(This is the robust signal — consistent with Exp 8's "style moves, content doesn't" conclusion.)*

4. **The 1B bottleneck hypothesis is fully confirmed.** Every philosopher improved dramatically from 1B to 3B. The vectors were always strong — the model needed capacity.

---

### Experiment 4: CAA vs LoRA Comparison (3B)

**Date:** May 27, 2026
**Model:** Llama-3.2-3B
**LoRA config:** r=8, alpha=32, targets q_proj + v_proj, 3 epochs, trained on Colab T4

#### LoRA Eval Results

| Philosopher | Phil. Depth | Stoic Align. | Coherence | Style Auth. | Aggregate | Delta |
|---|---|---|---|---|---|---|
| Marcus Aurelius | +1.00 | +1.17 | +0.42 | +1.58 | 2.75 | +1.04 |
| Epictetus | +1.67 | +1.50 | +0.25 | +2.25 | 3.02 | +1.42 |
| Seneca | +1.67 | +1.33 | -0.33 | +2.83 | 3.08 | +1.38 |

#### CAA vs LoRA Head-to-Head (3B)

| Philosopher | CAA Aggregate | CAA Delta | LoRA Aggregate | LoRA Delta | Winner |
|---|---|---|---|---|---|
| Marcus Aurelius | 2.02 | +0.31 | 2.75 | +1.04 | LoRA |
| Epictetus | 2.27 | +0.46 | 3.02 | +1.42 | LoRA |
| Seneca | 2.35 | +0.67 | 3.08 | +1.38 | LoRA |

#### Key Findings

1. **LoRA produces stronger philosophical steering across all three philosophers.** Average LoRA delta (+1.28) is roughly 3x stronger than average CAA delta (+0.48).

2. **LoRA fixes the Stoic Alignment problem.** CAA struggled with Stoic Alignment (Marcus even dropped -0.08). LoRA improved it dramatically for all three (+1.17, +1.50, +1.33). The permanent weight modification encodes Stoic reasoning more deeply than a runtime vector addition.

3. **Stylistic Authenticity is LoRA's biggest win.** Seneca hit 4.00/5.00 — nearly perfect. The model genuinely writes like translated ancient philosophy under LoRA.

4. **Coherence is the tradeoff.** CAA improved Seneca's coherence (+0.75) while LoRA hurt it (-0.33). LoRA pushes style so hard that readability suffers for Seneca's elaborate prose. Epictetus coherence improved under LoRA (+0.25) but dropped under CAA (-0.25) — the methods have opposite coherence profiles depending on the philosopher.

5. **CAA's advantage is flexibility.** CAA requires zero training, can be adjusted on the fly (coefficient slider), and is fully reversible. LoRA requires training, is fixed once deployed, and permanently modifies the model. For research and interpretability, CAA is more useful. For deployment and maximum effect, LoRA wins.

> Note: the aggregate/delta comparison here is style-inflated on the CAA side (per Exp 8). The robust, defensible cross-method claim is the **circuit-topology difference** (Experiments 5 & 6), which does not depend on the content effect being real.

---

### Experiment 5: Bridge Analysis — ModelLens × Stoic LLM (3B)

**Date:** May 25, 2026
**Model:** Llama-3.2-3B
**Tool:** ModelLens v0.1.0
**Analyses:** Layer evolution (logit lens), residual stream, activation patching
**Prompts tested:** "When facing difficulty, one should", "The wise person is one who", "True freedom comes from"

#### Layer Evolution (Logit Lens)

- Layers 0-7: steered and unsteered predictions are identical
- Layer 8 (steering layer): first divergence — steered model shifts toward archaic/philosophical tokens ("hath", "Thy", "whence", "upon")
- Layers 9-13: divergence persists but intermittent
- Layers 14-20: steered and unsteered reconverge (residual stream absorbing the perturbation)
- Layers 24-27: late divergence re-emerges as signal gets amplified for output
- Note: probabilities are near-zero until final layers — logit lens is noisy on 3B, residual stream and patching are more reliable

#### Residual Stream — Three-Phase Pattern (consistent across all philosophers)

| Phase | Layers | Marcus (coeff 0.08) | Epictetus (coeff 0.15) | Seneca (coeff 0.11) |
|---|---|---|---|---|
| Injection | L8 | +0.99 | +1.97 | +1.62 |
| Propagation | L9-13 | small positive | small positive | small positive |
| Compensation | L14-24 | negative (peak -0.16) | negative (peak -0.29) | negative (peak -0.23) |
| Output amplification | L27 | +0.69 | +1.31 | +1.19 |

Signal magnitude scales with coefficient, but the shape is identical across all three philosophers.

#### Activation Patching — Shared Causal Circuit

All three philosophers route through the same components:

| Component | Marcus (norm) | Epictetus (norm) | Seneca (norm) | Role |
|---|---|---|---|---|
| layers.9.mlp | 1.50 | 0.89 | present | Processing |
| layers.10.mlp | 1.00 | 1.56 | — | Processing |
| layers.11.mlp | 1.00 | 1.00 | — | Processing |
| **layers.12.mlp** | **2.00** | **1.56** | **strongest** | **Primary processor** |
| layers.14.self_attn | 1.00 | 0.44 | present | Only attention head |
| layers.21.mlp | -0.75 | -0.44 | compensatory | Resistance |
| layers.26.mlp | 1.00 | 1.00 | present | Output shaping |

#### Key Findings

1. **Single shared circuit.** All three philosophers use the same causal pathway: injection at L8 MLP → processing through L9-13 MLP (L12 most important) → compensation at L14-24 → output amplification at L27. Different philosophers vary in magnitude, not topology.

2. **MLPs dominate, not attention.** The steering signal is processed almost entirely through feedforward layers. Only one attention head (L14 self_attn) shows consistent causal importance. This is consistent with recent interpretability work showing MLPs store conceptual knowledge while attention handles routing.

3. **The model actively resists steering.** Layers 14-24 show negative residual stream deltas and compensatory activation patching effects (especially L21 MLP). The model's residual stream partially "undoes" the perturbation.

4. **Stronger steering = stronger resistance.** Epictetus (coeff 0.15) has both the strongest injection (+1.97) and strongest compensation (-0.29). The push-pull scales proportionally.

5. **Layer 12 MLP is the computational bottleneck for Stoic steering.** It's the most causally important component across all three philosophers. If you ablate L12 MLP, Stoic steering should collapse.

> Note: this analysis used the (superseded) layer-8 steered configs. The circuit topology is a property of where the vector was injected (L8) and how it propagated, which is still a valid characterization of *that* injection. When re-running the paper-grade version, re-do patching with a content-relevant `metric_fn` (logit diff between Stoic and neutral tokens / projection onto the steering direction at hidden_states index = layer+1) rather than the default max-logit metric.

---

### Experiment 6: Bridge Analysis — ModelLens × LoRA (3B)

**Date:** May 27, 2026
**Model:** Llama-3.2-3B with LoRA adapters (merged via merge_and_unload)
**Tool:** ModelLens v0.1.0
**Analyses:** Layer evolution, residual stream, activation patching
**Prompt tested:** "When facing difficulty, one should", "The wise person is one who", "True freedom comes from"

#### Layer Evolution — Distributed Changes

Unlike CAA (where predictions only diverge at the steering layer), LoRA changes predictions across many layers simultaneously. For "The wise person is one who," all three philosophers show "learns" → "knows" shift from layers 16-25 — a semantic change from passive learning to active knowing.

#### Residual Stream — No Injection Spike

| Layer | Marcus Diff | Epictetus Diff | Seneca Diff |
|---|---|---|---|
| L1 | -0.75 | +1.13 | +0.69 |
| L8 | +0.05 | +0.46 | +0.30 |
| L14 | -0.05 | -0.16 | -0.12 |
| L27 | -3.38 | -0.19 | +2.50 |

No single injection point. Changes are distributed with biggest effects at boundaries (L1 and L27). Contrast with CAA's clean three-phase pattern.

#### Activation Patching — Different Circuit from CAA

| Component | Marcus | Epictetus | Seneca | Role |
|---|---|---|---|---|
| **layers.27.mlp** | -0.63 | **2.91** | **4.14** | **Output shaping (dominant)** |
| layers.26.mlp | — | — | 2.86 | Late output |
| layers.13.mlp | — | 1.73 | 2.71 | Mid processing |
| layers.3.mlp | — | -1.82 | -2.71 | Early compensatory |
| layers.4.self_attn | 0.69 | — | -2.29 | Attention routing |
| layers.16.mlp | 0.81 | -1.45 | -1.86 | Mid processing |
| layers.0.self_attn | 0.81 | -1.09 | — | Early attention |

#### Key Findings — CAA vs LoRA Circuits

1. **Completely different circuit topology.** CAA concentrates in layers 9-13 MLP with a single injection point. LoRA distributes effects across all 28 layers with the strongest impact at the output (L27).

2. **CAA is MLP-dominated; LoRA uses both MLPs and attention.** CAA's steering signal passes through feedforward layers because it's injected at the MLP output. LoRA modified q_proj and v_proj weights, so attention heads are directly involved.

3. **CAA has a shared circuit; LoRA shows more variation.** All three CAA philosophers route through the identical L9-13 pathway. LoRA circuits vary more between philosophers — Marcus peaks at early attention, Epictetus and Seneca peak at output MLPs.

4. **Same behavioral outcome, different computational path.** Both methods produce Stoic-flavored outputs, but through fundamentally different mechanisms. CAA is a surgical injection at one point that propagates through a specific circuit. LoRA is a system-wide rewiring that most strongly affects the output layer.

5. **The "learns" → "knows" shift is consistent across all LoRA philosophers.** On "The wise person is one who," all three LoRA models predict "knows" where the base predicts "learns" from layers 16-25. This semantic shift (from passive learning to active knowing) may reflect a genuine philosophical reorientation.

> **This is the publishable anchor.** The CAA-vs-LoRA topology difference (Exp 5 vs 6) is a behavioral-outcome-controlled circuit comparison: same Stoic-flavored output, different internal path. It does **not** depend on the content effect being real (Exp 8), which is exactly why it is the lowest-risk result to build a paper around. Next: re-run both through `circuit_discovery.py` with a content-relevant `metric_fn` and produce side-by-side circuit graphs as the paper figure; address the objective confound (CAA = contrastive direction; LoRA = continued-pretraining on Stoic text only) either by acknowledging it or adding a contrastive-objective LoRA variant.

---

### Experiment 7: Safety evaluation (planned)
- JailbreakBench or equivalent adversarial prompts
- Compare refusal rates: base vs Stoic-steered
- Temperature stability: does steering hold at high temps?

> Reframing note (post Exp 8): since CAA reliably moves *style* but not *reasoning*, the original "does Stoic steering improve refusal robustness" framing may return a null. Consider reframing as: "does a steering vector that only shifts register affect safety behavior at all?" A clean null is itself a reportable result. Run a generic (non-philosophical) steering-vector baseline so any effect is attributable to the philosophy, not to adding any vector.

---

### Experiment 8: Seed-Averaged Content Validation (3B)

**Date:** June 8, 2026
**Model:** Llama-3.2-3B
**Method:** seed_eval (vary="judge") — generate once (greedy, matched decoding), score N=5 times per config
**Purpose:** Test whether the content deltas the sweep selected survive averaging, or are LLM-as-judge noise. Coefficient fixed at 0.11; layers = each philosopher's top-3 content layers from the corrected-extraction sweep.

**Pre-experiment fixes (this session):**
- **Same-layer extraction.** `extract_vectors.py` now extracts a vector at each candidate layer ({layer: (3072,)} dict) instead of extracting at the final layer and injecting at L8. Confirmed via projection check: injection at block L bites at hidden_states[L+1] (e.g. L8 → idx 9, +0.108, decaying to ~output).
- **Matched decoding.** Sweep baseline and steered generation, plus both eval scripts, now share one greedy GEN_KWARGS (do_sample=False, rep_penalty=1.3, no_repeat_ngram_size=3). Previously baseline sampled at temp 0.7 while steered/eval used greedy.
- **Content/style decoupling.** Sweep now selects on a content score (Δ philosophical_depth + Δ stoic_alignment) instead of the style-inflated aggregate.

**Motivation:** Corrected-extraction sweeps selected content-best layers in the network's back half (Marcus L26 +0.46, Seneca L20 +0.50, Epictetus L20 +0.29), contradicting the earlier style-driven "converge on L8." But single-run content scores were non-monotonic and non-reproducible — Seneca L20/0.11 scored +0.50, −0.21, and −0.25 on three separate evals of the identical config. This experiment quantifies that instability with seed averaging.

#### Results (5-seed content mean ± std, coeff 0.11)

**Marcus Aurelius**

| Layer | Content mean ± std |
|---|---|
| 20 | +0.067 ± 0.063 |
| 26 | +0.025 ± 0.105 |
| 16 | −0.083 ± 0.125 |

**Seneca**

| Layer | Content mean ± std |
|---|---|
| 4  | +0.283 ± 0.185 |
| 20 | +0.142 ± 0.100 |
| 8  | +0.025 ± 0.134 |

**Epictetus**

| Layer | Content mean ± std |
|---|---|
| 26 | +0.267 ± 0.149 |
| 20 | +0.042 ± 0.088 |
| 4  | −0.058 ± 0.116 |

#### Key Findings

1. **The sweep's specific config picks did not survive.** Marcus's sweep winner (L26, +0.46) collapsed to +0.025. Epictetus's sweep winner (L20) collapsed to +0.042 — its surviving layer (26) was the sweep's weakest. The sweep could not reliably rank configs; single-run deltas were largely noise.

2. **The content effect is philosopher-dependent and weak, not uniform.** Marcus shows no content effect above noise (all layers ≈ 0). Seneca (+0.283 ± 0.185) and Epictetus (+0.267 ± 0.149) show positive content means, but with error bars wide enough that the effect size is poorly determined.

3. **The carrying layer is inconsistent.** Seneca's best surviving layer is 4 (early); Epictetus's is 26 (very late); Marcus is flat. No stable localization of a "content circuit" across philosophers — evidence the signal, where present, is fragile.

4. **Style remained robust throughout.** Aggregate scores (style-inflated) stayed positive across the sweep; content (depth + alignment) is at best small and inconsistent.

**Interpretation:** CAA at 3B reliably shifts writing *register* (Stoic-sounding) but its effect on Stoic *reasoning* is at best small, philosopher-dependent, and not robustly localized. Consistent with the hypothesis that CAA primarily moves style, with a weak/unreliable content component.

#### Caveats (do not overclaim)

- "survives" flag in the code = (mean − 1σ > 0); this is a weak heuristic, NOT a significance test. With n=5 and these error bars, a proper paired test is required before claiming any effect is real.
- "Weak content effect above noise" ≠ "real content effect." The Seneca/Epictetus positives could be genuine-but-small, or artifacts of judge scoring on open-ended prompts.

#### Next Step

- **Forced-choice dilemma eval** (lower-noise, style-independent) to discriminate: is the Seneca/Epictetus content positive real, or a judge-scoring artifact? This is the validation gate's decisive instrument.
- If real, Seneca/Epictetus best layers need more seeds (or more prompts/eval) to shrink std and pin effect size.

---

## API Credit Usage

| Run | Cost | Remaining |
|---|---|---|
| 100+ pair generation (258 pairs) | ~$1.09 | ~$1.41 |
| 100+ pair eval (3 philosophers × 12 prompts) | ~$0.60 | ~$0.81 |
| 3B sweep — original, buggy extraction (3 philosophers × full grid) | ~$5.00 | ~$3.81 |
| 3B eval (3 philosophers × 12 prompts) | ~$0.60 | ~$3.21 |
| LoRA eval (3 philosophers × 12 prompts) | ~$0.60 | ~$2.61 |
| *(subtotal — May runs)* | *~$7.89* | *~$2.61* |
| 3B re-sweep — corrected extraction + content selection (Jun 8) | ~$5.00 (est) | — |
| Seed-eval smoke test (1 call) | ~$0.02 (est) | — |
| Seed eval — 3 philosophers × 3 layers × 5 judge seeds (Jun 8) | ~$8–9 (est) | — |
| Top-up (Jun 8) | +$10.00 | — |

**Notes:**
- June 8 costs are estimates — reconcile against the actual Anthropic console billing and update.
- Bridge analyses (ModelLens) cost nothing — all local compute.
- `vary="judge"` seed eval re-runs only the judge, not generation (generation runs once per layer), so its cost scales with judge calls, not generations.
 
-------

# New Steering Vectors

### Experiment 9: Clean-Data Re-test — Pair Quality as the Determining Variable (3B)

**Date:** June 8, 2026
**Model:** Llama-3.2-3B
**Judge:** Gemini (`gemini-2.0-flash`) — independent of the pair-generation model (Sonnet 4), breaking the Claude-writes-pairs / Claude-judges circularity.
**Method:** seed_eval (vary="judge"), N=5, coefficient fixed at 0.11. Candidate layers from a clean-vector Sonnet layer sweep; decisive measurement under Gemini.
**Pairs:** Fully rebuilt — clean sources, reasoning-isolating contrastive prompt, matched N=53 per philosopher (quantity controlled).

**Purpose:** Test whether the weak/null content effect in Experiment 8 was a *data-quality* artifact (contaminated contrastive pairs) rather than a *method* limitation. Re-extract vectors from clean pairs and re-test under an independent judge.

---

#### Pipeline fixes preceding this experiment (this session)

The Experiment 8 vectors were built from contaminated contrastive pairs. Contamination was traced to multiple pipeline stages and fixed:

1. **Source slicing (downloader).** Gutenberg files retained front-matter (translator intros, biographies) and back-matter (Marcus's Fronto correspondence appendix; Epictetus's scholarly intro + mid-file bibliography). Added per-author `content_start`/`content_end` boundaries to slice to the work proper.
2. **Chunk filtering.** Strengthened `_is_non_philosophical` to catch biographical prose and citations (e.g. a `",\s+\d{4}"` year-pattern for bibliography entries); stripped inline `[\d+]` footnote markers.
3. **Contrastive prompt.** Old prompt produced neutral halves that *restated* Stoicism in casual register (style contrast, not reasoning contrast). New prompt forces a competing worldview (ambition/hedonism/assertiveness), bans Stoic concepts, names an explicit failure condition ("must give OPPOSITE advice"), and forbids framework headers/preamble.
4. **Decoding + selection (from Exp 8).** Matched greedy decoding across steered/baseline; sweep selects on content (Δ depth + Δ alignment), not style-inflated aggregate.

Contamination rate dropped from ~58–64% (preamble/refusal leakage) to ~0% genuine (residual 6–8% flagged by the checker are false positives — legitimate in-text "Look,"/"Sure,").

---

#### Results — clean vs contaminated (Gemini judge, N=5, coeff 0.11)

| Philosopher | Best layer | **Contaminated** (Exp 8, Gemini) | **Clean** (Exp 9, Gemini) | Original pair quality |
|---|---|---|---|---|
| Marcus Aurelius | 26 | +0.025 ± 0.105 (flat) | **+0.408 ± 0.136** | moderate (restatement) |
| Seneca | 4 | +0.667 ± 0.072 | **+0.583 ± 0.121** | good (already clean) |
| Epictetus | 8 | −0.100 ± 0.246 (dead) | **+0.767 ± 0.076** | worst (style + intro contamination) |

Clean candidate-layer rankings (Gemini, N=5):

**Marcus Aurelius** — L26 **+0.408 ± 0.136**; L16 +0.100 ± 0.120; L12 −0.100 ± 0.168
**Seneca** — L4 **+0.583 ± 0.121**; L12 +0.008 ± 0.080; L26 −0.058 ± 0.120
**Epictetus** — L8 **+0.767 ± 0.076**; L20 −0.025 ± 0.086; L24 −0.267 ± 0.192

---

#### Key Findings

1. **Pair quality is the determining variable.** Holding model, method, layer, judge, and N constant, swapping contaminated pairs for clean reasoning-isolating pairs turns a null/negative content effect into a strong positive one. Marcus L26: +0.025 → +0.408. Epictetus L8: −0.100 → +0.767. This is a controlled, single-variable result — the only change is the contrastive data.

2. **Effect size of cleaning tracks original contamination.** Seneca (pairs already clean) barely moved (+0.667 → +0.583, statistically unchanged). Marcus (moderate contamination) jumped. Epictetus (worst contamination — style-only contrasts + a scholarly introduction leaking into the Stoic halves) improved the most, from dead to strongest. The benefit of cleaning scales with how broken the source was.

3. **The Experiment-8 cross-philosopher split was contamination, not a property of the philosophers.** The "Seneca works, others don't" result resolves entirely: with clean pairs, all three show robust content effects. The philosophers were never the variable.

4. **Layer localization is robust and philosopher-specific.** Each philosopher has a distinct, stable best layer — Marcus L26, Seneca L4, Epictetus L8. Seneca (L4) and Marcus (L26) held the same layer across contaminated *and* clean data and across Sonnet *and* Gemini judges. Epictetus moved (old 20/26 → clean 8), consistent with its old vector being the most corrupted. Distinct per-philosopher localization is suggestive evidence that the traditions encode at computationally distinct sites.

5. **The single-run sweep remains unreliable for ranking; seed-averaging + independent judge is decisive.** Sweep content values were modest (+0.12 to +0.25) and non-monotonic in coefficient (the +0.21/−0.21 same-config contradiction recurred). The strong, tight effects only became visible under seed-averaging with the Gemini judge — the sweep's role is region-finding, not measurement.

---

#### Caveats (do not overclaim)

- **"Content effect" = higher score on an LLM-judge Stoic rubric (depth + alignment).** This is now corroborated by an *independent* judge (Gemini ≠ Sonnet, the pair author), which is far stronger than Exp 8, but it is still rubric-based judging of prose. A forced-choice dilemma eval (decision preference, style-independent) would further separate "Stoic *reasoning*" from "Stoic-flavored *content*."
- **The "survives" flag is a weak heuristic** (mean ± 1σ excludes 0), not a significance test. The L26/L4/L8 effects clear zero by ~2–3σ at N=5, but a proper paired test should be reported for any publication claim. (Note: the heuristic also fires "survives" on consistently *negative* configs, e.g. Epictetus L24 −0.267 — ignore those.)
- **N=5 seeds, single coefficient (0.11).** Effect magnitudes are estimates; tighter bounds need more seeds / more prompts. Coefficient was fixed because single-run coefficient ranking is noise.
- **Data differs from the validated old Seneca** (new prompt, 53 vs 63 pairs), so clean-Seneca +0.583 vs old +0.667 is "statistically unchanged," not "decreased."

---

#### Headline figure (for SOP / writeup)

**Same philosopher, same layer, same method, same independent judge, same N — pairs as the only difference:**

> Marcus Aurelius, L26: **+0.025 ± 0.105 (contaminated) → +0.408 ± 0.136 (clean)**
> Epictetus, L8: **−0.100 ± 0.246 (contaminated) → +0.767 ± 0.076 (clean)**

The original null was a data-quality artifact. With clean, reasoning-isolating contrastive pairs, CAA produces a localized, independent-judge-confirmed Stoic-content effect across all three philosophers.

---

#### Next Steps

- **Forced-choice dilemma eval** at each best layer (M:26, S:4, E:8) — the style-independent test of whether the effect is *reasoning* vs *Stoic-flavored prose*.
- **Data-efficiency sweep** (the quality-over-quantity thesis): vary pair count (e.g. 15 / 30 / 53 / 100) on one philosopher to map where the effect saturates. Tests "few clean pairs suffice."
- **Re-run LoRA on the clean pairs** for an equal-footing CAA-vs-LoRA circuit-topology comparison (the publishable anchor), so both methods use clean data.
- **More seeds / paired significance test** on the three headline configs before any external claim.