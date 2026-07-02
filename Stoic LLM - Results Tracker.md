# Stoic LLM — Results Tracker

**Last updated:** July 1, 2026

---

## Current State of the Evidence (read this first)

Eleven experiments, one question: can Stoic reasoning be steered into an LLM — and at what depth does each method actually operate? Effects are measured at three levels: **style** (judge-scored register), **content** (judge-scored reasoning in prose), and **decision** (judge-free forced choice).

| Level | CAA (activation steering) | LoRA (weight adaptation, clean data) |
|---|---|---|
| Style / register | Robust (Exp 3b: +1.0–1.6) | Robust, stronger (Exp 4: +1.6–2.8) |
| Content / prose | Strong with clean pairs (Exp 9: +0.408 to +0.767, independent judge) | Not re-scored on clean adapters |
| **Decision / choice** | **None, at any coefficient up to 1.5** (Exp 10) | **Structured effects** (Exp 11): Seneca heavy-tailed shift; Marcus passivity prior; Epictetus null |

**The three headline results:**

1. **Pair quality is the determining variable for CAA content** (Exp 9). Controlled single-variable result: swapping contaminated for clean contrastive pairs flipped nulls to strong positives (Epictetus L8: −0.100 → +0.767, independent Gemini judge).
2. **CAA is a register direction, not a decision direction** (Exp 10). Calibrated forced-choice instrument (baseline 0.542), flat at every coefficient to the edge of incoherence. Three instruments converge on this boundary.
3. **LoRA reaches the decision layer where CAA does not** (Exp 11) — as the circuit topology (Exp 5 vs 6) predicted: CAA's mid-MLP injection is compensated away; LoRA's output-layer rewiring is not. But what LoRA installs is not uniform Stoic decision-making: Marcus = broad passivity prior (accepting 18+/4−, p=0.004; active flat), Seneca = heavy-tailed minority-of-items effect (t=2.39, sign test n.s.), Epictetus = null (smallest corpus).

**Open confounds (deliberately carried, documented in situ):** CAA contrastive objective vs LoRA continued-pretraining objective (method + objective jointly); corpus size perfectly confounded with philosopher identity; possible Senecan-idiom lexical echo in dilemma option wording (Exp 11 Amendment 2 — next experiment).

**Canonical configs:** Llama-3.2-3B; clean CAA vectors at Marcus L26 / Seneca L4 / Epictetus L8, coeff 0.11; clean LoRA adapters `lora_{author}_clean` (r=8, α=32, q+v, 3 epochs); dilemma set v2 (40 items, baseline 0.542).

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

## Experiment 3b: 3B Eval at Optimal Configs

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

## Experiment 4: CAA vs LoRA Comparison (3B)

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

## Experiment 5: Bridge Analysis — ModelLens × Stoic LLM (3B)

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

## Experiment 6: Bridge Analysis — ModelLens × LoRA (3B)

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

## Experiment 7: Safety evaluation (planned)
- JailbreakBench or equivalent adversarial prompts
- Compare refusal rates: base vs Stoic-steered
- Temperature stability: does steering hold at high temps?

> Reframing note (post Exp 8): since CAA reliably moves *style* but not *reasoning*, the original "does Stoic steering improve refusal robustness" framing may return a null. Consider reframing as: "does a steering vector that only shifts register affect safety behavior at all?" A clean null is itself a reportable result. Run a generic (non-philosophical) steering-vector baseline so any effect is attributable to the philosophy, not to adding any vector.

> **Status update (July 1):** still queued. Post Exp 10/11, the sharpest framing is now two-armed: (a) CAA arm — "does a register-only vector affect refusal behavior at all?" (clean null expected and reportable); (b) LoRA arm — "does an adapter that reaches decisions also shift refusal robustness?" Run with a generic-vector / generic-text-adapter baseline so any effect is attributable to the philosophy.

---

## Experiment 8: Seed-Averaged Content Validation (3B)

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

> **Status: DONE.** The forced-choice dilemma eval was built and run as Experiment 10; it resolved the discrimination decisively (CAA content positives do not translate to decision shifts — see Exp 10 synthesis). The seeds/effect-size follow-up was superseded by the clean-pairs re-test (Exp 9).

- **Forced-choice dilemma eval** (lower-noise, style-independent) to discriminate: is the Seneca/Epictetus content positive real, or a judge-scoring artifact? This is the validation gate's decisive instrument.
- If real, Seneca/Epictetus best layers need more seeds (or more prompts/eval) to shrink std and pin effect size.

---

## Experiment 9: Clean-Data Re-test — Pair Quality as the Determining Variable (3B)

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

- **Forced-choice dilemma eval** at each best layer (M:26, S:4, E:8) — the style-independent test of whether the effect is *reasoning* vs *Stoic-flavored prose*. **[DONE — Exp 10. Result: no decision-level effect from CAA at any coefficient.]**
- **Data-efficiency sweep** (the quality-over-quantity thesis): vary pair count (e.g. 15 / 30 / 53 / 100) on one philosopher to map where the effect saturates. Tests "few clean pairs suffice." **[PENDING]**
- **Re-run LoRA on the clean pairs** for an equal-footing CAA-vs-LoRA circuit-topology comparison (the publishable anchor), so both methods use clean data. **[DONE — clean adapters trained (Exp 11 header); behavioral comparison complete (Exp 11); clean-adapter circuit re-analysis via ModelLens still PENDING.]**
- **More seeds / paired significance test** on the three headline configs before any external claim. **[PENDING for Exp 9 content configs; done for the decision-level results (Exp 11 + amendments).]**

---

## Experiment 10: Forced-Choice Dilemma Evaluation (Decision-Level Test)

**Date:** June 24, 2026
**Model:** Llama-3.2-3B
**Method:** Judge-free, generation-free forced-choice probe. Each dilemma presents a situation with two options (one Stoic-consistent, one a defensible non-Stoic alternative) labeled A/B. Measure `P(stoic) = softmax over the two label tokens {A,B}` on the next token from a single forward pass. Every item is run in BOTH label orders and averaged, which cancels positional/label bias exactly. Steering applied via the same forward hook as `SteeringRunner` (MLP output at layer L).
**Purpose:** The validation gate's decisive instrument. Exp 8/9 tested *content* via an LLM judge on open-ended prose; this tests whether steering shifts an actual *decision*, with no judge in the loop and no dependence on style.
**Configs:** Marcus L26, Seneca L4, Epictetus L8 (the Exp 9 clean-pairs best layers), coeff 0.11.
**Cost:** $0 — all local CPU forward passes, no API calls.

---

#### 10a — Dilemma set v1 (FAILED CALIBRATION — ceiling effect)

First dilemma set paired the Stoic option against a non-Stoic option that *described bad behavior from the outside* (e.g. "stew about it for a week"). Any instruction-adjacent model rejects those regardless of philosophy.

| | Baseline mean P(stoic) | Result |
|---|---|---|
| v1 set | **0.881** | Ceiling effect — no headroom to measure |

Steered deltas (v1): Marcus +0.000 (t=1.14), Seneca +0.004 (t=6.19), Epictetus +0.001 (t=2.97).

> **The v1 deltas are uninterpretable**, NOT a null. Baseline 0.881 means the model already maxes the Stoic option; the test measured "prefers sensible-sounding advice," which was saturated. Seneca's t=6.19 on a +0.004 move is still useful information — it proves the **instrument is precise** (resolves sub-percent effects); the target was the problem, not the ruler.

#### 10b — Dilemma set v2 (calibrated, competing-values)

Rebuilt so the non-Stoic option is *genuinely good advice* drawn from values that compete with Stoicism: healthy emotional expression, self-advocacy, legitimate ambition, self-care/boundaries, persistence, attachment. 40 items, 23 accepting / 17 active stance.

| | Baseline mean P(stoic) | Gate |
|---|---|---|
| v2 set | **0.542** | PASS (target 0.35–0.65 with spread) |

**Steered results, coeff 0.11:**

| Philosopher | ΔP(stoic) | Δlog-odds | t(lo) | accepting (ΔP) | active (ΔP) |
|---|---|---|---|---|---|
| Marcus | −0.000 | +0.000 | 0.35 | −0.000 | +0.000 |
| Seneca | +0.000 | +0.006 | 0.83 | +0.001 | −0.001 |
| Epictetus | −0.001 | −0.002 | −0.71 | −0.001 | −0.001 |

All three flat. Every t under 1. Both stance buckets flat for all three. **Clean null at a calibrated baseline** — unlike v1, this null is real.

#### 10c — Epictetus coefficient sweep (forecloses understeering objection)

Epictetus chosen as the highest-signal probe (strongest Exp 9 content effect +0.767; historically needed the highest coefficient). Generation-free, so high coefficients are safe — only the option logits need to move.

| Coeff | P(stoic) | ΔP | Δlog-odds | t(lo) |
|---|---|---|---|---|
| 0.11 | 0.541 | −0.001 | −0.002 | −0.71 |
| 0.20 | 0.541 | −0.001 | −0.002 | −0.34 |
| 0.40 | 0.540 | −0.002 | −0.006 | −0.48 |
| 0.80 | 0.537 | −0.005 | −0.022 | −0.83 |
| 1.50 | 0.530 | −0.011 | −0.063 | −1.16 |

---

#### Key Findings

1. **CAA does not move decisions at any strength.** Even the strongest-signal vector, pushed to coeff 1.5 (edge of incoherence), produces no shift toward the Stoic option. The understeering objection is foreclosed.

2. **The drift is monotonic and slightly NEGATIVE, not random.** P(stoic) slides 0.541 → 0.530 as coefficient rises, with Δlog-odds growing steadily more negative and |t| increasing. The smooth, dose-responsive shape rules out degeneration (which would scatter P toward 0.5 erratically) — the A/B logits still track the options; steering just pushes the wrong way, weakly. Not significant (t never clears 1.2), but directionally against the hypothesis.

3. **The instrument is validated.** Same harness resolved a +0.004 effect at t=6.19 in v1. It reads ~0 here because there is ~0 to read, not from lack of sensitivity.

#### Cross-Instrument Synthesis (the headline)

> **Extended by Exp 11:** this table characterizes CAA alone; the two-method version (CAA vs LoRA) in Exp 11 supersedes it as the current synthesis, and Exp 11's Amendment 1 revises the LoRA column's strength.

Three independent instruments now converge on the same boundary for the Stoic CAA direction:

| Level | Instrument | Result |
|---|---|---|
| **Style / register** | Exp 3b (judge, Style Auth.) | Robust: +1.0 to +1.6 |
| **Content / reasoning** | Exp 8/9 (judge, content score) | Weak, philosopher-dependent, wide error bars (Exp 9 clean: Marcus L26 +0.408, Epictetus L8 +0.767) |
| **Decision / choice** | Exp 10 (judge-free, forced choice) | None at any coefficient; slightly negative |

> **The Stoic CAA direction is a register direction, not a decision direction.** It changes how the model *talks*, weakly/unreliably shifts how it *reasons* in prose, and does not change what it *chooses*. Two judge-free instruments and one judge-based instrument agree. This is a stronger, more defensible claim than any single positive result — a clean characterization of the limits of activation steering.

#### Caveats

- A flat CAA sweep does not yet imply *no method* reaches decisions — that requires the LoRA eval (below). Do not generalize "CAA is register-only" to "the philosophy is style-only" until LoRA is tested.
- The slight negative drift is below significance; report it as "no movement toward Stoic (if anything, marginally against)," not as a real anti-Stoic effect.
- Forced-choice A/B is a mildly unnatural probe for a base model; the calibrated 0.542 baseline with per-item spread is what licenses trust in the deltas.

#### Next Step

> **Status: DONE — Exp 11.** The prediction resolved in the interesting direction: LoRA moves decisions where CAA doesn't, with structure (see Exp 11 + amendments).

- **LoRA dilemma eval (the thesis test).** Run this exact harness on the merged LoRA models. Exp 6 predicts a possible split: LoRA's strongest causal effect sits at the output layer (L27 MLP; "learns→knows" shift L16–25) where decisions form, while CAA's circuit lives in mid-MLPs and is partially compensated away (Exp 5 L14–24 resistance). **If LoRA moves decisions where CAA doesn't, the circuit topology predicts the behavioral split** — a mechanism plus a behavioral consequence. If LoRA is also flat, the decision-level null is general at 3B and the safety-mechanism claim retires to "register intervention" rather than relocating to LoRA.

---

## Experiment 11: LoRA Dilemma Evaluation — Decision-Level Test, Clean-Data Adapters

**Date:** June 30, 2026
**Model:** Llama-3.2-3B base + clean-data LoRA adapters (merged via merge_and_unload, fresh base per adapter)
**Method:** Identical harness to Exp 10 (forced-choice, judge-free, both label orders averaged, same 40-item v2 dilemma set, same PROMPT_TEMPLATE). The only change vs the CAA condition: no steering hook — the adapter weights carry the intervention. Baseline computed on the unmodified base; **identical baseline basis as the CAA run (0.542, reproduced exactly)**.
**Adapters:** Retrained on Exp-9-cleaned chunked text (Colab T4), recipe held identical to Exp 4 (r=8, alpha=32, q_proj+v_proj, 3 epochs) so the only variable vs the old adapters is clean data. Training corpus sizes: Marcus 437 chunks (*Meditations*), Seneca 540 (*Moral Letters*), Epictetus 123 (*Enchiridion* only).
**Integrity:** Reuse-the-base merging was caught mutating the base in an aborted first run (PEFT `peft_config` stacking warning); rerun with fresh-base-per-adapter. Base integrity check: start 0.542 | end 0.542 | drift 0.0000 — clean.
**Cost:** $0 — local CPU forward passes; adapter training on free Colab T4.

#### Results (vs base baseline P(stoic) = 0.542)

| Philosopher | ΔP(stoic) | Δlog-odds | t(lo) | accepting (ΔP) | active (ΔP) | Read |
|---|---|---|---|---|---|---|
| **Seneca** | **+0.061** | **+0.308** | **2.39** | **+0.078** | **+0.039** | **Decision shift — both buckets positive** |
| Marcus | +0.031 | +0.161 | 2.18 | +0.065 | −0.011 | Passivity pattern — accepting only, active negative |
| Epictetus | +0.000 | +0.003 | 0.18 | +0.005 | −0.005 | Flat |

#### Key Findings

1. **LoRA reaches the decision where CAA does not.** On the identical instrument where CAA was flat at every coefficient up to 1.5 (Exp 10c), LoRA moves the choice. Seneca: Δlog-odds +0.308, t=2.39, positive in BOTH stance buckets — the pre-registered signature of a reasoning shift rather than a passivity prior.

2. **The stance bucketing caught a confound that would otherwise read as a second success.** Marcus's t=2.18 looks comparable to Seneca's, but the buckets split (+0.065 accepting / −0.011 active): the Marcus adapter shifts the model toward calm/accepting options, not toward Stoic choices as such. Reported as a passivity-pattern result, not a decision shift.

3. **Decision movement orders exactly with training corpus size.** Seneca (540 chunks) > Marcus (437) > Epictetus (123, Enchiridion only, no Discourses). Three points with philosopher identity fully confounded with corpus size — recorded as a **hypothesis, not a claim**: an Epictetus adapter retrained on Enchiridion + Discourses should move if data volume is the driver. Notably, Epictetus had the *strongest* judge-scored content effect under CAA (Exp 9, +0.767) and shows *nothing* at the decision level under LoRA — content-in-prose and decision-shift are dissociable.

4. **The Exp 5/6 circuit prediction is confirmed at the behavioral level.** CAA's mid-MLP injection (Exp 5: L9-13 processing, L14-24 compensation) never reaches the choice; LoRA's distributed rewiring with dominant output-layer effects (Exp 6: L27 MLP, "learns→knows" shift L16-25) does. Mechanism → prediction → behavioral confirmation.

#### Cross-Method Synthesis (updates the Exp 10 table)

| Level | Instrument | CAA | LoRA (clean) |
|---|---|---|---|
| Style / register | Judge, Style Auth. (Exp 3b/4) | Robust (+1.0–1.6) | Robust, stronger (+1.6–2.8) |
| Content / prose reasoning | Judge, content score (Exp 8/9) | Weak, philosopher-dependent | (not re-scored on clean adapters) |
| **Decision / choice** | **Judge-free forced choice (Exp 10/11)** | **None at any coefficient** | **Seneca: yes (both buckets). Marcus: passivity only. Epictetus: none.** |

> **Headline:** Same behavioral target, same 40 dilemmas, same 0.542 baseline. Activation steering (CAA) moves register but not choice at any strength; weight-level adaptation (LoRA) can move choice — and where it does, the effect ordering tracks training data volume, and the circuit topology (Exp 5 vs 6) predicted the split.

#### Caveats (do not overclaim)

- **Statistics:** n=40 items, t=2.39 ≈ p≈0.02 uncorrected; three philosophers tested, so Seneca survives multiple-comparisons correction only marginally. Forward passes are deterministic — variance is across items, so firming this up requires **more dilemmas**, not more seeds. Sign test over per-item deltas (in the saved JSON) is the cheap distribution-free companion; run before writing up.
- **Objective confound (structural, carried from Exp 4/6):** CAA uses the contrastive direction (Stoic − neutral); LoRA is continued pretraining on Stoic text only. The CAA-vs-LoRA gap is therefore method + objective jointly. A contrastive-objective LoRA variant is the isolation experiment if a reviewer pushes.
- **Corpus-size hypothesis is confounded:** philosopher identity and corpus size covary perfectly across the three adapters. The Epictetus full-corpus retrain is the discriminating experiment.
- Forced-choice A/B remains a mildly unnatural probe for a base model; the calibrated 0.542 baseline (reproduced across CAA and LoRA runs) is what licenses the deltas.

#### Next Steps

1. **Sign test on Seneca's per-item deltas** (from results JSON) — distribution-free robustness check, zero cost.
2. **Expand the dilemma set** (40 → 80+) to shrink item-level error; this is the only route to tighter CIs on a deterministic probe.
3. **Epictetus full-corpus retrain** (Enchiridion + Discourses) — discriminates the corpus-size hypothesis.
4. **(Reviewer-proofing, lower priority)** Contrastive-objective LoRA variant to isolate method from objective.
5. Fold into the paper: Exp 5/6 circuit graphs → prediction → Exp 10/11 behavioral split as the narrative spine.

---

#### Amendment (July 1, 2026): Sign Tests and Bucket Localization

Distribution-free follow-up on the per-item deltas (overall sign test + per-concept/per-stance breakdown). The two tests disagree in an instructive way, and the claim strength is revised accordingly.

**Overall sign tests (items moved +/− of 40):** Marcus 27/13 (p=0.039, uncorrected — does not survive 3-way correction), Seneca 25/15 (p=0.15, n.s.), Epictetus 17/23 (n.s., tilt negative).

**Marcus — the passivity characterization is now the statistically strongest result in the run.** Accepting-stance items: 18+/4−, p=0.004, mean Δlo +0.331. Active-stance items: 9/9, mean −0.048. The concept buckets that light up (indifference_externals 6+/0−, amor_fati, dichotomy_of_control) are the accepting-dominant concepts — the concept pattern is the stance pattern relabeled. **Revised claim: the Meditations adapter installs a broad acceptance/passivity prior, not Stoic decision-making.** Robust, precisely characterized, and a cautionary result about what "philosophical" fine-tuning actually installs.

**Seneca — the effect is heavy-tailed, NOT concept-localized.** Seven of eight concept means are positive with no dominant bucket; magnitudes are large where counts are mixed (amor_fati: 3+/2− yet mean +0.748). Active items split 9/9 by sign but retain a positive mean (+0.200) — the ups are bigger than the downs. The t-test (2.39) saw the magnitudes; the sign test (25/40) saw the weak direction bias; both are right. **Revised claim: the Letters adapter produces large decision shifts on a minority of items scattered across concepts, rather than a uniform nudge.** Whether the tail is systematic (shared item features) or fragile (idiosyncratic) is not determinable at n=40.

**Epictetus — null at every resolution** (overall, per-concept, per-stance). Full-corpus retrain hypothesis unchanged.

**Revised headline (supersedes the Exp 11 headline above):** The CAA/LoRA asymmetry stands — CAA moved nothing on any test at any coefficient; LoRA produces real, structured decision-level effects. But no LoRA adapter clears both parametric and distribution-free significance for a *Stoic reasoning* shift: Marcus's effect is a passivity prior (stance-dissociated), Seneca's is heavy-tailed magnitude without breadth, Epictetus is null. The defensible sentence: **weight-level adaptation reaches the decision layer in ways activation steering does not — but what it installs is not yet uniform Stoic decision-making.**

**Revised next steps (supersede items 1–2 above):**
1. **Qualitative read of Seneca's top ~10 movers** (largest |Δlo| in the JSON, zero cost): heavy-tailed effects usually share a feature the concept labels miss (wording, proximity to the Letters' actual subject matter — wealth, status, grief). Shared feature → v3 design input; no pattern → v3 becomes a straight replication test of a fragile tail.
2. **Expand the dilemma set uniformly** (not concept-oversampled — there are no hot concepts) to 80+; at n=40 "concentrated real effect" vs "lucky subset" is not distinguishable.
3. Multiple-comparisons discipline going forward: all bucket p-values here are post-hoc on 8-way slices of n=40 and are treated as hypothesis-generating only.

---

#### Amendment 2 (July 1, 2026): Qualitative Read of Seneca's Top Movers — Two Rival Mechanisms

Amendment 1's next-step 1 executed: the 10 largest |Δlog-odds| items for Seneca were read against their dilemma texts. The tail is **not random** — 8 of 10 large movers are positive and cluster on plausible features — but two rival mechanisms fit it:

**Hypothesis A — topic proximity.** The big movers sit on the *Moral Letters'* home turf: aging/bodily decline (fate_05, +2.24 — Seneca's letters on old age), loss of a valued possession (ext_02, +1.38 — the grief consolations), declining indulgence (self_01, +1.55 — letters on luxury), status envy at a peer's promotion (ext_04, +1.06 — his ambition theme), fear before a committed task (emot_03, +1.30). Prediction: the adapter moves decisions most on subject matter dense in its training text.

**Hypothesis B — lexical echo (the subtler style confound).** Some top movers' *Stoic option wording* contains Seneca's own signature idiom: ext_02's option says the possession "was only ever **on loan**" — near-verbatim Senecan metaphor (fortune lends and reclaims); fate_02's "play the part well" is the classical role metaphor. If the adapter raises the probability of Seneca-sounding language, the choice moves *wherever the option happens to sound Senecan* — a register effect reaching the decision through the option's phrasing, i.e. the style confound relocated inside the forced-choice instrument itself. The option texts were not controlled for idiom.

Two negative movers noted: ctrl_03 (grave prognosis, −1.29) and self_04 (unearned praise, −1.08); ctrl_03 is *also* core Letters territory (illness), which cuts slightly against a pure topic story.

**Discriminating experiments (pre-registered before further looks at the data):**
1. **Cheap first check:** judge-score each of the 40 v2 Stoic options for Stoic/Senecan idiom (1–5) and correlate with Seneca's per-item Δlog-odds. Strong correlation → Hypothesis B live, phrasing axis mandatory in v3. Cost: ~40 judge calls.
2. **v3 dilemma set as a 2×2:** (Letters-core topic vs off-topic) × (plain-modern vs Stoic-idiom option phrasing), same situation and decision within phrasing pairs. Topic proximity predicts movement tracks the topic axis; lexical echo predicts it tracks the phrasing axis. Plain-worded, off-topic Stoic options still moving would be the strongest available reasoning claim.

**Status of the decision-level claim pending these:** "LoRA reaches the decision layer" stands (the CAA/LoRA asymmetry is unaffected); "what it installs" remains open between *topic-conditional decision shift* and *idiom-mediated register effect*.

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
| Dilemma evals — Exp 10 (CAA v1, v2, coeff sweep) | $0 (local) | — |
| LoRA clean retrain (Colab T4) + dilemma eval — Exp 11 | $0 (free T4 + local) | — |

**Notes:**
- June 8 costs are estimates — reconcile against the actual Anthropic console billing and update.
- Bridge analyses (ModelLens) cost nothing — all local compute.
- `vary="judge"` seed eval re-runs only the judge, not generation (generation runs once per layer), so its cost scales with judge calls, not generations.