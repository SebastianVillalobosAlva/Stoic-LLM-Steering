# Stoic LLM Steering

Steering language models to write like ancient Stoic philosophers using **Contrastive Activation Addition (CAA)** and **LoRA fine-tuning**, with a systematic evaluation framework.

## Overview

This project explores two techniques for steering Llama-3.2-1B to generate text in the style of Marcus Aurelius, Seneca, and Epictetus:

1. **Contrastive Activation Addition (CAA)** — extracts activation differences between Stoic and neutral text to create steering vectors applied at inference time with zero training cost
2. **LoRA Fine-tuning** — parameter-efficient fine-tuning directly on classical Stoic texts

The project includes an automated data pipeline, an LLM-as-judge evaluation framework, and systematic hyperparameter sweeps across layers and coefficients.

## Quickstart

```bash
git clone https://github.com/SebastianVillalobosAlva/Stoic-LLM-Steering.git
cd Stoic-LLM-Steering
pip install -e .

export ANTHROPIC_API_KEY=your_key_here

# Run steering with optimal configuration
python scripts/run_steering.py

# Run evaluation sweep
python scripts/run_eval.py
```

## Project Structure

```
stoic-llm/
├── stoic_llm/
│   ├── data/                # Data pipeline (download, process, pair generation)
│   ├── steering/            # CAA steering (vector extraction + inference)
│   ├── lora/                # LoRA fine-tuning (data prep, training, inference)
│   ├── eval/                # Evaluation framework (judge, sweep, comparison)
│   ├── model.py             # Model loader
│   └── config.py            # Consolidated configuration
├── data/                    # Data files (texts, pairs, steering vectors)
├── models/                  # Trained LoRA adapters
├── results/                 # Evaluation results (sweeps, comparisons)
├── scripts/                 # Entry point scripts
└── app/                     # Streamlit demo
```

## Technical Approach

### Data Pipeline

Stoic texts are scraped from Project Gutenberg (Marcus Aurelius's Meditations, Seneca's Moral Letters, Epictetus's Enchiridion), cleaned, and chunked into paragraphs. Claude API generates neutral paraphrases of each chunk, creating 30 contrastive pairs per philosopher that preserve semantic content while removing stylistic markers.

### Contrastive Activation Addition (CAA)

CAA extracts a steering vector by computing the mean activation difference between Stoic and neutral texts across all contrastive pairs. At inference time, this vector is added to a specific layer's MLP output, nudging the model's generation toward Stoic style with zero additional training.

```python
# Extract steering vector
steering_vector = mean(stoic_activations - neutral_activations)

# Apply at inference
mlp_output = mlp_output + coefficient * steering_vector
```

### LoRA Fine-tuning

LoRA adapts only 0.07% of model parameters (851k out of 1.2B) by adding low-rank decomposition matrices to the query and value projection layers. Each philosopher gets a separate adapter trained on their texts for 3 epochs.

## Evaluation Results

### Methodology

We use an **LLM-as-judge** framework to systematically evaluate steering effectiveness. Claude scores each model output on four dimensions (1-5 scale):

- **Philosophical Depth** — engagement with Stoic concepts
- **Stoic Alignment** — adherence to core Stoic doctrines (dichotomy of control, virtue as sole good, living according to nature)
- **Coherence** — clarity and logical flow
- **Stylistic Authenticity** — resemblance to translated ancient philosophical text

### Hyperparameter Sweep

We ran a two-stage sweep for each philosopher: first testing layers 4-14 at a fixed coefficient, then sweeping coefficients at the best layer. All evaluations compare steered vs unsteered Llama-3.2-1B outputs on the same prompts.

| Philosopher | Best Layer | Best Coefficient | Aggregate Score |
|---|---|---|---|
| Marcus Aurelius | 10 | 0.11 | 2.08 |
| Epictetus | 12 | 0.05 | 2.33 |
| Seneca | 14 | 0.30 | 2.08 |

### Key Findings

**Each philosopher requires a different steering configuration.** The optimal layer and coefficient vary across philosophers, reflecting differences in their writing styles and how the model represents them internally:

- **Epictetus** (direct, aphoristic) — responds to light steering (coefficient 0.05) at a mid-depth layer (12). Achieved the highest aggregate score (2.33), likely because the Enchiridion's concise style is easier for a 1B model to approximate.
- **Marcus Aurelius** (reflective, contemplative) — works best with moderate steering (0.11) at an earlier layer (10), consistent with the more distributed nature of the Meditations' reflective style.
- **Seneca** (rhetorical, epistolary) — requires the strongest steering (0.30) at the deepest layer (14), suggesting his complex rhetorical style is harder to elicit and needs a more aggressive intervention.

**Steering improves Stoic alignment but with modest gains on a 1B model.** The delta between steered and unsteered outputs is positive but small (+0.17 for Marcus Aurelius), reflecting the limited representational capacity of Llama-3.2-1B. We expect larger gains with more contrastive pairs (currently 30 per philosopher) and larger models.

**Excessive steering degrades coherence.** Across all philosophers, high coefficients (0.20+) generally reduce output quality, confirming the expected tradeoff between steering strength and text coherence. Seneca is the exception, likely due to noise from limited evaluation prompts.

### Coefficient-Quality Tradeoff (Marcus Aurelius, Layer 10)

```
Coefficient  |  Aggregate Score
-------------|------------------
    0.030    |      1.67
    0.050    |      1.75
    0.080    |      1.67
    0.110    |      2.08  ← optimal
    0.150    |      1.17
    0.200    |      1.25
    0.300    |      1.00
```

Full sweep results are available in `results/sweeps/`.

## CAA vs LoRA

| Aspect | CAA | LoRA |
|---|---|---|
| **Training Cost** | None (zero-shot) | ~15 min per philosopher |
| **Inference Control** | Adjustable layer + coefficient | Fixed after training |
| **Stability** | Moderate (prompt-sensitive) | High |
| **Style Transfer** | Weak-moderate | Strong |
| **Best Use Case** | Research + interpretability | Production deployment |

CAA's advantage is that it requires no training and is adjustable at inference — you can dial the coefficient up or down in real time. LoRA produces more stable, higher-quality outputs but is fixed once trained. For interpretability research, CAA is more valuable because the steering vector itself can be analyzed mechanistically.

## Future Work

- [ ] Test on larger models (Llama-3.2-7B, Mistral-7B)
- [ ] Expand training data (100+ pairs per philosopher)
- [ ] CAA vs LoRA comparison with matched evaluation metrics
- [ ] Bridge analysis with [ModelLens](https://github.com/SebastianVillalobosAlva/ModelLens) — mechanistic explanation of why steering works at specific layers
- [ ] Deploy as public web app
- [x] Create evaluation metrics for "Stoic-ness"
- [x] Systematic hyperparameter sweep

## Technologies

- **PyTorch** — deep learning framework
- **Transformers** — model architecture and training
- **PEFT** — LoRA implementation
- **Anthropic Claude API** — neutral text generation and LLM-as-judge evaluation
- **Streamlit** — interactive demo

## Author

**Sebastian Villalobos**
- GitHub: [@SebastianVillalobosAlva](https://github.com/SebastianVillalobosAlva)
- LinkedIn: [Sebastian Villalobos Alva](https://www.linkedin.com/in/sebastian-villalobos-alva/)