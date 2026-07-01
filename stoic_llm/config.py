from pathlib import Path
from typing import Any

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# Generation Config
GEN_KWARGS: dict[str, Any] = dict(
    max_new_tokens=100,
    do_sample=False,
    repetition_penalty=1.3,
    no_repeat_ngram_size=3,
)

# Model Config
DEVICE = "cpu"

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = DATA_DIR / "config"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKED_DIR = DATA_DIR / "chunked"
VECTORS_DIR = DATA_DIR / "steering_vectors"
LORA_TRAINING_DIR = DATA_DIR / "lora_training"

# Model Paths
MODELS_DIR = PROJECT_ROOT / "models"

# Results Paths
RESULTS_DIR = PROJECT_ROOT / "results"
SWEEPS_DIR = RESULTS_DIR / "sweeps"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"
JUDGES_DIR = RESULTS_DIR / "judges"

# Sources Config
SOURCES_CONFIG = CONFIG_DIR / "sources.json"

# Steering Defaults
LAYER_IDX = 12
COEFFICIENT = 0.11
TEMPERATURE = 0.7
MAX_TOKENS = 100
DEFAULT_PROMPTS = [
    # Original 3
    "When facing difficulty, one should",
    "The nature of virtue is",
    "To live well means",
    # Dichotomy of control
    "What is not in our power",
    "We suffer not from events but from",
    # Virtue ethics
    "The wise person is one who",
    "True freedom comes from",
    # Adversity / amor fati
    "The obstacle in the path becomes",
    "Suffering teaches us that",
    # Death and impermanence
    "We should remember that all things",
    "Each day should be lived as if",
    # Duty and cosmopolitanism
    "Our duty to others requires",
]

# Create Directories
for d in [
    RAW_DIR,
    PROCESSED_DIR,
    CONFIG_DIR,
    CHUNKED_DIR,
    VECTORS_DIR,
    LORA_TRAINING_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SWEEPS_DIR,
    COMPARISONS_DIR,
    JUDGES_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

NEUTRAL_PAIR_PROMPT = """Below is a philosophical passage from {author_name}. Your job is to give advice about the SAME situation, but reasoning from a worldview that genuinely DISAGREES with Stoicism — not Stoicism in plainer words.

Pick a competing framework and argue from it, e.g.:
- Ambition/achievement: pursue status, wealth, and winning; external success IS what matters
- Hedonism: maximize pleasure and comfort; avoid discomfort rather than accept it
- Assertiveness/self-advocacy: change your circumstances, push back, demand more
- Emotional expression: feel and express anger/desire fully rather than governing them

Hard requirements:
- Reach a recommendation a Stoic would REJECT. The conclusion itself must differ, not just the wording.
- FORBIDDEN (these are Stoic ideas — do not endorse any of them, even casually): accepting what you can't control, focusing on what's "up to you", indifference to externals (reputation, money, body, outcomes), virtue/character as the main good, "this won't matter in the long run", inner tranquility over external change, others' opinions don't matter.
- Do NOT use a calm, detached, or "wise" self-help tone. Write as someone who actively wants the external thing — the promotion, the win, the pleasure, the apology owed to them.

FAILURE CONDITION: If your rewrite could be summarized the same way as the original passage, you have failed. The original and your rewrite must give OPPOSITE life advice, not the same advice in different words.
- Output ONLY the advice itself. No headers, no preamble, no labeling which framework you are using, no meta-commentary. Start directly with the advice and write it as continuous prose.

Passage:
{stoic_text}"""


# NEUTRAL_PAIR_PROMPT = """Below is a philosophical passage from {author_name}. Rewrite it as modern conventional advice about the SAME topic, but using mainstream reasoning — not Stoic philosophy.

# Rules:
# - Same topic, DIFFERENT reasoning (e.g. if the passage says "ignore insults because others' opinions aren't in your control", the rewrite might say "stand up for yourself and set boundaries")
# - No Stoic concepts: no dichotomy of control, no "according to nature", no indifference to externals, no virtue as sole good
# - Use casual modern tone, like advice from a friend or self-help blog
# - Similar length to the original

# Passage:
# {stoic_text}"""
