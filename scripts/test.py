"""
Sanity-check the Stoic metrics before using them in discover_circuit.
Not a unit test — a "look at the numbers and confirm they make sense" script.

Run:  python scripts/test_metrics.py
"""

import torch
from stoic_llm.model import ModelLoader
from stoic_llm.config import VECTORS_DIR
from stoic_llm.eval.metrics import (
    make_stoic_token_metric,
    make_steering_projection_metric,
)

# --- edit these after the re-sweep ---
MODEL_SIZE = "3B"
AUTHOR = "marcus_aurelius"
LAYER = 8
COEFF = 0.11
PROMPT = "When facing difficulty, one should"

STOIC_WORDS = ["virtue", "reason", "control", "accept", "nature"]
NEUTRAL_WORDS = ["money", "win", "fight", "comfort", "success"]

loader = ModelLoader(MODEL_SIZE)
model, tokenizer = loader.load()


# ── Check 1: does each word map to a clean token? ─────────────────────
# LLaMA BPE splits words and treats leading spaces as separate tokens, so
# `encode(w)[0]` can silently grab a fragment. Verify the first id decodes
# back to (a recognizable piece of) the word.
def show(label, words):
    print(f"\n{label}:")
    for w in words:
        for variant in (w, " " + w):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            print(
                f"  {variant!r:12} -> ids={ids}  "
                f"first={ids[0]} decodes {tokenizer.decode([ids[0]])!r}"
            )


show("STOIC", STOIC_WORDS)
show("NEUTRAL", NEUTRAL_WORDS)
# If the no-space variant fragments but the leading-space one is clean,
# prepend a space to your word lists (or fix first_id in metrics.py).

# ── Set up steered vs unsteered forward passes ────────────────────────
loaded = torch.load(
    VECTORS_DIR / f"{AUTHOR}_steering_{MODEL_SIZE}.pt", weights_only=True
)
inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
if not isinstance(loaded, dict):
    raise ValueError("Old single-tensor format — run extract_vectors.py 3B first")
vec = loaded[LAYER]
assert vec.ndim == 1, f"expected (hidden,), got {tuple(vec.shape)}"


def run(steered, **kw):
    handle = None
    if steered:

        def hook(m, i, o):
            return o + COEFF * vec

        handle = model.model.layers[LAYER].mlp.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(**inputs, **kw)
    finally:
        if handle:
            handle.remove()
    return out


# ── Check 2: token metric — steering should RAISE the Stoic−neutral gap ─
token_metric = make_stoic_token_metric(tokenizer, STOIC_WORDS, NEUTRAL_WORDS)
base_t = token_metric(run(False))
steer_t = token_metric(run(True))
print(
    f"\nToken metric:  base={base_t:+.4f}  steered={steer_t:+.4f}  Δ={steer_t - base_t:+.4f}"
)
print("  expect Δ > 0 (steering pushes toward Stoic vocabulary)")

# ── Check 3: projection metric — find the right hidden_states index ────
# hidden_states[0] = embeddings; hidden_states[i] = output of block (i-1).
# Injection is at block LAYER's MLP, so the jump appears at hidden_states[LAYER+1],
# NOT [LAYER]. Print all indices so you can SEE where steering bites and set
# the index in make_steering_projection_metric accordingly.
base_hs = run(False, output_hidden_states=True).hidden_states
steer_hs = run(True, output_hidden_states=True).hidden_states

print(type(vec), getattr(vec, "shape", None))

v = vec / (vec.norm() + 1e-10)
print("\nProjection onto steering direction, per hidden_states index:")
print(f"  {'idx':>3} {'base':>9} {'steered':>9} {'Δ':>9}")
for i in range(len(base_hs)):
    b = (base_hs[i][:, -1, :] @ v.to(base_hs[i].dtype)).mean().item()
    s = (steer_hs[i][:, -1, :] @ v.to(steer_hs[i].dtype)).mean().item()
    mark = "  <-- injection block output" if i == LAYER + 1 else ""
    print(f"  {i:>3} {b:>9.3f} {s:>9.3f} {s - b:>+9.3f}{mark}")
print("  expect a clear positive Δ starting around idx LAYER+1;")
print("  use that index in make_steering_projection_metric.")
