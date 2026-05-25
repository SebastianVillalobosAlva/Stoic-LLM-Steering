"""
scripts/bridge_analysis.py — ModelLens × Stoic LLM Bridge Analysis

Runs mechanistic interpretability on steered vs unsteered 3B model:
  1. Layer evolution comparison (logit lens) — what does the model
     "believe" at each layer, steered vs unsteered?
  2. Residual stream analysis — how does steering change information flow?
  3. Activation patching — which components are causally responsible?

Usage:
  python scripts/bridge_analysis.py
"""

import json
import torch
from datetime import datetime
from modellens import ModelLens
from stoic_llm.model import ModelLoader
from stoic_llm.config import VECTORS_DIR, RESULTS_DIR

# ── Setup ────────────────────────────────────────────────────────────
BRIDGE_DIR = RESULTS_DIR / "bridge"
BRIDGE_DIR.mkdir(parents=True, exist_ok=True)

model_size = "3B"
loader = ModelLoader(model_size)
model, tokenizer = loader.load()

lens = ModelLens(model)
lens.adapter.set_tokenizer(tokenizer)

print(
    f"ModelLens loaded: {lens.adapter.type_of_adapter}, {lens.adapter.architecture_family}"
)
print(f"Available analyses: {lens.available_analyses()}")

# ── Configs ──────────────────────────────────────────────────────────
configs = {
    "marcus_aurelius": {"layer": 8, "coefficient": 0.08},
    "epictetus": {"layer": 8, "coefficient": 0.15},
    "seneca": {"layer": 8, "coefficient": 0.11},
}

test_prompts = [
    "When facing difficulty, one should",
    "The wise person is one who",
    "True freedom comes from",
]


# ── Steering Hook ────────────────────────────────────────────────────
def steer_and_tokenize(prompt, author, cfg):
    """Load steering vector and return hooked generate inputs."""
    vector_path = VECTORS_DIR / f"{author}_steering_{model_size}.pt"
    vector = torch.load(vector_path, map_location="cpu", weights_only=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs, vector, cfg["layer"], cfg["coefficient"]


# ── Analysis 1: Layer Evolution Comparison ───────────────────────────
def run_layer_evolution_bridge(author, cfg):
    """Compare what the model believes at each layer, steered vs unsteered."""
    from modellens.analysis.layer_evolution import (
        run_layer_evolution,
        run_layer_evolution_comparison,
        summarize_comparison,
    )

    print(f"\n{'='*60}")
    print(f"LAYER EVOLUTION — {author}")
    print(f"{'='*60}")

    results = {}
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")

        # Unsteered run
        lens.clear()
        unsteered = run_layer_evolution(
            lens,
            inputs,
            tokenizer=tokenizer,
            top_k=10,
            capture_full_logits=True,
        )

        # Steered run — add hook
        vector_path = VECTORS_DIR / f"{author}_steering_{model_size}.pt"
        vector = torch.load(vector_path, map_location="cpu", weights_only=True)

        hook = model.model.layers[cfg["layer"]].mlp.register_forward_hook(
            lambda mod, inp, out, v=vector, c=cfg["coefficient"]: out + c * v
        )

        lens.clear()
        steered = run_layer_evolution(
            lens,
            inputs,
            tokenizer=tokenizer,
            top_k=10,
            capture_full_logits=True,
        )
        hook.remove()

        # Compare
        print(f"\nPrompt: '{prompt}'")

        if unsteered["layers"] and steered["layers"]:
            # Show top prediction at each layer for both
            print(f"  {'Layer':<35s} {'Unsteered Top-1':<20s} {'Steered Top-1':<20s}")
            print(f"  {'-'*35} {'-'*20} {'-'*20}")

            u_layers = {l["layer_name"]: l for l in unsteered["layers"]}
            s_layers = {l["layer_name"]: l for l in steered["layers"]}

            for name in unsteered["layers_ordered"]:
                u = u_layers.get(name, {})
                s = s_layers.get(name, {})

                u_tok = u.get("top_k_tokens", ["?"])[0] if u else "?"
                s_tok = s.get("top_k_tokens", ["?"])[0] if s else "?"
                u_prob = u.get("top1_prob", 0) if u else 0
                s_prob = s.get("top1_prob", 0) if s else 0

                marker = (
                    " ← STEERING LAYER"
                    if name == f"model.layers.{cfg['layer']}"
                    else ""
                )
                print(
                    f"  {name:<35s} {u_tok:<12s} ({u_prob:.3f}) {s_tok:<12s} ({s_prob:.3f}){marker}"
                )

        results[prompt] = {
            "unsteered_entropy": unsteered.get("entropy_trajectory", []),
            "steered_entropy": steered.get("entropy_trajectory", []),
            "unsteered_top1": [
                l.get("top_k_tokens", ["?"])[0] for l in unsteered.get("layers", [])
            ],
            "steered_top1": [
                l.get("top_k_tokens", ["?"])[0] for l in steered.get("layers", [])
            ],
        }

    return results


# ── Analysis 2: Residual Stream ──────────────────────────────────────
def run_residual_bridge(author, cfg):
    """Compare residual stream contributions, steered vs unsteered."""
    from modellens.analysis.residual_stream import run_residual_analysis

    print(f"\n{'='*60}")
    print(f"RESIDUAL STREAM — {author}")
    print(f"{'='*60}")

    prompt = test_prompts[0]
    inputs = tokenizer(prompt, return_tensors="pt")

    # Unsteered
    lens.clear()
    unsteered = run_residual_analysis(lens, inputs)

    # Steered
    vector_path = VECTORS_DIR / f"{author}_steering_{model_size}.pt"
    vector = torch.load(vector_path, map_location="cpu", weights_only=True)

    hook = model.model.layers[cfg["layer"]].mlp.register_forward_hook(
        lambda mod, inp, out, v=vector, c=cfg["coefficient"]: out + c * v
    )

    lens.clear()
    steered = run_residual_analysis(lens, inputs)
    hook.remove()

    # Compare contributions
    print(f"\nPrompt: '{prompt}'")
    print(f"  {'Layer':<35s} {'Unsteered δ':>12s} {'Steered δ':>12s} {'Diff':>10s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")

    for name in unsteered["contributions"]:
        u = unsteered["contributions"][name]["delta_norm"]
        s = steered["contributions"].get(name, {}).get("delta_norm", 0)
        diff = s - u
        marker = " ← STEERING" if name == f"model.layers.{cfg['layer']}" else ""
        print(f"  {name:<35s} {u:>12.4f} {s:>12.4f} {diff:>+10.4f}{marker}")

    return {
        "unsteered": {
            k: v["delta_norm"] for k, v in unsteered["contributions"].items()
        },
        "steered": {k: v["delta_norm"] for k, v in steered["contributions"].items()},
    }


# ── Analysis 3: Activation Patching ─────────────────────────────────
def run_patching_bridge(author, cfg):
    """Find which components are causally responsible for steered behavior.

    Strategy: capture steered activations first with the hook active,
    then run patching WITHOUT the hook, injecting steered activations
    one layer at a time to see which component reproduces the steered output.
    """
    print(f"\n{'='*60}")
    print(f"ACTIVATION PATCHING — {author}")
    print(f"{'='*60}")

    prompt = test_prompts[0]
    inputs = tokenizer(prompt, return_tensors="pt")

    vector_path = VECTORS_DIR / f"{author}_steering_{model_size}.pt"
    vector = torch.load(vector_path, map_location="cpu", weights_only=True)

    # Step 1: Get unsteered (clean) output metric
    with torch.no_grad():
        clean_out = model(**inputs)
    clean_logits = clean_out.logits[:, -1, :]
    clean_metric = clean_logits.max(dim=-1).values.mean().item()

    # Step 2: Get steered output metric + capture all activations
    patchable = lens.adapter.get_patchable_layers()
    available = dict(model.named_modules())
    steered_activations = {}

    hooks = []
    for name in patchable:

        def make_capture(n):
            def hook_fn(mod, inp, out):
                if isinstance(out, tuple):
                    steered_activations[n] = tuple(
                        o.detach().clone() if o is not None else None for o in out
                    )
                else:
                    steered_activations[n] = out.detach().clone()

            return hook_fn

        h = available[name].register_forward_hook(make_capture(name))
        hooks.append(h)

    # Add steering hook
    steering_hook = model.model.layers[cfg["layer"]].mlp.register_forward_hook(
        lambda mod, inp, out, v=vector, c=cfg["coefficient"]: out + c * v
    )

    with torch.no_grad():
        steered_out = model(**inputs)
    steered_logits = steered_out.logits[:, -1, :]
    steered_metric = steered_logits.max(dim=-1).values.mean().item()

    # Remove all hooks
    for h in hooks:
        h.remove()
    steering_hook.remove()

    total_effect = steered_metric - clean_metric

    # Step 3: Patch one layer at a time — inject steered activation
    # into an otherwise unsteered forward pass
    patch_effects = {}
    for target in patchable:
        if target not in steered_activations:
            continue

        def make_patch(act):
            def hook_fn(mod, inp, out):
                return act

            return hook_fn

        h = available[target].register_forward_hook(
            make_patch(steered_activations[target])
        )
        with torch.no_grad():
            patched_out = model(**inputs)
        h.remove()

        patched_logits = patched_out.logits[:, -1, :]
        patched_metric = patched_logits.max(dim=-1).values.mean().item()
        effect = patched_metric - clean_metric

        patch_effects[target] = {
            "patched_metric": patched_metric,
            "effect": effect,
            "normalized_effect": effect / (total_effect + 1e-10),
        }

    patch_results = {
        "clean_metric": clean_metric,
        "corrupted_metric": steered_metric,
        "total_effect": total_effect,
        "patch_effects": patch_effects,
    }

    # Show top causal components
    effects = patch_results["patch_effects"]
    sorted_effects = sorted(
        effects.items(),
        key=lambda x: abs(x[1]["normalized_effect"]),
        reverse=True,
    )

    print(f"\nPrompt: '{prompt}'")
    print(f"Total effect: {patch_results['total_effect']:.4f}")
    print(f"\nTop 10 causally important components:")
    print(f"  {'Component':<40s} {'Effect':>10s} {'Normalized':>12s}")
    print(f"  {'-'*40} {'-'*10} {'-'*12}")

    for name, data in sorted_effects[:10]:
        marker = " ← STEERING" if f"layers.{cfg['layer']}." in name else ""
        print(
            f"  {name:<40s} {data['effect']:>10.4f} {data['normalized_effect']:>12.4f}{marker}"
        )

    return {
        "total_effect": patch_results["total_effect"],
        "top_components": [
            {"name": n, "effect": d["effect"], "normalized": d["normalized_effect"]}
            for n, d in sorted_effects[:20]
        ],
    }


# ── Run Everything ───────────────────────────────────────────────────
all_results = {}

for author, cfg in configs.items():
    print(f"\n\n{'#'*60}")
    print(f"# BRIDGE ANALYSIS: {author.upper()}")
    print(f"{'#'*60}")

    author_results = {}
    author_results["layer_evolution"] = run_layer_evolution_bridge(author, cfg)
    author_results["residual_stream"] = run_residual_bridge(author, cfg)
    author_results["activation_patching"] = run_patching_bridge(author, cfg)
    all_results[author] = author_results

# Save results
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = BRIDGE_DIR / f"bridge_{model_size}_{ts}.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ Bridge results saved to {out}")
