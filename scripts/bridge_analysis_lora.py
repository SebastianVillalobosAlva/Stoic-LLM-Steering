import json
import torch
from datetime import datetime
from copy import deepcopy
from peft import PeftModel
from modellens import ModelLens
from modellens.analysis.layer_evolution import run_layer_evolution
from modellens.analysis.residual_stream import run_residual_analysis
from stoic_llm.model import ModelLoader
from stoic_llm.config import MODELS_DIR, RESULTS_DIR

# ── Setup ────────────────────────────────────────────────────────────
BRIDGE_DIR = RESULTS_DIR / "bridge"
BRIDGE_DIR.mkdir(parents=True, exist_ok=True)

model_size = "3B"

test_prompts = [
    "When facing difficulty, one should",
    "The wise person is one who",
    "True freedom comes from",
]

authors = ["marcus_aurelius", "epictetus", "seneca"]


# ── Analysis Functions ───────────────────────────────────────────────
def run_single_analysis(model, tokenizer, label, layer_names=None):
    """Run layer evolution and residual stream on a single model."""
    lens = ModelLens(model)
    lens.adapter.set_tokenizer(tokenizer)

    if layer_names is None:
        layer_names = lens.adapter.get_sequential_layers()

    evolution_results = {}
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        lens.clear()
        result = run_layer_evolution(
            lens,
            inputs,
            tokenizer=tokenizer,
            top_k=10,
            capture_full_logits=True,
            layer_names=layer_names,
        )
        evolution_results[prompt] = {
            "top1": [l.get("top_k_tokens", ["?"])[0] for l in result.get("layers", [])],
            "top1_probs": [l.get("top1_prob", 0) for l in result.get("layers", [])],
            "layers_ordered": result.get("layers_ordered", []),
        }

    # Residual stream on first prompt
    inputs = tokenizer(test_prompts[0], return_tensors="pt")
    lens.clear()
    residual = run_residual_analysis(lens, inputs, layer_names=layer_names)

    lens.clear()
    return evolution_results, residual


def run_patching_analysis(base_model, lora_model, tokenizer):
    """Capture LoRA activations, inject into base model one at a time."""
    lens = ModelLens(base_model)
    lens.adapter.set_tokenizer(tokenizer)

    prompt = test_prompts[0]
    inputs = tokenizer(prompt, return_tensors="pt")
    patchable = lens.adapter.get_patchable_layers()
    available = dict(base_model.named_modules())

    # Step 1: Base metric
    with torch.no_grad():
        base_out = base_model(**inputs)
    base_metric = base_out.logits[:, -1, :].max(dim=-1).values.mean().item()

    # Step 2: LoRA metric + capture activations
    lora_activations = {}
    hooks = []
    lora_available = dict(lora_model.named_modules())

    for name in patchable:
        lora_name = name  # After merge_and_unload, names are same as base
        if lora_name not in lora_available:
            continue

        def make_capture(base_n):
            def hook_fn(mod, inp, out):
                if isinstance(out, tuple):
                    lora_activations[base_n] = tuple(
                        o.detach().clone() if o is not None else None for o in out
                    )
                else:
                    lora_activations[base_n] = out.detach().clone()

            return hook_fn

        h = lora_available[lora_name].register_forward_hook(make_capture(name))
        hooks.append(h)

    with torch.no_grad():
        lora_out = lora_model(**inputs)
    lora_metric = lora_out.logits[:, -1, :].max(dim=-1).values.mean().item()

    for h in hooks:
        h.remove()

    total_effect = lora_metric - base_metric

    # Step 3: Patch one at a time into base model
    patch_effects = {}
    for target in patchable:
        if target not in lora_activations or target not in available:
            continue

        def make_patch(act):
            def hook_fn(mod, inp, out):
                return act

            return hook_fn

        h = available[target].register_forward_hook(
            make_patch(lora_activations[target])
        )
        with torch.no_grad():
            patched_out = base_model(**inputs)
        h.remove()

        patched_metric = patched_out.logits[:, -1, :].max(dim=-1).values.mean().item()
        effect = patched_metric - base_metric

        patch_effects[target] = {
            "patched_metric": patched_metric,
            "effect": effect,
            "normalized_effect": effect / (total_effect + 1e-10),
        }

    lens.clear()
    return {
        "base_metric": base_metric,
        "lora_metric": lora_metric,
        "total_effect": total_effect,
        "patch_effects": patch_effects,
    }


def print_evolution_comparison(base_evo, lora_evo, prompt):
    """Print side-by-side layer evolution."""
    print(f"\nPrompt: '{prompt}'")
    b = base_evo[prompt]
    l = lora_evo[prompt]

    layers = b["layers_ordered"]
    l_layers = l["layers_ordered"]

    print(f"  {'Layer':<35s} {'Base Top-1':<20s} {'LoRA Top-1':<20s}")
    print(f"  {'-'*35} {'-'*20} {'-'*20}")

    for i, name in enumerate(layers):
        b_tok = b["top1"][i] if i < len(b["top1"]) else "?"
        l_tok = l["top1"][i] if i < len(l["top1"]) else "?"
        b_prob = b["top1_probs"][i] if i < len(b["top1_probs"]) else 0
        l_prob = l["top1_probs"][i] if i < len(l["top1_probs"]) else 0

        diff = " ← CHANGED" if b_tok.strip() != l_tok.strip() else ""
        print(
            f"  {name:<35s} {b_tok:<12s} ({b_prob:.3f}) {l_tok:<12s} ({l_prob:.3f}){diff}"
        )


def print_residual_comparison(base_res, lora_res):
    """Print residual stream comparison."""
    print(f"\nPrompt: '{test_prompts[0]}'")
    print(f"  {'Layer':<35s} {'Base δ':>12s} {'LoRA δ':>12s} {'Diff':>10s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")

    for name in base_res["contributions"]:
        b = base_res["contributions"][name]["delta_norm"]
        l = lora_res["contributions"].get(name, {}).get("delta_norm", 0)
        diff = l - b
        marker = " ← BIG CHANGE" if abs(diff) > 0.5 else ""
        print(f"  {name:<35s} {b:>12.4f} {l:>12.4f} {diff:>+10.4f}{marker}")


def print_patching(patch_results):
    """Print top patching effects."""
    effects = patch_results["patch_effects"]
    sorted_effects = sorted(
        effects.items(),
        key=lambda x: abs(x[1]["normalized_effect"]),
        reverse=True,
    )

    print(f"\nTotal effect: {patch_results['total_effect']:.4f}")
    print(f"\nTop 10 causally important components:")
    print(f"  {'Component':<40s} {'Effect':>10s} {'Normalized':>12s}")
    print(f"  {'-'*40} {'-'*10} {'-'*12}")

    for name, data in sorted_effects[:10]:
        print(
            f"  {name:<40s} {data['effect']:>10.4f} {data['normalized_effect']:>12.4f}"
        )


# ── Run Everything ───────────────────────────────────────────────────
all_results = {}

# Step 1: Run base model analysis ONCE
print(f"\n{'#'*60}")
print("# BASE MODEL ANALYSIS")
print(f"{'#'*60}")

loader = ModelLoader(model_size)
base_model, tokenizer = loader.load()

base_evolution, base_residual = run_single_analysis(base_model, tokenizer, "base")

# Save base layer names for LoRA runs
base_lens = ModelLens(base_model)
base_layer_names = base_lens.adapter.get_sequential_layers()
base_lens.clear()

print("✓ Base model analysis complete")

# Step 2: For each philosopher, load LoRA and compare
for author in authors:
    print(f"\n\n{'#'*60}")
    print(f"# BRIDGE ANALYSIS (LoRA): {author.upper()}")
    print(f"{'#'*60}")

    lora_path = MODELS_DIR / "3B" / author

    # Deep copy base model BEFORE loading adapter
    # PeftModel modifies base model in place, so we need a clean copy
    import copy

    base_copy = copy.deepcopy(base_model)

    lora_model = PeftModel.from_pretrained(base_copy, str(lora_path))
    lora_model = lora_model.merge_and_unload()  # Merge LoRA weights into the model
    lora_model.eval()

    # Layer evolution
    print(f"\n{'='*60}")
    print(f"LAYER EVOLUTION (LoRA) — {author}")
    print(f"{'='*60}")

    lora_evolution, lora_residual = run_single_analysis(
        lora_model,
        tokenizer,
        "lora",
        layer_names=base_layer_names,
    )

    for prompt in test_prompts:
        print_evolution_comparison(base_evolution, lora_evolution, prompt)

    # Residual stream
    print(f"\n{'='*60}")
    print(f"RESIDUAL STREAM (LoRA) — {author}")
    print(f"{'='*60}")

    print_residual_comparison(base_residual, lora_residual)

    # Activation patching
    print(f"\n{'='*60}")
    print(f"ACTIVATION PATCHING (LoRA) — {author}")
    print(f"{'='*60}")

    patch_results = run_patching_analysis(base_model, lora_model, tokenizer)
    print_patching(patch_results)

    # Store results
    all_results[author] = {
        "layer_evolution": lora_evolution,
        "residual_stream": {
            "base": {
                k: v["delta_norm"] for k, v in base_residual["contributions"].items()
            },
            "lora": {
                k: v["delta_norm"] for k, v in lora_residual["contributions"].items()
            },
        },
        "activation_patching": {
            "total_effect": patch_results["total_effect"],
            "top_components": [
                {"name": n, "effect": d["effect"], "normalized": d["normalized_effect"]}
                for n, d in sorted(
                    patch_results["patch_effects"].items(),
                    key=lambda x: abs(x[1]["normalized_effect"]),
                    reverse=True,
                )[:20]
            ],
        },
    }

    # Unload LoRA
    del lora_model

# Save results
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = BRIDGE_DIR / f"bridge_lora_{model_size}_{ts}.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n✓ Bridge results saved to {out}")
