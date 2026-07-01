"""Forced-choice dilemma evaluation for Stoic steering vectors.

Style-independent test of whether CAA steering shifts *decisions*
(reasoning), not just register (style).

Measurement: present a situation with two options labeled A/B, take ONE
forward pass, and compare next-token probability mass on the option labels.
P(stoic) = softmax over the two label tokens only. Every dilemma is run in
BOTH label orders (stoic=A, then stoic=B) and averaged, which cancels
positional/label bias exactly. No generation, no judge — pure logits.

Usage:
    python scripts/run_dilemma_eval.py
or:
    ev = DilemmaEval(model, tokenizer, dilemmas_path)
    results = ev.run_all(configs)   # configs per philosopher
    ev.save_results(results)

Place the dilemma set at: data/config/dilemmas.json
Steering vectors expected as {layer: tensor} dicts (post-Exp-8 format)
at: data/steering_vectors/{author}_clean.pt
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional
from peft import PeftModel
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if placed elsewhere
DILEMMAS_PATH = PROJECT_ROOT / "data" / "config" / "dilemma-set-v2.json"
VECTORS_DIR = PROJECT_ROOT / "data" / "steering_vectors"
RESULTS_DIR = PROJECT_ROOT / "results" / "dilemmas-v2"

PROMPT_TEMPLATE = (
    "Consider the following situation and choose the better course of action.\n\n"
    "Situation: {situation}\n\n"
    "A) {option_a}\n"
    "B) {option_b}\n\n"
    "The better course of action is option"
)


class DilemmaEval:
    """Judge-free forced-choice evaluation of steering vectors."""

    def __init__(self, model, tokenizer, dilemmas_path: Path = DILEMMAS_PATH):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self._hook_handle = None

        with open(dilemmas_path) as f:
            payload = json.load(f)
        self.dilemmas: list[dict] = payload["dilemmas"]
        self.meta: dict = payload.get("meta", {})

        # Option-label token ids. Llama tokenizes " A" / " B" with leading
        # space as single tokens; fail loudly if that assumption breaks.
        self.tok_a = self._single_token_id(" A")
        self.tok_b = self._single_token_id(" B")

    def _single_token_id(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Label {text!r} tokenizes to {len(ids)} tokens ({ids}); "
                "pick a label that is a single token for this tokenizer."
            )
        return ids[0]

    def _register_hook(
        self, vector: torch.Tensor, layer_idx: int, coefficient: float
    ) -> None:
        """Add coefficient * vector to the MLP output at layer_idx.

        Same injection site/convention as SteeringRunner. Always remove
        any existing hook first (no stacking).
        """
        self._remove_hook()
        vec = vector.to(dtype=next(self.model.parameters()).dtype)

        def hook(_module, _inputs, output):
            return output + coefficient * vec

        target = self.model.model.layers[layer_idx].mlp
        self._hook_handle = target.register_forward_hook(hook)

    def _remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def _p_first_label(self, prompt: str) -> float:
        """P(label 'A') normalized over {A, B} from one forward pass."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        logits = self.model(**inputs).logits[0, -1]
        two = torch.stack([logits[self.tok_a], logits[self.tok_b]]).float()
        return torch.softmax(two, dim=0)[0].item()

    def p_stoic(self, dilemma: dict) -> float:
        """Order-debiased P(stoic option): mean over both label orders."""
        p1 = self._p_first_label(
            PROMPT_TEMPLATE.format(
                situation=dilemma["situation"],
                option_a=dilemma["stoic"],
                option_b=dilemma["nonstoic"],
            )
        )  # stoic is A -> want P(A)
        p2 = self._p_first_label(
            PROMPT_TEMPLATE.format(
                situation=dilemma["situation"],
                option_a=dilemma["nonstoic"],
                option_b=dilemma["stoic"],
            )
        )  # stoic is B -> want P(B) = 1 - P(A)
        return 0.5 * (p1 + (1.0 - p2))

    def eval_condition(
        self,
        vector: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        coefficient: float = 0.0,
    ) -> dict[str, float]:
        """P(stoic) for every dilemma under one condition.

        vector=None -> unsteered baseline.
        """
        steered = vector is not None
        try:
            if steered:
                self._register_hook(vector, layer_idx, coefficient)
            out: dict[str, float] = {}
            for d in self.dilemmas:
                out[d["id"]] = self.p_stoic(d)
            return out
        finally:
            self._remove_hook()

    @staticmethod
    def _paired_stats(deltas: list[float]) -> dict:
        n = len(deltas)
        mean = sum(deltas) / n
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(var)
        se = std / math.sqrt(n) if n > 0 else float("nan")
        t = mean / se if se > 0 else float("nan")
        result = {"n": n, "mean_delta": mean, "std": std, "t_stat": t, "p_value": None}
        try:
            from scipy import stats as sps

            result["p_value"] = float(
                sps.ttest_rel([0.0] * n, [-d for d in deltas]).pvalue
            )
        except ImportError:
            result["note"] = (
                "scipy not installed; p_value omitted (report t_stat, df=n-1)"
            )
        return result

    def _bucketed(self, deltas_by_id: dict[str, float], key: str) -> dict[str, dict]:
        buckets: dict[str, list[float]] = {}
        for d in self.dilemmas:
            buckets.setdefault(d[key], []).append(deltas_by_id[d["id"]])
        return {k: self._paired_stats(v) for k, v in buckets.items()}

    def run_all(self, configs: dict[str, dict]) -> dict:
        """Run baseline once, then each philosopher's steered condition.

        configs example:
            {"marcus":    {"layer": 26, "coeff": 0.11, "vector_file": "marcus_clean.pt"},
             "seneca":    {"layer": 4,  "coeff": 0.11, "vector_file": "seneca_clean.pt"},
             "epictetus": {"layer": 8,  "coeff": 0.11, "vector_file": "epictetus_clean.pt"}}

        vector_file should load to a {layer: tensor} dict (Exp-8+ format)
        or a single tensor.
        """
        t0 = time.time()
        print(f"Baseline (unsteered) over {len(self.dilemmas)} dilemmas x 2 orders ...")
        baseline = self.eval_condition()

        results: dict = {
            "meta": {
                "n_dilemmas": len(self.dilemmas),
                "configs": {
                    k: {kk: vv for kk, vv in v.items() if kk != "vector"}
                    for k, v in configs.items()
                },
                "measurement": "P(stoic) = softmax over {A,B} label tokens, averaged over both label orders",
            },
            "baseline_p_stoic": baseline,
            "baseline_mean": sum(baseline.values()) / len(baseline),
            "philosophers": {},
        }

        for name, cfg in configs.items():
            layer, coeff = cfg["layer"], cfg["coeff"]
            loaded = torch.load(VECTORS_DIR / cfg["vector_file"], map_location="cpu")
            vector = loaded[layer] if isinstance(loaded, dict) else loaded

            print(f"{name}: layer {layer}, coeff {coeff} ...")
            steered = self.eval_condition(vector, layer, coeff)
            deltas = {i: steered[i] - baseline[i] for i in steered}
            deltas_logodds = {
                i: self._logit(steered[i]) - self._logit(baseline[i]) for i in steered
            }

            results["philosophers"][name] = {
                "steered_p_stoic": steered,
                "steered_mean": sum(steered.values()) / len(steered),
                "per_item_delta": deltas,
                "per_item_delta_logit": deltas_logodds,
                "overall": self._paired_stats(list(deltas.values())),
                "overall_logodds": self._paired_stats(list(deltas_logodds.values())),
                "by_stance": self._bucketed(deltas, "stoic_stance"),
                "by_concept": self._bucketed(deltas, "concept"),
            }

        results["meta"]["runtime_sec"] = round(time.time() - t0, 1)
        return results

    def _logit(self, p: float, eps: float = 1e-6) -> float:
        p = min(max(p, eps), 1 - eps)  # clamp so 0/1 don't blow up
        return math.log(p / (1 - p))

    def sweep_coefficients(
        self,
        name: str,
        layer: int,
        vector_file: str,
        coefficients: list[float],
    ) -> dict:
        """Baseline once, then sweep one philosopher's vector over coefficients.

        Judge-free + generation-free, so high coefficients are safe here —
        we only need the option logits to move, not coherent text.
        """
        loaded = torch.load(VECTORS_DIR / vector_file, map_location="cpu")
        vector = loaded[layer] if isinstance(loaded, dict) else loaded

        baseline = self.eval_condition()
        base_mean = sum(baseline.values()) / len(baseline)

        out = {"name": name, "layer": layer, "baseline_mean": base_mean, "by_coeff": {}}
        for c in coefficients:
            print(f"Trying coeff - {c}")
            steered = self.eval_condition(vector, layer, c)
            d_p = {i: steered[i] - baseline[i] for i in steered}
            d_lo = {
                i: self._logit(steered[i]) - self._logit(baseline[i]) for i in steered
            }
            out["by_coeff"][c] = {
                "steered_mean": sum(steered.values()) / len(steered),
                "overall": self._paired_stats(list(d_p.values())),
                "overall_logodds": self._paired_stats(list(d_lo.values())),
                "by_stance": self._bucketed(d_p, "stoic_stance"),
            }
        return out

    @staticmethod
    def summarize_sweep(sweep: dict) -> str:
        lines = [
            f"{sweep['name']} (layer {sweep['layer']})  baseline P(stoic)={sweep['baseline_mean']:.3f}",
            f"{'coeff':>6} {'P(stoic)':>10} {'ΔP':>9} {'Δlog-odds':>11} {'t(lo)':>7}",
        ]
        for c, r in sweep["by_coeff"].items():
            lines.append(
                f"{c:>6.2f} {r['steered_mean']:>10.3f} "
                f"{r['overall']['mean_delta']:>+9.3f} "
                f"{r['overall_logodds']['mean_delta']:>+11.3f} "
                f"{r['overall_logodds']['t_stat']:>7.2f}"
            )
        return "\n".join(lines)

    @staticmethod
    def save_results(results: dict, out_dir: Path = RESULTS_DIR) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"dilemma_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved -> {path}")
        return path

    @staticmethod
    def summarize(results: dict) -> str:
        """Human-readable summary table."""
        lines = [
            f"Baseline mean P(stoic): {results['baseline_mean']:.3f}",
            "",
            f"{'philosopher':<12} {'ΔP(stoic)':>10} {'Δlog-odds':>11} {'t(lo)':>7}   stance check (ΔP)",
        ]
        for name, r in results["philosophers"].items():
            dp = r["overall"]["mean_delta"]
            lo = r["overall_logodds"]["mean_delta"]
            t_lo = r["overall_logodds"]["t_stat"]
            stance = r["by_stance"]
            stance_str = "  ".join(
                f"{k}: {v['mean_delta']:+.3f}" for k, v in sorted(stance.items())
            )
            lines.append(
                f"{name:<12} {dp:>+10.3f} {lo:>+11.3f} {t_lo:>7.2f}   {stance_str}"
            )
        lines += [
            "",
            "Δlog-odds decompresses effects the baseline ceiling hides; t(lo) is its paired t.",
            "Read: Δlog-odds > 0 with t(lo) >~ 2 across BOTH stance buckets = reasoning shift.",
            "Positive only on 'accepting' = passivity confound, not Stoic reasoning.",
            "",
            f"Calibration gate: baseline mean P(stoic) = {results['baseline_mean']:.3f} "
            f"(want 0.35-0.65 with spread before trusting deltas).",
        ]
        return "\n".join(lines)


"""LoRA variant of the forced-choice dilemma eval.

Reuses the full DilemmaEval machinery (p_stoic, both-order debiasing,
paired stats, stance/concept buckets). The ONLY difference from CAA: there
is no steering hook. The adapter is merged into the weights, so the
"steered" condition is just a forward pass through the merged model.

The base-model baseline is computed ONCE, on the unmodified base, and is
the SAME baseline the CAA run used — so the CAA-vs-LoRA comparison holds
everything fixed except the intervention.

Usage:
    from stoic_llm.eval.dilemma import DilemmaEval
    from stoic_llm.model import ModelLoader

    base, tok = ModelLoader().load()           # unmodified base
    ev = LoRADilemmaEval(base, tok)            # computes baseline on base
    results = ev.run_all_lora({
        "marcus":    "models/lora_marcus_clean",
        "seneca":    "models/lora_seneca_clean",
        "epictetus": "models/lora_epictetus_clean",
    })
    ev.save_results(results)
    print(ev.summarize(results))
"""


class LoRADilemmaEval(DilemmaEval):
    """Dilemma eval for LoRA adapters (merged weights, no hook)."""

    def __init__(self, base_model, tokenizer, dilemmas_path=None):
        # Keep a handle to the pristine base so we can swap adapters in/out.
        kwargs = {"dilemmas_path": dilemmas_path} if dilemmas_path else {}
        super().__init__(base_model, tokenizer, **kwargs)
        self._base_model = base_model
        self._base_state = "base"  # which model is currently in self.model

    # ---- override: "steered" = merged adapter, no hook ----
    @torch.no_grad()
    def eval_condition_lora(self, merged_model) -> dict[str, float]:
        """P(stoic) for every dilemma using an already-merged LoRA model."""
        prev = self.model
        self.model = merged_model
        try:
            return {d["id"]: self.p_stoic(d) for d in self.dilemmas}
        finally:
            self.model = prev  # restore base for the next condition / baseline

    # ---- load + merge one adapter onto a FRESH copy of the base ----
    def _merged(self, adapter_dir: str):
        from stoic_llm.model import ModelLoader
        from peft import PeftModel

        base = ModelLoader("3B").load()[0]  # fresh base, no prior adapters
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged.config.use_cache = True
        return merged

    # ---- full run ----
    def run_all_lora(self, adapter_dirs: dict[str, str]) -> dict:
        t0 = time.time()

        # Baseline on the unmodified base — identical to the CAA run's baseline.
        print(
            f"Baseline (base model) over {len(self.dilemmas)} dilemmas x 2 orders ..."
        )
        baseline = self.eval_condition()  # vector=None -> pure base
        base_mean = sum(baseline.values()) / len(baseline)

        results = {
            "meta": {
                "n_dilemmas": len(self.dilemmas),
                "method": "LoRA (merged weights, no hook)",
                "adapter_dirs": {k: str(v) for k, v in adapter_dirs.items()},
                "measurement": "P(stoic) = softmax over {A,B}, averaged over both label orders",
                "note": "Baseline computed on unmodified base; same baseline basis as CAA run.",
            },
            "baseline_p_stoic": baseline,
            "baseline_mean": base_mean,
            "philosophers": {},
        }

        for name, adapter_dir in adapter_dirs.items():
            print(f"{name}: merging {adapter_dir} ...")
            merged = self._merged(adapter_dir)
            try:
                steered = self.eval_condition_lora(merged)
            finally:
                # merge_and_unload mutates a copy; drop it and reclaim memory.
                # NOTE: confirm self._base_model is still pristine after this
                # (see caution in the chat). If not, reload base per author.
                del merged
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            deltas = {i: steered[i] - baseline[i] for i in steered}
            deltas_lo = {
                i: self._logit(steered[i]) - self._logit(baseline[i]) for i in steered
            }
            results["philosophers"][name] = {
                "steered_p_stoic": steered,
                "steered_mean": sum(steered.values()) / len(steered),
                "per_item_delta": deltas,
                "per_item_delta_logit": deltas_lo,
                "overall": self._paired_stats(list(deltas.values())),
                "overall_logodds": self._paired_stats(list(deltas_lo.values())),
                "by_stance": self._bucketed(deltas, "stoic_stance"),
                "by_concept": self._bucketed(deltas, "concept"),
            }

        results["meta"]["runtime_sec"] = round(time.time() - t0, 1)
        return results
