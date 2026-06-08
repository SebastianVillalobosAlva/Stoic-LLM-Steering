import json
import statistics
import torch
from pathlib import Path
from typing import Dict, List, Optional, Literal
from datetime import datetime
from stoic_llm.config import (
    SWEEPS_DIR,
    DEFAULT_PROMPTS,
    LAYER_IDX,
    COEFFICIENT,
    GEN_KWARGS,
)
from stoic_llm.steering.runner import SteeringRunner
from stoic_llm.eval.judge import StoicJudge, summarize_eval


class SteeringSweep:
    """
    Run hyperparameter sweeps over steering layer and coefficient,
    scoring each configuration with the LLM-as-judge.
    """

    def __init__(
        self,
        model,
        tokenizer,
        vector_path: str,
        judge: Optional[StoicJudge] = None,
        prompts: Optional[List[str]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.vector_path = vector_path
        self.judge = judge or StoicJudge()
        self.prompts = prompts or DEFAULT_PROMPTS

        # Generate unsteered baseline once
        self._baseline = None

    def _get_baseline(self) -> List[str]:
        """Generate unsteered outputs (cached)."""
        if self._baseline is not None:
            return self._baseline

        print("Generating unsteered baseline...")
        outputs = []
        for prompt in self.prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            result = self.model.generate(**inputs, **GEN_KWARGS)
            text = self.tokenizer.decode(result[0], skip_special_tokens=True)
            outputs.append(text)

        self._baseline = outputs
        return outputs

    def _run_steered(self, layer: int, coefficient: float) -> List[str]:
        """Generate steered outputs for a given layer and coefficient."""
        runner = SteeringRunner(
            file_path=self.vector_path,
            model=self.model,
            tokenizer=self.tokenizer,
            layer=layer,
            coefficient=coefficient,
            prompts=self.prompts,
            do_sample=False,  # override the bad __init__ default
            temperature=0.0,
        )

        outputs = runner.run_model_with_hook(
            return_output=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        runner.cleanup()

        return outputs

    def _run_steered_sampled(
        self, layer: int, coefficient: float, temperature: float
    ) -> List[str]:
        """Steered generation with SAMPLING (for vary='generation' seed eval).
        Seed must be set by the caller via torch.manual_seed before this."""
        runner = SteeringRunner(
            file_path=self.vector_path,
            model=self.model,
            tokenizer=self.tokenizer,
            layer=layer,
            coefficient=coefficient,
            prompts=self.prompts,
            do_sample=True,
            temperature=temperature,
        )
        outputs = runner.run_model_with_hook(
            return_output=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        runner.cleanup()
        return outputs

    def seed_eval(
        self,
        layer: int,
        coefficient: float,
        author: str = "unknown",
        n_seeds: int = 5,
        vary: Literal["judge", "generation"] = "judge",
        temperature: float = 0.7,
    ) -> Dict:
        """
        Replicated evaluation of ONE config to get content mean ± std.

        vary="judge": generate once (greedy), score n_seeds times.
            Isolates LLM-as-judge variance. Use this to test whether the
            noise you saw in the sweep is judge noise (it almost certainly is,
            since sweep decoding is greedy/deterministic).

        vary="generation": sample n_seeds generations (do_sample=True, one
            seed each), score each once. Measures total pipeline variance.
            Requires stochastic decoding to be meaningful.

        Returns per-seed content scores plus mean/std for content and aggregate.
        """
        print(f"\n{'='*60}")
        print(f"SEED EVAL — {author} L{layer} c{coefficient} "
              f"× {n_seeds} seeds (vary={vary})")
        print(f"{'='*60}")

        baseline = self._get_baseline()

        per_seed = []  # list of (content, aggregate)

        if vary == "judge":
            # Generate ONCE (greedy, matched to sweep), judge n times.
            steered = self._run_steered(layer, coefficient)
            for s in range(n_seeds):
                er = self.judge.evaluate_steering(
                    prompts=self.prompts,
                    steered_outputs=steered,
                    unsteered_outputs=baseline,
                    author=author,
                    metadata={"layer": layer, "coefficient": coefficient,
                              "seed": s, "vary": "judge"},
                )
                per_seed.append((er["content"], er["avg_steered"]["aggregate"]))
                print(f"  seed {s}: content={er['content']:+.3f}  "
                      f"aggregate={er['avg_steered']['aggregate']:.3f}")

        else:  # vary == "generation"
            for s in range(n_seeds):
                torch.manual_seed(s)
                steered = self._run_steered_sampled(layer, coefficient, temperature)
                er = self.judge.evaluate_steering(
                    prompts=self.prompts,
                    steered_outputs=steered,
                    unsteered_outputs=baseline,
                    author=author,
                    metadata={"layer": layer, "coefficient": coefficient,
                              "seed": s, "vary": "generation"},
                )
                per_seed.append((er["content"], er["avg_steered"]["aggregate"]))
                print(f"  seed {s}: content={er['content']:+.3f}  "
                      f"aggregate={er['avg_steered']['aggregate']:.3f}")

        contents = [c for c, _ in per_seed]
        aggregates = [a for _, a in per_seed]

        def mean_std(xs):
            m = statistics.mean(xs)
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            return m, sd

        c_mean, c_std = mean_std(contents)
        a_mean, a_std = mean_std(aggregates)

        result = {
            "eval_type": "seed",
            "author": author,
            "layer": layer,
            "coefficient": coefficient,
            "n_seeds": n_seeds,
            "vary": vary,
            "content_scores": contents,
            "aggregate_scores": aggregates,
            "content_mean": c_mean,
            "content_std": c_std,
            "aggregate_mean": a_mean,
            "aggregate_std": a_std,
            "content_ci_excludes_zero": (c_mean - c_std) > 0 or (c_mean + c_std) < 0,
            "timestamp": datetime.now().isoformat(),
        }

        print(f"\n  content   = {c_mean:+.3f} ± {c_std:.3f}")
        print(f"  aggregate = {a_mean:.3f} ± {a_std:.3f}")
        verdict = ("SURVIVES (±1σ excludes 0)" if result["content_ci_excludes_zero"]
                   else "NOT distinguishable from 0")
        print(f"  content effect: {verdict}")

        return result

    def seed_eval_candidates(
        self,
        candidates: List[Dict],
        author: str = "unknown",
        n_seeds: int = 5,
        vary: Literal["judge", "generation"] = "judge",
    ) -> Dict:
        """
        Run seed_eval over several candidate (layer, coefficient) configs and
        rank by content_mean. `candidates` = [{"layer": 20, "coefficient": 0.15}, ...]
        """
        runs = []
        for cfg in candidates:
            runs.append(self.seed_eval(
                layer=cfg["layer"],
                coefficient=cfg["coefficient"],
                author=author,
                n_seeds=n_seeds,
                vary=vary,
            ))

        runs.sort(key=lambda r: r["content_mean"], reverse=True)

        print(f"\n{'='*60}\nCANDIDATE RANKING — {author}\n{'='*60}")
        print(f"  {'layer':>5} {'coeff':>6} {'content':>16} {'survives':>10}")
        for r in runs:
            surv = "yes" if r["content_ci_excludes_zero"] else "no"
            print(f"  {r['layer']:>5} {r['coefficient']:>6.3f} "
                  f"{r['content_mean']:>+8.3f} ± {r['content_std']:<5.3f} {surv:>10}")

        return {
            "eval_type": "seed_candidates",
            "author": author,
            "n_seeds": n_seeds,
            "vary": vary,
            "runs": runs,
            "best": runs[0] if runs else None,
            "timestamp": datetime.now().isoformat(),
        }

    def sweep_layers(
        self,
        layers: Optional[List[int]] = None,
        coefficient: float = COEFFICIENT,
        author: str = "unknown",
    ) -> Dict:
        """
        Test steering at different layers with a fixed coefficient.

        Args:
            layers: List of layer indices to test.
                    Default: [4, 6, 8, 10, 12, 14]
            coefficient: Fixed coefficient to use
            author: Philosopher name for labeling

        Returns:
            Dict with per-layer results and best configuration
        """
        if layers is None:
            layers = [4, 6, 8, 10, 12, 14]

        print(f"\n{'='*60}")
        print(f"LAYER SWEEP — coefficient={coefficient}, {len(layers)} layers")
        print(f"{'='*60}")

        baseline = self._get_baseline()
        layer_results = []

        for layer in layers:
            print(f"\nLayer {layer}:")
            steered = self._run_steered(layer, coefficient)

            eval_result = self.judge.evaluate_steering(
                prompts=self.prompts,
                steered_outputs=steered,
                unsteered_outputs=baseline,
                author=author,
                metadata={"layer": layer, "coefficient": coefficient},
            )

            layer_results.append(
                {
                    "layer": layer,
                    "coefficient": coefficient,
                    "avg_steered": eval_result["avg_steered"],
                    "avg_unsteered": eval_result["avg_unsteered"],
                    "avg_deltas": eval_result["avg_deltas"],
                    "aggregate": eval_result["avg_steered"]["aggregate"],
                    "content": eval_result["content"],
                    "comparisons": eval_result["comparisons"],
                }
            )

            print(f"  Aggregate: {eval_result['avg_steered']['aggregate']:.2f}")

        # Find best layer
        best = max(layer_results, key=lambda r: r["content"])

        result = {
            "sweep_type": "layer",
            "author": author,
            "fixed_coefficient": coefficient,
            "layers_tested": layers,
            "results": layer_results,
            "best_layer": best["layer"],
            "best_content": best["content"],
            "timestamp": datetime.now().isoformat(),
        }

        print(f"\n✓ Best layer: {best['layer']} (aggregate: {best['aggregate']:.2f})")

        return result

    def sweep_coefficients(
        self,
        coefficients: Optional[List[float]] = None,
        layer: int = LAYER_IDX,
        author: str = "unknown",
    ) -> Dict:
        """
        Test different steering strengths at a fixed layer.

        Args:
            coefficients: List of coefficients to test.
                         Default: [0.03, 0.05, 0.08, 0.11, 0.15, 0.2, 0.3]
            layer: Fixed layer index
            author: Philosopher name for labeling

        Returns:
            Dict with per-coefficient results and best configuration
        """
        if coefficients is None:
            coefficients = [0.03, 0.05, 0.08, 0.11, 0.15, 0.2, 0.3]

        print(f"\n{'='*60}")
        print(f"COEFFICIENT SWEEP — layer={layer}, {len(coefficients)} values")
        print(f"{'='*60}")

        baseline = self._get_baseline()
        coeff_results = []

        for coeff in coefficients:
            print(f"\nCoefficient {coeff}:")
            steered = self._run_steered(layer, coeff)

            eval_result = self.judge.evaluate_steering(
                prompts=self.prompts,
                steered_outputs=steered,
                unsteered_outputs=baseline,
                author=author,
                metadata={"layer": layer, "coefficient": coeff},
            )

            coeff_results.append(
                {
                    "layer": layer,
                    "coefficient": coeff,
                    "avg_steered": eval_result["avg_steered"],
                    "avg_unsteered": eval_result["avg_unsteered"],
                    "avg_deltas": eval_result["avg_deltas"],
                    "aggregate": eval_result["avg_steered"]["aggregate"],
                    "content": eval_result["content"],
                    "comparisons": eval_result["comparisons"],
                }
            )

            print(f"  Aggregate: {eval_result['avg_steered']['aggregate']:.2f}")

        # Find best coefficient
        best = max(coeff_results, key=lambda r: r["content"])

        result = {
            "sweep_type": "coefficient",
            "author": author,
            "fixed_layer": layer,
            "coefficients_tested": coefficients,
            "results": coeff_results,
            "best_coefficient": best["coefficient"],
            "best_content": best["content"],
            "timestamp": datetime.now().isoformat(),
        }

        print(
            f"\n✓ Best coefficient: {best['coefficient']} (aggregate: {best['aggregate']:.2f})"
        )

        return result

    def full_sweep(
        self,
        layers: Optional[List[int]] = None,
        coefficients: Optional[List[float]] = None,
        author: str = "unknown",
    ) -> Dict:
        """
        Two-stage sweep: first find best layer, then sweep coefficients
        at that layer.

        Returns:
            Dict with both sweep results and final optimal configuration
        """
        print(f"\n{'='*60}")
        print(f"FULL SWEEP — {author}")
        print(f"{'='*60}")

        # Stage 1: Find best layer
        layer_sweep = self.sweep_layers(
            layers=layers,
            coefficient=COEFFICIENT,
            author=author,
        )
        best_layer = layer_sweep["best_layer"]

        # Stage 2: Sweep coefficients at best layer
        coeff_sweep = self.sweep_coefficients(
            coefficients=coefficients,
            layer=best_layer,
            author=author,
        )
        best_coeff = coeff_sweep["best_coefficient"]

        result = {
            "sweep_type": "full",
            "author": author,
            "layer_sweep": layer_sweep,
            "coefficient_sweep": coeff_sweep,
            "optimal": {
                "layer": best_layer,
                "coefficient": best_coeff,
                "content": coeff_sweep["best_content"],  # was best_aggregate
            },
            "timestamp": datetime.now().isoformat(),
        }

        print(f"\n{'='*60}")
        print(f"OPTIMAL: layer={best_layer}, coefficient={best_coeff}")
        print(
            f"Content score: {coeff_sweep['best_content']:+.2f}"
        )  # was best_aggregate
        print(f"{'='*60}")

        return result

    def save_results(self, results: Dict, filename: Optional[str] = None) -> Path:
        """Save sweep results to JSON."""
        if filename is None:
            sweep_type = results.get("sweep_type", "sweep")
            author = results.get("author", "unknown")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{sweep_type}_{author}_{ts}.json"

        output_path = SWEEPS_DIR / filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✓ Sweep results saved to {output_path}")
        return output_path


def summarize_sweep(results: Dict) -> str:
    """Human-readable summary of sweep results.

    Selection is on `content` (Stoic content delta = depth + alignment).
    `aggregate` (absolute, all four dimensions) is shown for reference only —
    the ← best marker tracks content, so it may not sit on the max aggregate.
    """
    sweep_type = results.get("sweep_type", "unknown")
    author = results.get("author", "unknown")

    lines = [f"Sweep: {sweep_type} — {author}", ""]

    if sweep_type in ("layer", "full"):
        layer_data = results if sweep_type == "layer" else results["layer_sweep"]
        lines.append(
            f"Layer Sweep (coefficient={layer_data.get('fixed_coefficient', '?')}):"
        )
        lines.append(f"  {'layer':>5}  {'content':>8}  {'aggregate':>9}")

        for r in layer_data["results"]:
            marker = " ← best" if r["layer"] == layer_data["best_layer"] else ""
            lines.append(
                f"  {r['layer']:>5d}  {r['content']:>+8.2f}  {r['aggregate']:>9.2f}{marker}"
            )
        lines.append("")

    if sweep_type in ("coefficient", "full"):
        coeff_data = (
            results if sweep_type == "coefficient" else results["coefficient_sweep"]
        )
        lines.append(f"Coefficient Sweep (layer={coeff_data.get('fixed_layer', '?')}):")
        lines.append(f"  {'coeff':>5}  {'content':>8}  {'aggregate':>9}")

        for r in coeff_data["results"]:
            marker = (
                " ← best" if r["coefficient"] == coeff_data["best_coefficient"] else ""
            )
            lines.append(
                f"  {r['coefficient']:>5.3f}  {r['content']:>+8.2f}  {r['aggregate']:>9.2f}{marker}"
            )
        lines.append("")

    if sweep_type == "full":
        opt = results["optimal"]
        lines.append(
            f"Optimal: layer={opt['layer']}, coefficient={opt['coefficient']}, "
            f"content={opt.get('content', float('nan')):+.2f}, "
            f"aggregate={opt.get('aggregate', float('nan')):.2f}"
        )

    return "\n".join(lines)
