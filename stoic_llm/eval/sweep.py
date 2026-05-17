import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from stoic_llm.config import (
    SWEEPS_DIR,
    DEFAULT_PROMPTS,
    LAYER_IDX,
    COEFFICIENT,
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
            result = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
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
        )

        outputs = runner.run_model_with_hook(return_output=True)
        runner.cleanup()

        return outputs

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
                    "comparisons": eval_result["comparisons"],
                }
            )

            print(f"  Aggregate: {eval_result['avg_steered']['aggregate']:.2f}")

        # Find best layer
        best = max(layer_results, key=lambda r: r["aggregate"])

        result = {
            "sweep_type": "layer",
            "author": author,
            "fixed_coefficient": coefficient,
            "layers_tested": layers,
            "results": layer_results,
            "best_layer": best["layer"],
            "best_aggregate": best["aggregate"],
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
                    "comparisons": eval_result["comparisons"],
                }
            )

            print(f"  Aggregate: {eval_result['avg_steered']['aggregate']:.2f}")

        # Find best coefficient
        best = max(coeff_results, key=lambda r: r["aggregate"])

        result = {
            "sweep_type": "coefficient",
            "author": author,
            "fixed_layer": layer,
            "coefficients_tested": coefficients,
            "results": coeff_results,
            "best_coefficient": best["coefficient"],
            "best_aggregate": best["aggregate"],
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
                "aggregate": coeff_sweep["best_aggregate"],
            },
            "timestamp": datetime.now().isoformat(),
        }

        print(f"\n{'='*60}")
        print(f"OPTIMAL: layer={best_layer}, coefficient={best_coeff}")
        print(f"Aggregate score: {coeff_sweep['best_aggregate']:.2f}")
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
    """Human-readable summary of sweep results."""
    sweep_type = results.get("sweep_type", "unknown")
    author = results.get("author", "unknown")

    lines = [f"Sweep: {sweep_type} — {author}", ""]

    if sweep_type in ("layer", "full"):
        layer_data = results if sweep_type == "layer" else results["layer_sweep"]
        lines.append(
            f"Layer Sweep (coefficient={layer_data.get('fixed_coefficient', '?')}):"
        )

        for r in layer_data["results"]:
            marker = " ← best" if r["layer"] == layer_data["best_layer"] else ""
            lines.append(
                f"  Layer {r['layer']:>2d}: aggregate={r['aggregate']:.2f}{marker}"
            )
        lines.append("")

    if sweep_type in ("coefficient", "full"):
        coeff_data = (
            results if sweep_type == "coefficient" else results["coefficient_sweep"]
        )
        lines.append(f"Coefficient Sweep (layer={coeff_data.get('fixed_layer', '?')}):")

        for r in coeff_data["results"]:
            marker = (
                " ← best" if r["coefficient"] == coeff_data["best_coefficient"] else ""
            )
            lines.append(
                f"  {r['coefficient']:.3f}: aggregate={r['aggregate']:.2f}{marker}"
            )
        lines.append("")

    if sweep_type == "full":
        opt = results["optimal"]
        lines.append(
            f"Optimal: layer={opt['layer']}, coefficient={opt['coefficient']}, aggregate={opt['aggregate']:.2f}"
        )

    return "\n".join(lines)
