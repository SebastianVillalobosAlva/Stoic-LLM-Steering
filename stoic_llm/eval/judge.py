import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import anthropic
from stoic_llm.config import JUDGES_DIR

STOIC_RUBRIC = """
Score the following text on how well it reflects Stoic philosophical principles.
Evaluate on these 4 dimensions, each scored 1-5:

1. PHILOSOPHICAL DEPTH (1-5)
   1 = No philosophical content, generic or off-topic
   3 = Some philosophical ideas but surface-level
   5 = Deep engagement with Stoic concepts (virtue, reason, nature, acceptance)

2. STOIC ALIGNMENT (1-5)
   1 = Contradicts Stoic principles or is philosophically neutral
   3 = Loosely aligned with Stoic ideas
   5 = Clearly reflects core Stoic doctrines (dichotomy of control, virtue as
       sole good, living according to nature, rational acceptance)

3. COHERENCE (1-5)
   1 = Incoherent, repetitive, or nonsensical
   3 = Readable but disorganized or partially repetitive
   5 = Clear, well-structured, logically flowing

4. STYLISTIC AUTHENTICITY (1-5)
   1 = Modern casual language, no philosophical register
   3 = Some philosophical tone but inconsistent
   5 = Reads like translated ancient philosophical text (aphoristic,
       contemplative, uses philosophical vocabulary naturally)

Respond ONLY with a JSON object in this exact format, no other text:
{"philosophical_depth": X, "stoic_alignment": X, "coherence": X, "stylistic_authenticity": X, "reasoning": "brief explanation"}
""".strip()


class StoicJudge:
    """
    Uses Claude to evaluate model outputs against a Stoic philosophy rubric.
    """

    def __init__(
        self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY env var or pass api_key."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def score(self, text: str, prompt: str = "") -> Dict:
        """
        Score a single text output against the Stoic rubric.

        Args:
            text: The generated text to evaluate
            prompt: The prompt that produced the text (for context)

        Returns:
            Dict with scores per dimension and reasoning
        """
        user_message = f"{STOIC_RUBRIC}\n\n"
        if prompt:
            user_message += f"PROMPT: {prompt}\n\n"
        user_message += f"TEXT TO EVALUATE:\n{text}"

        message = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": user_message}],
        )

        response_text = message.content[0].text

        try:
            scores = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            scores = _extract_json(response_text)

        # Compute aggregate score
        dimensions = [
            "philosophical_depth",
            "stoic_alignment",
            "coherence",
            "stylistic_authenticity",
        ]
        valid_scores = [scores.get(d, 0) for d in dimensions]
        scores["aggregate"] = sum(valid_scores) / len(valid_scores)

        return scores

    def evaluate_batch(
        self,
        outputs: List[Dict],
        delay: float = 0.5,
    ) -> List[Dict]:
        """
        Score a batch of outputs.

        Args:
            outputs: List of {"prompt": str, "text": str, ...} dicts
            delay: Seconds between API calls to avoid rate limits

        Returns:
            List of dicts with original data + scores
        """
        results = []
        total = len(outputs)

        for i, item in enumerate(outputs, 1):
            print(f"Scoring {i}/{total}...", end="\r")

            scores = self.score(item["text"], item.get("prompt", ""))
            result = {**item, "scores": scores}
            results.append(result)

            if i < total:
                time.sleep(delay)

        print(f"✓ Scored {total} outputs")
        return results

    def compare(
        self,
        prompt: str,
        unsteered_text: str,
        steered_text: str,
    ) -> Dict:
        """
        Score both steered and unsteered outputs for the same prompt
        and compute the improvement.

        Returns:
            Dict with unsteered scores, steered scores, and deltas
        """
        unsteered_scores = self.score(unsteered_text, prompt)
        time.sleep(0.5)
        steered_scores = self.score(steered_text, prompt)

        dimensions = [
            "philosophical_depth",
            "stoic_alignment",
            "coherence",
            "stylistic_authenticity",
            "aggregate",
        ]

        deltas = {}
        for d in dimensions:
            u = unsteered_scores.get(d, 0)
            s = steered_scores.get(d, 0)
            deltas[d] = s - u

        return {
            "prompt": prompt,
            "unsteered": {"text": unsteered_text, "scores": unsteered_scores},
            "steered": {"text": steered_text, "scores": steered_scores},
            "deltas": deltas,
        }

    def evaluate_steering(
        self,
        prompts: List[str],
        steered_outputs: List[str],
        unsteered_outputs: List[str],
        author: str = "unknown",
        metadata: Optional[Dict] = None,
        delay: float = 0.5,
    ) -> Dict:
        """
        Full evaluation: compare steered vs unsteered across multiple prompts.

        Args:
            prompts: List of prompts
            steered_outputs: Steered model outputs (same order as prompts)
            unsteered_outputs: Unsteered model outputs (same order as prompts)
            author: Philosopher name for labeling
            metadata: Extra info (layer, coefficient, etc.)
            delay: Seconds between API calls

        Returns:
            Dict with per-prompt comparisons and aggregate summary
        """
        if len(prompts) != len(steered_outputs) != len(unsteered_outputs):
            raise ValueError(
                "prompts, steered_outputs, and unsteered_outputs must have same length."
            )

        comparisons = []
        for i, (prompt, steered, unsteered) in enumerate(
            zip(prompts, steered_outputs, unsteered_outputs), 1
        ):
            print(f"Evaluating prompt {i}/{len(prompts)}...")
            comp = self.compare(prompt, unsteered, steered)
            comparisons.append(comp)
            if i < len(prompts):
                time.sleep(delay)

        # Aggregate
        dimensions = [
            "philosophical_depth",
            "stoic_alignment",
            "coherence",
            "stylistic_authenticity",
            "aggregate",
        ]

        avg_steered = {}
        avg_unsteered = {}
        avg_deltas = {}

        for d in dimensions:
            steered_vals = [c["steered"]["scores"].get(d, 0) for c in comparisons]
            unsteered_vals = [c["unsteered"]["scores"].get(d, 0) for c in comparisons]
            delta_vals = [c["deltas"].get(d, 0) for c in comparisons]

            avg_steered[d] = sum(steered_vals) / len(steered_vals)
            avg_unsteered[d] = sum(unsteered_vals) / len(unsteered_vals)
            avg_deltas[d] = sum(delta_vals) / len(delta_vals)

        result = {
            "author": author,
            "num_prompts": len(prompts),
            "comparisons": comparisons,
            "avg_steered": avg_steered,
            "avg_unsteered": avg_unsteered,
            "avg_deltas": avg_deltas,
            "content": self._content_score(avg_deltas),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        return result

    def _content_score(self, avg_deltas):
        return (avg_deltas["philosophical_depth"] + avg_deltas["stoic_alignment"]) / 2

    def save_results(self, results: Dict, filename: Optional[str] = None) -> Path:
        """Save evaluation results to JSON."""
        if filename is None:
            author = results.get("author", "unknown")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{author}_{ts}.json"

        output_path = JUDGES_DIR / filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✓ Results saved to {output_path}")
        return output_path


# =========================================================================
# Helpers
# =========================================================================


def _extract_json(text: str) -> Dict:
    """Try to extract JSON from a response that might have extra text."""
    import re

    match = re.search(r"\{[^}]+\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "philosophical_depth": 0,
        "stoic_alignment": 0,
        "coherence": 0,
        "stylistic_authenticity": 0,
        "reasoning": f"Failed to parse response: {text[:200]}",
    }


def summarize_eval(results: Dict) -> str:
    """Human-readable summary of evaluation results."""
    lines = [
        f"Evaluation: {results['author']} ({results['num_prompts']} prompts)",
        "",
        f"  {'Dimension':<25s} {'Unsteered':>10s} {'Steered':>10s} {'Delta':>10s}",
        f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}",
    ]

    dimensions = [
        "philosophical_depth",
        "stoic_alignment",
        "coherence",
        "stylistic_authenticity",
        "aggregate",
    ]

    for d in dimensions:
        u = results["avg_unsteered"].get(d, 0)
        s = results["avg_steered"].get(d, 0)
        delta = results["avg_deltas"].get(d, 0)
        marker = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        label = d.replace("_", " ").title()
        lines.append(f"  {label:<25s} {u:>10.2f} {s:>10.2f} {delta:>+10.2f} {marker}")

    if results.get("metadata"):
        lines.append("")
        lines.append(
            f"  Layer: {results['metadata'].get('layer', '?')}, "
            f"Coefficient: {results['metadata'].get('coefficient', '?')}"
        )

    return "\n".join(lines)
