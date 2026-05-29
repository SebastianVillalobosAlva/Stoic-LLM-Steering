"""
scripts/extract_vectors.py — Extract steering vectors

Usage:
  python scripts/extract_vectors.py          # defaults to 1B
  python scripts/extract_vectors.py 3B       # uses 3B
"""

import sys
from stoic_llm.model import ModelLoader
from stoic_llm.steering.extractor import ActivationExtractor
from stoic_llm.config import PROCESSED_DIR, VECTORS_DIR

model_size = sys.argv[1] if len(sys.argv) > 1 else "1B"

CANDIDATE_LAYERS = {
    "1B": [4, 6, 8, 10, 12, 14],
    "3B": [4, 8, 12, 16, 20, 24, 26],
}
layers = CANDIDATE_LAYERS[model_size]

loader = ModelLoader(model_size)
model, tokenizer = loader.load()
extractor = ActivationExtractor(model, tokenizer)

for author in ["marcus_aurelius", "seneca", "epictetus"]:
    print(f"\n{'='*60}")
    print(f"Extracting steering vector for {author} ({model_size})")
    print(f"{'='*60}")

    pairs_file = PROCESSED_DIR / author / "neutral_pairs.json"
    output_file = VECTORS_DIR / f"{author}_steering_{model_size}.pt"
    vectors = extractor.compute_layered_steering_vectors(str(pairs_file), layers)
    extractor.save_steering_vectors(vectors, output_file)

    print(f"✓ Saved to {output_file}")

print("\nDone! All steering vectors extracted.")
