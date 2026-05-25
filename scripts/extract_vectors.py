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

loader = ModelLoader(model_size)
model, tokenizer = loader.load()
extractor = ActivationExtractor(model, tokenizer)

authors = ["marcus_aurelius", "seneca", "epictetus"]

for author in authors:
    print(f"\n{'='*60}")
    print(f"Extracting steering vector for {author} ({model_size})")
    print(f"{'='*60}")

    pairs_file = PROCESSED_DIR / author / "neutral_pairs.json"
    output_file = VECTORS_DIR / f"{author}_steering_{model_size}.pt"

    steering_vector = extractor.compute_steering_vector(str(pairs_file))
    extractor.save_steering_vector(steering_vector, output_file)

    print(f"✓ Saved to {output_file}")

print("\nDone! All steering vectors extracted.")
