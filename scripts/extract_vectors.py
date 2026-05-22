from stoic_llm.model import ModelLoader
from stoic_llm.steering.extractor import ActivationExtractor
from stoic_llm.config import PROCESSED_DIR, VECTORS_DIR

# Load model
loader = ModelLoader()
model, tokenizer = loader.load()
extractor = ActivationExtractor(model, tokenizer)

# Extract steering vectors for each philosopher
authors = ["marcus_aurelius", "seneca", "epictetus"]

for author in authors:
    print(f"\n{'='*60}")
    print(f"Extracting steering vector for {author}")
    print(f"{'='*60}")

    pairs_file = PROCESSED_DIR / author / "neutral_pairs.json"
    output_file = VECTORS_DIR / f"{author}_steering.pt"

    steering_vector = extractor.compute_steering_vector(str(pairs_file))
    extractor.save_steering_vector(steering_vector, output_file)

    print(f"✓ Saved to {output_file}")

print("\nDone! All steering vectors extracted.")
