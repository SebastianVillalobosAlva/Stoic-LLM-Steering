import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from packages.steering_extractor.activation_extractor import ActivationExtractor
from packages.steering_extractor.model_loader import ModelLoader
from packages.text_downloader.config import PROCESSED_DIR
from packages.steering_extractor.config import VECTORS_DIR


def main():
    loader = ModelLoader()
    model, tokenizer = loader.load()
    extractor = ActivationExtractor(model, tokenizer)

    all_neutral_pairs = list(PROCESSED_DIR.rglob("*.json"))
    for i, neutral_pair in enumerate(all_neutral_pairs, 1):
        steering_vector_file_name = VECTORS_DIR / (
            str(neutral_pair.parent.name) + "_steering.pt"
        )

        steering_vector = extractor.compute_steering_vector(neutral_pair)
        extractor.save_steering_vector(steering_vector, steering_vector_file_name)

        print(
            f"\n🎉 Steering vector extraction for {str(neutral_pair.parent.name)} complete!"
        )


if __name__ == "__main__":
    main()
