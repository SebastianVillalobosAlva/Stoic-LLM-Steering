import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from packages.steering_runner.runner import SteeringRunner
from packages.steering_extractor.model_loader import ModelLoader
from packages.steering_extractor.config import VECTORS_DIR


def main():
    loader = ModelLoader()
    model, tokenizer = loader.load()

    all_steering_vectors = list(VECTORS_DIR.rglob("*.pt"))
    for steering_vector_file in all_steering_vectors:
        steering_runner = SteeringRunner(steering_vector_file, model, tokenizer)
        print(f"\n🎉 Running model for {steering_runner.author}\n")
        steering_runner.run_model_with_hook()
        steering_runner.cleanup()


if __name__ == "__main__":
    main()
