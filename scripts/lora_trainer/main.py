import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from packages.lora_trainer.data_prep import LoRADataPrep
from packages.lora_trainer.trainer import LoRATrainer
from packages.lora_trainer.runner import LoRARunner

if __name__ == "__main__":
    # prep = LoRADataPrep()
    # prep.prepare_all_authors()

    # trainer = LoRATrainer()
    # trainer.train_all_authors()

    runner = LoRARunner()

    prompts = [
        "When facing difficulty, one should",
        "The nature of virtue is",
        "To live well means",
    ]

    for author in ["marcus_aurelius", "seneca", "epictetus"]:
        print(f"\n{'='*70}")
        print(f"🏛️ {author.upper()}")
        print("=" * 70)

        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 70)
            result = runner.generate(author, prompt)
            print(result)
