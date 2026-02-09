import json
from pathlib import Path
from packages.text_downloader.config import PROCESSED_DIR


class LoRADataPrep:
    def __init__(self, output_dir="data/lora_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def prepare_author_data(self, author_name):
        """Convert Stoic texts to training format for one author"""
        pairs_file = PROCESSED_DIR / author_name / "neutral_pairs.json"

        with open(pairs_file) as f:
            data = json.load(f)

        training_data = []
        for pair in data["pairs"]:
            training_data.append({"text": pair["stoic_text"]})

        return training_data

    def save_training_data(self, author_name):
        """Save training data as JSONL"""
        data = self.prepare_author_data(author_name)
        output_file = self.output_dir / f"{author_name}_train.jsonl"

        with open(output_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        print(f"✅ Saved {len(data)} training examples for {author_name}")
        return output_file

    def prepare_all_authors(self):
        """Prepare data for all three authors"""
        authors = ["marcus_aurelius", "seneca", "epictetus"]
        for author in authors:
            self.save_training_data(author)
