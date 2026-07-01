from stoic_llm.data.processor import TextProcessor
from stoic_llm.config import PROCESSED_DIR

FILES = {
    "marcus_aurelius": "meditations.txt",
    "seneca": "moral_letters.txt",
    "epictetus": "enchiridion.txt",
}

for author, filename in FILES.items():
    path = PROCESSED_DIR / author / filename
    if not path.exists():
        print(f"⚠ {author}: {path} not found — skipping")
        continue
    print(f"\nChunking {author} from {path.name}...")
    processor = TextProcessor(text_path=str(path))
    processor.chunk_by_paragraph()
