"""
scripts/train_lora.py — Prepare data and train LoRA adapters

Usage:
  python scripts/train_lora.py              # defaults to 1B on cpu
  python scripts/train_lora.py 3B           # 3B on cpu
  python scripts/train_lora.py 3B mps       # 3B on Apple GPU
"""

import sys
from stoic_llm.lora.data_prep import LoRADataPrep
from stoic_llm.lora.trainer import LoRATrainer

model_size = sys.argv[1] if len(sys.argv) > 1 else "1B"
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

# Step 1: Prepare training data
print(f"\n{'='*60}")
print("PREPARING TRAINING DATA")
print(f"{'='*60}")

prep = LoRADataPrep()
prep.prepare_all_authors()

# Step 2: Train LoRA adapters
print(f"\n{'='*60}")
print(f"TRAINING LORA ADAPTERS ({model_size})")
print(f"{'='*60}")

trainer = LoRATrainer(model_size=model_size)
trainer.train_all_authors(device=device)

print(f"\n{'='*60}")
print("Done! All LoRA adapters trained.")
print(f"{'='*60}")
