import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import MODEL_NAME, DEVICE


class ModelLoader:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.device = DEVICE
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer"""
        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=self.device
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Model memory: ~{torch.mps.current_allocated_memory() / 1e9:.2f} GB")

        return self.model, self.tokenizer
