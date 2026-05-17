import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from stoic_llm.config import MODEL_NAME, DEVICE


class ModelLoader:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer"""
        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=self.device
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✓ Model loaded on {self.device}")

        return self.model, self.tokenizer
