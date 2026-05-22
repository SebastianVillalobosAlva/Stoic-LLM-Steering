import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from stoic_llm.config import DEVICE

MODELS = {
    "1B": {
        "name": "meta-llama/Llama-3.2-1B",
        "dtype": torch.float32,
        "num_layers": 16,
        "hidden_dim": 2048,
    },
    "3B": {
        "name": "meta-llama/Llama-3.2-3B",
        "dtype": torch.float16,
        "num_layers": 28,
        "hidden_dim": 3072,
    },
}


class ModelLoader:
    def __init__(self, model_size="1B", device=DEVICE):
        if model_size not in MODELS:
            raise ValueError(
                f"Unknown model size '{model_size}'. Choose from: {list(MODELS.keys())}"
            )

        self.model_size = model_size
        self.model_name = MODELS[model_size]["name"]
        self.dtype = MODELS[model_size]["dtype"]
        self.num_layers = MODELS[model_size]["num_layers"]
        self.hidden_dim = MODELS[model_size]["hidden_dim"]
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer"""
        print(f"Loading {self.model_name} ({self.model_size}, {self.dtype})...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=self.dtype,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✓ Model loaded on {self.device} ({self.dtype})")

        return self.model, self.tokenizer
