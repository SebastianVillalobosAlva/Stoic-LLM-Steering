import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path


class LoRARunner:
    def __init__(
        self, base_model_name="meta-llama/Llama-3.2-1B", lora_models_dir="./lora_models"
    ):
        self.base_model_name = base_model_name
        self.lora_models_dir = Path(lora_models_dir)

        # Load base model and tokenizer once
        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="cpu", torch_dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.current_model = None
        self.current_author = None

    def load_author_model(self, author_name):
        """Load LoRA adapter for specific author"""
        if self.current_author == author_name:
            return  # Already loaded

        print(f"Loading LoRA adapter for {author_name}...")
        lora_path = self.lora_models_dir / author_name

        self.current_model = PeftModel.from_pretrained(self.base_model, str(lora_path))
        self.current_author = author_name
        print(f"✅ Loaded {author_name} adapter")

    def generate(
        self, author_name, prompt, max_tokens=150, temperature=0.7, do_sample=True
    ):
        """Generate text using LoRA model"""
        self.load_author_model(author_name)

        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.current_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
