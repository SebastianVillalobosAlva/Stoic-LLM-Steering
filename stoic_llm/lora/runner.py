import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from stoic_llm.model import MODELS
from stoic_llm.config import MODELS_DIR, DEVICE


class LoRARunner:
    def __init__(self, model_size="1B", lora_models_dir=MODELS_DIR):
        cfg = MODELS[model_size]
        self.base_model_name = cfg["name"]
        self.lora_models_dir = lora_models_dir / model_size

        print(f"Loading base model ({model_size})...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map=DEVICE,
            torch_dtype=cfg["dtype"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.current_model = None
        self.current_author = None

    def load_author_model(self, author_name):
        """Load LoRA adapter for specific author"""
        if self.current_author == author_name:
            return

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
