import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from stoic_llm.model import MODELS
from stoic_llm.config import MODELS_DIR, LORA_TRAINING_DIR, DEVICE


class LoRATrainer:
    def __init__(
        self, model_size="1B", output_dir=MODELS_DIR, data_dir=LORA_TRAINING_DIR
    ):
        cfg = MODELS[model_size]
        self.model_name = cfg["name"]
        self.model_dtype = cfg["dtype"]
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

        print(f"Loading model: {self.model_name} ({model_size})")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_lora_config(self):
        """Configure LoRA parameters"""
        return LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def train_author(
        self, author_name, epochs=3, batch_size=2, learning_rate=2e-4, device="cpu"
    ):
        """Train LoRA adapter for one author"""
        print(f"\n🏛️ Training LoRA for {author_name} on {device}...")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.model_dtype,
        ).to(device)

        lora_config = self._get_lora_config()
        model = get_peft_model(model, lora_config)

        print("Trainable parameters:")
        model.print_trainable_parameters()

        data_file = self.data_dir / f"{author_name}_train.jsonl"
        dataset = load_dataset("json", data_files=str(data_file))

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
            )

        tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        author_output_dir = self.output_dir / author_name
        training_args = TrainingArguments(
            output_dir=str(author_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=50,
            logging_steps=10,
            learning_rate=learning_rate,
            save_total_limit=2,
            fp16=False,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator,
        )

        print(f"Starting training for {author_name}...")
        trainer.train()

        model.save_pretrained(str(author_output_dir))
        self.tokenizer.save_pretrained(str(author_output_dir))

        print(f"✅ LoRA adapter saved to {author_output_dir}")

    def train_all_authors(self, device="cpu"):
        for author in ["marcus_aurelius", "seneca", "epictetus"]:
            self.train_author(author, device=device)
            print(f"\n{'='*70}\n")
