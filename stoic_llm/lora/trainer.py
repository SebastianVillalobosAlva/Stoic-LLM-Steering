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
from pathlib import Path


class LoRATrainer:
    def __init__(
        self,
        model_name="meta-llama/Llama-3.2-1B",
        output_dir="./lora_models",
        data_dir="data/lora_training",
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load base model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_lora_config(self):
        """Configure LoRA parameters"""
        return LoraConfig(
            r=8,  # rank - higher = more parameters
            lora_alpha=32,  # scaling factor
            target_modules=["q_proj", "v_proj"],  # which layers to adapt
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def train_author(self, author_name, epochs=3, batch_size=2, learning_rate=2e-4):
        """Train LoRA adapter for one author"""
        print(f"\n🏛️ Training LoRA for {author_name}...")

        # Load fresh model for each author
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="cpu", torch_dtype=torch.float32
        )

        # Add LoRA adapters
        lora_config = self._get_lora_config()
        model = get_peft_model(model, lora_config)

        print("Trainable parameters:")
        model.print_trainable_parameters()

        # Load training data
        data_file = self.data_dir / f"{author_name}_train.jsonl"
        dataset = load_dataset("json", data_files=str(data_file))

        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
            )

        tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

        # Data collator handles creating labels automatically
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False  # We're doing causal LM, not masked LM
        )

        # Training arguments
        author_output_dir = self.output_dir / author_name
        training_args = TrainingArguments(
            output_dir=str(author_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=50,
            logging_steps=10,
            learning_rate=learning_rate,
            save_total_limit=2,
            fp16=False,  # M4 uses MPS, not CUDA
            report_to="none",  # disable wandb
        )

        # Create trainer with data collator
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator,
        )

        # Train!
        print(f"Starting training for {author_name}...")
        trainer.train()

        # Save adapter
        model.save_pretrained(str(author_output_dir))
        self.tokenizer.save_pretrained(str(author_output_dir))

        print(f"✅ LoRA adapter saved to {author_output_dir}")

    def train_all_authors(self):
        """Train LoRA adapters for all three philosophers"""
        authors = ["marcus_aurelius", "seneca", "epictetus"]
        for author in authors:
            self.train_author(author)
            print(f"\n{'='*70}\n")
