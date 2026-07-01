# ============================================================
# Stoic LLM — LoRA training on CLEAN data (Colab, T4)
# ============================================================
# Retrains LoRA adapters for the three philosophers on the
# Exp-9 cleaned Stoic text, so the LoRA condition is comparable
# to the clean CAA vectors in the dilemma eval.
#
# RECIPE IS HELD IDENTICAL TO EXP 4 (r=8, alpha=32, q_proj+v_proj,
# 3 epochs). The ONLY thing that changes vs the old adapters is the
# training text (clean vs pre-clean) — same logic as Exp 9 (hold
# everything fixed, change only the data). Do not retune here, or
# you confound "clean data" with "new hyperparams."
#
# Inputs  (upload to Drive): clean chunked text per philosopher
# Outputs (saved to Drive):  LoRA adapters, one dir per philosopher
#
# After this runs: download adapters to the Mac, merge_and_unload,
# run the SAME dilemma_eval harness (no hook — weights already modified).
# ============================================================


# ===== CELL 1: install =====
# Pin versions close to your local repo (transformers>=4.30, peft).
!pip -q install "transformers>=4.44" "peft>=0.11" "datasets>=2.19" "accelerate>=0.30"


# ===== CELL 2: HF auth (Llama-3.2-3B is gated) =====
# Put your token in Colab Secrets (key icon, left sidebar) as HF_TOKEN,
# or this falls back to an interactive prompt.
from huggingface_hub import login
try:
    from google.colab import userdata
    login(token=userdata.get("HF_TOKEN"))
except Exception:
    login()


# ===== CELL 3: mount Drive =====
from google.colab import drive
drive.mount("/content/drive")


# ===== CELL 4: config =====
from pathlib import Path
import torch

MODEL_NAME = "meta-llama/Llama-3.2-3B"

# --- paths: adjust DRIVE_ROOT to wherever you put the repo data ---
DRIVE_ROOT = Path("/content/drive/MyDrive/stoic-llm")
# clean chunked text per philosopher. Expected: a JSON file that is
# either a list[str] of chunks, or list[dict] with a "text" field.
CHUNKED_DIR = DRIVE_ROOT / "data" / "chunked"
ADAPTER_OUT = DRIVE_ROOT / "models"           # adapters saved here
ADAPTER_OUT.mkdir(parents=True, exist_ok=True)

# clean chunked filenames per author (edit to match what you uploaded)
AUTHOR_FILES = {
    "marcus":    "marcus_clean_chunks.json",
    "seneca":    "seneca_clean_chunks.json",
    "epictetus": "epictetus_clean_chunks.json",
}

# --- LoRA recipe: IDENTICAL to Exp 4. Do not change for this run. ---
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]

# --- training args ---
EPOCHS = 3
LR = 2e-4
MAX_LEN = 512
BATCH = 2                 # T4-safe with grad checkpointing; drop to 1 if OOM
GRAD_ACCUM = 8            # effective batch = BATCH * GRAD_ACCUM = 16
SEED = 0


# ===== CELL 5: load base model + tokenizer =====
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token   # same as ModelLoader

def load_base():
    """Fresh base model each author so adapters never stack."""
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
    )
    m.config.use_cache = False               # required for grad checkpointing
    return m


# ===== CELL 6: data prep (Stoic text only — continued-pretraining) =====
# NOTE ON THE OBJECTIVE CONFOUND (from the build plan): LoRA here is
# plain causal-LM on Stoic text only; it ignores the neutral halves,
# while CAA uses the contrastive *direction*. So a CAA-vs-LoRA gap is
# partly method and partly objective. Keep this recipe (it matches the
# Exp 5/6 mechanistic analysis); a contrastive-objective LoRA is a
# SEPARATE experiment, not a tweak to this one.
import json
from datasets import Dataset

def load_chunks(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    if data and isinstance(data[0], dict):
        return [d["text"] for d in data if d.get("text", "").strip()]
    return [t for t in data if isinstance(t, str) and t.strip()]

def build_dataset(chunks: list[str]) -> Dataset:
    def tok(batch):
        out = tokenizer(
            batch["text"], truncation=True, max_length=MAX_LEN, padding="max_length",
        )
        out["labels"] = [ids.copy() for ids in out["input_ids"]]
        return out
    ds = Dataset.from_dict({"text": chunks})
    return ds.map(tok, batched=True, remove_columns=["text"])


# ===== CELL 7: train ONE author (single run — double-training bug fixed) =====
# The old LoRATrainer.train_all_authors called train_author twice per
# author and saved the second (CPU) run. Here each author is trained
# exactly ONCE, on GPU, and saved.
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, set_seed

def train_author(author: str, chunks_file: str) -> Path:
    set_seed(SEED)
    print(f"\n=== {author}: {chunks_file} ===")

    chunks = load_chunks(CHUNKED_DIR / chunks_file)
    print(f"  {len(chunks)} clean chunks")
    ds = build_dataset(chunks)

    model = load_base()
    model.enable_input_require_grads()        # needed: grad ckpt + PEFT
    model.gradient_checkpointing_enable()
    peft_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    out_dir = ADAPTER_OUT / f"lora_{author}_clean"
    args = TrainingArguments(
        output_dir=str(out_dir / "_ckpt"),
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=True,
        logging_steps=10,
        save_strategy="no",                   # we save the adapter manually below
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()

    model.save_pretrained(str(out_dir))       # saves ONLY the adapter (small)
    tokenizer.save_pretrained(str(out_dir))
    print(f"  saved -> {out_dir}")

    del model, trainer
    torch.cuda.empty_cache()
    return out_dir


# ===== CELL 8: train all three =====
adapter_dirs = {}
for author, fname in AUTHOR_FILES.items():
    adapter_dirs[author] = train_author(author, fname)

print("\nDone. Adapters:")
for a, d in adapter_dirs.items():
    print(f"  {a}: {d}")


# ===== CELL 9: sanity check (optional) =====
# Quick eyeball that the adapter shifts register before you download.
from peft import PeftModel

def quick_gen(author: str, prompt="When facing difficulty, one should"):
    base = load_base()
    merged = PeftModel.from_pretrained(base, str(adapter_dirs[author])).merge_and_unload()
    ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = merged.generate(**ids, max_new_tokens=40, do_sample=False,
                          repetition_penalty=1.3, no_repeat_ngram_size=3)
    print(f"[{author}] {tokenizer.decode(out[0], skip_special_tokens=True)}")
    del base, merged
    torch.cuda.empty_cache()

for a in AUTHOR_FILES:
    quick_gen(a)


# ============================================================
# NEXT (local, on the Mac):
#   1. Download the lora_{author}_clean dirs from Drive.
#   2. In the dilemma harness, load base + PeftModel.from_pretrained(...)
#      .merge_and_unload(), then call eval_condition() with NO vector
#      (vector=None) — the weights already carry the steering, so no hook.
#      Compare against the SAME base-model baseline you already computed.
#   3. Read the table the same way: does LoRA move the DECISION where
#      CAA didn't? (Exp 6 predicts it might — output-layer rewiring.)
# ============================================================
