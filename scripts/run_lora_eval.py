import sys
from stoic_llm.model import ModelLoader
from stoic_llm.lora.runner import LoRARunner
from stoic_llm.eval.judge import StoicJudge, summarize_eval
from stoic_llm.config import DEFAULT_PROMPTS
from typing import Any

model_size = sys.argv[1] if len(sys.argv) > 1 else "1B"

# Load base model for unsteered baseline
loader = ModelLoader(model_size)
model, tokenizer = loader.load()

# Load LoRA runner
lora = LoRARunner(model_size)
judge = StoicJudge()

authors = ["marcus_aurelius", "epictetus", "seneca"]

# Identical decoding on both sides so the only difference is the LoRA adapter
gen_kwargs: dict[str, Any] = dict(
    max_new_tokens=100,
    do_sample=False,
    repetition_penalty=1.3,
    no_repeat_ngram_size=3,
)

for author in authors:
    print(f"\n{'='*60}")
    print(f"{author} — LoRA ({model_size})")
    print(f"{'='*60}")

    # Generate LoRA outputs
    lora_outputs = [
        lora.generate(author_name=author, prompt=prompt, **gen_kwargs)
        for prompt in DEFAULT_PROMPTS
    ]

    # Generate unsteered baseline
    unsteered = []
    for prompt in DEFAULT_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, **gen_kwargs)
        unsteered.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    results = judge.evaluate_steering(
        prompts=DEFAULT_PROMPTS,
        steered_outputs=lora_outputs,
        unsteered_outputs=unsteered,
        author=author,
        metadata={
            "model_size": model_size,
            "method": "lora",
            "epochs": 3,
        },
    )

    print(summarize_eval(results))
    judge.save_results(results)
