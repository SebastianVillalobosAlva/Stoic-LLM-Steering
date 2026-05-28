import sys
from stoic_llm.model import ModelLoader
from stoic_llm.lora.runner import LoRARunner
from stoic_llm.eval.judge import StoicJudge, summarize_eval
from stoic_llm.config import DEFAULT_PROMPTS

model_size = sys.argv[1] if len(sys.argv) > 1 else "1B"

# Load base model for unsteered baseline
loader = ModelLoader(model_size)
model, tokenizer = loader.load()

# Load LoRA runner
lora = LoRARunner(model_size)
judge = StoicJudge()

authors = ["marcus_aurelius", "epictetus", "seneca"]

for author in authors:
    print(f"\n{'='*60}")
    print(f"{author} — LoRA ({model_size})")
    print(f"{'='*60}")

    # Generate LoRA outputs
    lora_outputs = []
    for prompt in DEFAULT_PROMPTS:
        output = lora.generate(
            author,
            prompt,
            max_tokens=100,
            temperature=0.7,
            do_sample=True,
        )
        lora_outputs.append(output)

    # Generate unsteered baseline
    unsteered = []
    for prompt in DEFAULT_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
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
