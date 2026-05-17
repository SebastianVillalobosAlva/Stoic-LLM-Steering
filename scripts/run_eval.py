from stoic_llm.model import ModelLoader
from stoic_llm.eval.sweep import SteeringSweep, summarize_sweep

loader = ModelLoader()
model, tokenizer = loader.load()

sweep = SteeringSweep(
    model=model,
    tokenizer=tokenizer,
    vector_path="data/steering_vectors/seneca_steering.pt",
)

# Full two-stage sweep
results = sweep.full_sweep(author="seneca")
print(summarize_sweep(results))
sweep.save_results(results)
