from stoic_llm.model import ModelLoader
from stoic_llm.eval.sweep import SteeringSweep
from stoic_llm.eval.judge import StoicJudge
from stoic_llm.config import VECTORS_DIR

# load the 3B model + tokenizer
loader = ModelLoader("3B")
model, tokenizer = loader.load()

# build the sweep object for one philosopher
sweep = SteeringSweep(
    model=model,
    tokenizer=tokenizer,
    vector_path=str(VECTORS_DIR / "seneca_steering_3B.pt"),
    judge=StoicJudge(),
)

# ONE config, ONE seed, judge-variance mode → a single judge call
result = sweep.seed_eval(
    layer=20,
    coefficient=0.11,
    author="seneca",
    n_seeds=1,
    vary="judge",
)
print(result)
