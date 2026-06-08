import torch
from stoic_llm.config import VECTORS_DIR

d = torch.load(VECTORS_DIR / "marcus_aurelius_steering_3B.pt", weights_only=True)
print(type(d))
if isinstance(d, dict):
    for k, v in d.items():
        print(f"  layer {k!r}: type={type(v).__name__} shape={getattr(v,'shape',None)}")
else:
    print("  not a dict — old format:", getattr(d, "shape", None))
