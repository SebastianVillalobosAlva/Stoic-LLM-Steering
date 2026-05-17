import json

with open("results/sweeps/full_marcus_aurelius_20260517_014516.json") as f:
    results = json.load(f)

coeff_sweep = results["coefficient_sweep"]
best_result = [r for r in coeff_sweep["results"] if r["coefficient"] == 0.11][0]

print(f"Unsteered aggregate: {best_result['avg_unsteered']['aggregate']:.2f}")
print(f"Steered aggregate: {best_result['avg_steered']['aggregate']:.2f}")
print(f"Delta: {best_result['avg_deltas']['aggregate']:+.2f}")
