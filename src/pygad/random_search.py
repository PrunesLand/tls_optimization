"""
Random Search — evaluates 100 random traffic light configs, 10 times each.
"""

import json, sys, time
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import BASELINE_TRAFFIC_DATA
from src.genetic_algorithm.fitness_evaluation import fitness_function as _traffic_fitness
from src.pygad.DG2_grouping import build_traffic_fitness_wrapper

NUM_SOLUTIONS = 100
NUM_REPEATS = 10

if __name__ == "__main__":
    with open(BASELINE_TRAFFIC_DATA) as fh:
        baseline_data = json.load(fh)

    wrapper, n, lb, ub, labels = build_traffic_fitness_wrapper(
        baseline_data=baseline_data,
        fitness_function=_traffic_fitness,
    )

    rng = np.random.default_rng(42)
    results = []

    for i in range(NUM_SOLUTIONS):
        solution = rng.uniform(lb, ub)

        for r in range(NUM_REPEATS):
            t0 = time.time()
            fitness = float(wrapper(solution))
            elapsed = time.time() - t0

            print(f"Solution {i+1:3d}/{NUM_SOLUTIONS} | "
                  f"Repeat {r+1:2d}/{NUM_REPEATS} | "
                  f"Fitness: {fitness:12.2f} | "
                  f"Time: {elapsed:.4f}s")

            results.append({
                "solution": i, "repeat": r + 1,
                "fitness": fitness, "time_s": round(elapsed, 4),
            })

    # Save
    out_path = Path("src/outputs/random_search_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {len(results)} evaluations -> {out_path}")
