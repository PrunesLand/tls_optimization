"""
Random Search — evaluates 100 random traffic light configs, 10 times each, in parallel.
"""

import json, sys, time, os
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import BASELINE_TRAFFIC_DATA, NUM_PROCESSORS
from src.genetic_algorithm.fitness_evaluation import fitness_function as _traffic_fitness
from src.pygad.DG2_grouping import build_traffic_fitness_wrapper

NUM_SOLUTIONS = 100
NUM_REPEATS = 10

def _eval_worker(args):
    """Picklable worker for parallel execution."""
    wrapper, sol_idx, rep, solution = args
    t0 = time.time()
    fitness = float(wrapper(solution))
    elapsed = time.time() - t0
    return sol_idx, rep, fitness, elapsed

if __name__ == "__main__":
    with open(BASELINE_TRAFFIC_DATA) as fh:
        baseline_data = json.load(fh)

    wrapper, n, lb, ub, labels = build_traffic_fitness_wrapper(
        baseline_data=baseline_data,
        fitness_function=_traffic_fitness,
    )

    rng = np.random.default_rng(42)
    results = []

    # Generate all tasks up front
    tasks = []
    for i in range(NUM_SOLUTIONS):
        solution = rng.uniform(lb, ub)
        for r in range(NUM_REPEATS):
            tasks.append((wrapper, i, r + 1, solution))

    n_workers = NUM_PROCESSORS or os.cpu_count() or 1
    print(f"Starting parallel random search with {n_workers} workers...")
    print(f"Total evaluations: {len(tasks)}")

    t_start = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_eval_worker, task) for task in tasks]
        
        for idx, future in enumerate(as_completed(futures)):
            sol_idx, rep, fitness, elapsed = future.result()
            print(f"Completed {idx+1:4d}/{len(tasks)} | "
                  f"Solution {sol_idx+1:3d} Rep {rep:2d} | "
                  f"Fitness: {fitness:12.2f} | Time: {elapsed:.4f}s")
            
            results.append({
                "solution": sol_idx,
                "repeat": rep,
                "fitness": fitness,
                "time_s": round(elapsed, 4),
            })
            
    total_time = time.time() - t_start
    print(f"\nAll done in {total_time:.1f} seconds!")

    # Sort results by solution and repeat to keep it neat
    results.sort(key=lambda x: (x["solution"], x["repeat"]))

    # Save
    out_path = Path("src/outputs/random_search_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved {len(results)} evaluations -> {out_path}")
