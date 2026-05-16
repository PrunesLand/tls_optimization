"""
Random Search — evaluates random traffic light configs in parallel.

Generates NUM_SOLUTIONS random solutions, evaluates each NUM_REPEATS times,
and saves per-evaluation results plus the overall best result at the end.

Usage:  python -m src.pygad.random_search
"""

import json, sys, time, os, copy
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import BASELINE_TRAFFIC_DATA, NUM_PROCESSORS
from src.genetic_algorithm.fitness_evaluation import fitness_function as _traffic_fitness
from src.decomposition.DG2_grouping import build_traffic_fitness_wrapper

NUM_SOLUTIONS = 100
NUM_REPEATS = 1

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

    # Store solutions so we can reconstruct the best one later
    solutions = []
    for i in range(NUM_SOLUTIONS):
        solutions.append(rng.uniform(lb, ub))

    # Generate all tasks up front
    tasks = []
    for i, solution in enumerate(solutions):
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

    # ── Find the best result ────────────────────────────────────────────
    best_entry = min(results, key=lambda x: x["fitness"])
    best_sol_idx = best_entry["solution"]
    best_fitness = best_entry["fitness"]
    best_solution = solutions[best_sol_idx]

    print(f"\nBest: Solution {best_sol_idx + 1} | Fitness: {best_fitness:.2f}")

    # Reconstruct best configuration JSON
    tls_to_genes = {}
    gene_idx = 0
    for tls_id in sorted(baseline_data["tls_data"]):
        phases = sorted(baseline_data["tls_data"][tls_id])
        tls_to_genes[tls_id] = (gene_idx, gene_idx + len(phases))
        gene_idx += len(phases)

    best_json = copy.deepcopy(baseline_data)
    for tls_id in sorted(best_json["tls_data"]):
        if tls_id not in tls_to_genes:
            continue
        s, e = tls_to_genes[tls_id]
        raw = best_solution[s:e]
        keys = sorted(best_json["tls_data"][tls_id])
        num_phases = len(keys)

        total = sum(raw)
        if total <= 0:
            dur = [90 // num_phases] * num_phases
            dur[-1] += 90 - sum(dur)
        else:
            dur = [max(1, int(round(d * 90 / total))) for d in raw]
            diff = 90 - sum(dur)
            if diff:
                dur[int(np.argmax(dur))] += diff

        for i, pk in enumerate(keys):
            best_json["tls_data"][tls_id][pk]["duration"] = int(dur[i])

    best_json["composite_cost"] = best_fitness

    # ── Save ────────────────────────────────────────────────────────────
    output = {
        "evaluations": results,
        "best_configuration": best_json,
        "best_fitness": best_fitness,
        "best_solution_index": best_sol_idx,
        "total_time_s": round(total_time, 2),
        "num_solutions": NUM_SOLUTIONS,
        "num_repeats": NUM_REPEATS,
        "total_evaluations": len(results),
        "algorithm": "random_search",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = Path("src/outputs/random_search_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Saved {len(results)} evaluations → {out_path}")

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("─" * 42)
    print(f"{'Best fitness':<25} {best_fitness:>15.2f}")
    print(f"{'Best solution index':<25} {best_sol_idx:>15}")
    print(f"{'Total evaluations':<25} {len(results):>15}")
    print(f"{'Total time (s)':<25} {total_time:>15.2f}")
