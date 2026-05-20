"""
Random Search — evaluates randomly/strategically sampled traffic light configs in parallel.

Generates NUM_SOLUTIONS solutions based on a strategy, evaluates each NUM_REPEATS times,
and saves the overall best result.
Runs on 9 scenarios (euclidian, shortest, fastest X random, baseline, mixed)
to provide a 1:1 comparison base.

Usage:  python -m src.pygad.random_search
"""

import json, sys, time, os, copy
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import BASELINE_TRAFFIC_DATA, NUM_PROCESSORS, LT_GOMEA_BASELINE_NOISE_STD, MAX_EVALS
from src.genetic_algorithm.fitness_evaluation import fitness_function as _traffic_fitness
from src.decomposition.DG2_grouping import build_traffic_fitness_wrapper

NUM_SOLUTIONS = MAX_EVALS
NUM_REPEATS = 1
GENE_LOW, GENE_HIGH = 5.0, 85.0

def _eval_worker(args):
    """Picklable worker for parallel execution."""
    wrapper, sol_idx, rep, solution = args
    t0 = time.time()
    fitness = float(wrapper(solution))
    elapsed = time.time() - t0
    return sol_idx, rep, fitness, elapsed


def init_population(strategy, n, num_genes, baseline_vec, noise_std, rng):
    """Create initial population: 'random', 'baseline', or 'mixed'."""
    if strategy == "random":
        return rng.uniform(GENE_LOW, GENE_HIGH, (n, num_genes))

    elif strategy == "baseline":
        pop = np.tile(baseline_vec, (n, 1))
        pop += rng.normal(0, noise_std, pop.shape) * pop
        return np.clip(pop, GENE_LOW, GENE_HIGH)

    elif strategy == "mixed":
        half = n // 2
        rand = rng.uniform(GENE_LOW, GENE_HIGH, (half, num_genes))
        base = np.tile(baseline_vec, (n - half, 1))
        base += rng.normal(0, noise_std, base.shape) * base
        return np.vstack([rand, np.clip(base, GENE_LOW, GENE_HIGH)])

    raise ValueError(f"Unknown strategy: {strategy}")


def build_gene_map(baseline_data):
    """Builds a gene map and baseline vector for population initialization."""
    tls_to_genes = {}
    idx = 0
    baseline = []

    for tls_id in sorted(baseline_data["tls_data"]):
        phases = sorted(baseline_data["tls_data"][tls_id])
        tls_to_genes[tls_id] = (idx, idx + len(phases))
        for pk in phases:
            baseline.append(float(baseline_data["tls_data"][tls_id][pk]["duration"]))
        idx += len(phases)

    return tls_to_genes, idx, np.array(baseline)


def run_single_search(tree_name, strategy, baseline_data, wrapper, num_genes, baseline_vec, tls_to_genes, out_dir, rng):
    """Run a single Random Search experiment block."""
    print(f"\n{'='*60}")
    print(f"Random Search | Tree (Label): {tree_name} | Strategy: {strategy} | Solutions: {NUM_SOLUTIONS}")
    print(f"{'='*60}")

    solutions = init_population(strategy, NUM_SOLUTIONS, num_genes, baseline_vec, LT_GOMEA_BASELINE_NOISE_STD, rng)
    
    tasks = []
    for i, solution in enumerate(solutions):
        for r in range(NUM_REPEATS):
            tasks.append((wrapper, i, r + 1, solution))

    n_workers = NUM_PROCESSORS or os.cpu_count() or 1
    t_start = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_eval_worker, task) for task in tasks]
        
        for idx, future in enumerate(as_completed(futures)):
            sol_idx, rep, fitness, elapsed = future.result()
            results.append({
                "solution": sol_idx,
                "repeat": rep,
                "fitness": fitness,
                "time_s": round(elapsed, 4),
            })
            
    total_time = time.time() - t_start
    results.sort(key=lambda x: (x["solution"], x["repeat"]))

    best_entry = min(results, key=lambda x: x["fitness"])
    best_sol_idx = best_entry["solution"]
    best_fitness = best_entry["fitness"]
    best_solution = solutions[best_sol_idx]

    print(f"Done in {total_time:.1f}s | Best: Solution {best_sol_idx + 1} | Fitness: {best_fitness:.2f}")

    # Reconstruct best configuration JSON
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

    label = f"{tree_name}_{strategy}"
    output = {
        "evaluations": results,
        "best_configuration": best_json,
        "best_fitness": best_fitness,
        "best_solution_index": int(best_sol_idx),
        "total_time_s": round(total_time, 2),
        "num_solutions": NUM_SOLUTIONS,
        "num_repeats": NUM_REPEATS,
        "total_evaluations": len(results),
        "algorithm": "random_search",
        "tree": tree_name,
        "strategy": strategy,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = out_dir / f"random_search_{label}.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    return best_fitness, total_time


def run_all_experiments():
    """Run all 9 experiments to match LT-GOMEA structure."""
    with open(BASELINE_TRAFFIC_DATA) as fh:
        baseline_data = json.load(fh)

    wrapper, num_genes, _, _, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline_data,
        fitness_function=_traffic_fitness,
    )
    
    tls_to_genes, _, baseline_vec = build_gene_map(baseline_data)

    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "src" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = ["shortest", "euclidian", "fastest"]
    strategies = ["random", "baseline", "mixed"]
    summary = {}
    
    rng = np.random.default_rng(42)

    for tree_name in trees:
        for strat in strategies:
            label = f"{tree_name}_{strat}"
            try:
                best_cost, elapsed = run_single_search(
                    tree_name, strat, baseline_data, wrapper, num_genes, baseline_vec, tls_to_genes, out_dir, rng
                )
                summary[label] = {"best": best_cost, "time_s": elapsed}
            except Exception as e:
                print(f"ERROR [{label}]: {e}")
                import traceback; traceback.print_exc()
                summary[label] = {"error": str(e)}

    # Print results table
    print(f"\n{'Tree Label':<15} {'Strategy':<10} {'Best':>12} {'Time':>8}")
    print("─" * 47)
    for label, info in summary.items():
        t, s = label.rsplit("_", 1)
        if "error" in info:
            print(f"{t:<15} {s:<10} {'ERROR':>12}")
        else:
            print(f"{t:<15} {s:<10} {info['best']:>12.2f} {info['time_s']:>7.1f}s")


if __name__ == "__main__":
    run_all_experiments()
