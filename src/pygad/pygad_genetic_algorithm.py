"""
Standard Genetic Algorithm using PyGAD for traffic light optimization.

Optimizes the baseline TLS configuration using a classic GA (selection,
crossover, mutation) with parallel fitness evaluation via SUMO.

Output format matches lt_gomea_optimizer.py for easy comparison.

Usage:  python -m src.pygad.pygad_genetic_algorithm
"""

import pygad
import numpy as np
import json
import copy
import time
import os
import sys
from pathlib import Path

# Add project root to sys.path to import config and other modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (
    PYGAD_POPULATION_SIZE,
    PYGAD_NUM_GENERATIONS,
    PYGAD_MUTATION_PERCENT_GENES,
    PYGAD_NUM_PARENTS_MATING,
    PYGAD_KEEP_PARENTS,
    NUM_PROCESSORS,
    BASELINE_TRAFFIC_DATA,
)
from src.genetic_algorithm.fitness_evaluation import fitness_function
from src.decomposition.DG2_grouping import build_traffic_fitness_wrapper

# ── Module-level state (needed for picklable fitness func) ──────────
_wrapper = None
_fitness_history = []


def pygad_fitness_func(ga_instance, solution, solution_idx):
    """PyGAD fitness callback — must be module-level for pickling."""
    try:
        cost = float(_wrapper(solution))
        return -cost
    except Exception as e:
        print(f"Error evaluating fitness: {e}")
        return -9999999.0


def _on_generation(ga_instance):
    """PyGAD generation callback — must be module-level for pickling."""
    gen = ga_instance.generations_completed
    _, solution_fitness, _ = ga_instance.best_solution()
    best_cost = -float(solution_fitness)
    pop_fitness = ga_instance.last_generation_fitness
    mean_cost = -float(np.mean(pop_fitness))
    _fitness_history.append({
        "gen": gen, "best": best_cost, "mean": mean_cost,
    })
    print(f"Gen {gen:3d} | Best: {best_cost:.2f} | Mean: {mean_cost:.2f}")


def run_genetic_algorithm():
    """Run a standard GA on the traffic light baseline and save results."""
    global _wrapper, _fitness_history
    _fitness_history = []

    # ── Load baseline ───────────────────────────────────────────────────
    with open(BASELINE_TRAFFIC_DATA, "r") as f:
        baseline_data = json.load(f)

    _wrapper, num_genes, x_lower, x_upper, labels = build_traffic_fitness_wrapper(
        baseline_data=baseline_data, fitness_function=fitness_function,
    )

    # Build gene map for JSON reconstruction
    tls_to_genes = {}
    idx = 0
    for tls_id in sorted(baseline_data["tls_data"]):
        phases = sorted(baseline_data["tls_data"][tls_id])
        tls_to_genes[tls_id] = (idx, idx + len(phases))
        idx += len(phases)

    n_workers = NUM_PROCESSORS or os.cpu_count() or 1

    print(f"Number of genes (phases): {num_genes}")
    print(f"Population size: {PYGAD_POPULATION_SIZE}")
    print(f"Generations: {PYGAD_NUM_GENERATIONS}")
    print(f"Parallel workers: {n_workers}")

    # ── Configure & run ─────────────────────────────────────────────────
    gene_space = [{"low": float(lo), "high": float(hi)}
                  for lo, hi in zip(x_lower, x_upper)]

    ga_instance = pygad.GA(
        num_generations=PYGAD_NUM_GENERATIONS,
        num_parents_mating=PYGAD_NUM_PARENTS_MATING,
        fitness_func=pygad_fitness_func,
        sol_per_pop=PYGAD_POPULATION_SIZE,
        num_genes=num_genes,
        gene_type=float,
        gene_space=gene_space,
        mutation_percent_genes=PYGAD_MUTATION_PERCENT_GENES,
        keep_parents=PYGAD_KEEP_PARENTS,
        on_generation=_on_generation,
        save_best_solutions=True,
        parallel_processing=["process", n_workers],
    )

    print("Starting Genetic Algorithm...")
    t0 = time.time()
    ga_instance.run()
    elapsed = time.time() - t0
    print(f"GA completed in {elapsed:.2f}s")

    # ── Extract best solution ───────────────────────────────────────────
    solution, solution_fitness, _ = ga_instance.best_solution()
    best_cost = -float(solution_fitness)
    print(f"Best solution cost: {best_cost:.2f}")

    # Reconstruct the full TLS JSON from the flat gene vector
    best_json = copy.deepcopy(baseline_data)
    for tls_id in sorted(best_json["tls_data"]):
        if tls_id not in tls_to_genes:
            continue
        s, e = tls_to_genes[tls_id]
        raw = solution[s:e]
        keys = sorted(best_json["tls_data"][tls_id])
        n = len(keys)

        total = sum(raw)
        if total <= 0:
            dur = [90 // n] * n
            dur[-1] += 90 - sum(dur)
        else:
            dur = [max(1, int(round(d * 90 / total))) for d in raw]
            diff = 90 - sum(dur)
            if diff:
                dur[int(np.argmax(dur))] += diff

        for i, pk in enumerate(keys):
            best_json["tls_data"][tls_id][pk]["duration"] = int(dur[i])

    best_json["composite_cost"] = best_cost

    # ── Save results ────────────────────────────────────────────────────
    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "src" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "best_configuration": best_json,
        "best_fitness": best_cost,
        "fitness_history": _fitness_history,
        "time_s": round(elapsed, 2),
        "algorithm": "standard_ga",
        "pop_size": PYGAD_POPULATION_SIZE,
        "generations": PYGAD_NUM_GENERATIONS,
        "mutation_percent_genes": PYGAD_MUTATION_PERCENT_GENES,
        "num_parents_mating": PYGAD_NUM_PARENTS_MATING,
        "keep_parents": PYGAD_KEEP_PARENTS,
        "num_genes": num_genes,
        "seed": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_file = out_dir / "pygad_best_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {out_file}")

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("─" * 42)
    print(f"{'Best fitness':<25} {best_cost:>15.2f}")
    print(f"{'Generations':<25} {PYGAD_NUM_GENERATIONS:>15}")
    print(f"{'Population size':<25} {PYGAD_POPULATION_SIZE:>15}")
    print(f"{'Time (s)':<25} {elapsed:>15.2f}")


if __name__ == "__main__":
    run_genetic_algorithm()