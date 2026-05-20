"""
Standard Genetic Algorithm using PyGAD for traffic light optimization.

Optimizes the baseline TLS configuration using a classic GA (selection,
crossover, mutation) with parallel fitness evaluation via SUMO.

Output format matches custom_optimizer.py for easy comparison.
Runs using only the baseline initialization strategy.

Usage:  python -m src.algorithms.simple_genetic_algorithm
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
    LT_GOMEA_BASELINE_NOISE_STD,
    GENE_LOW, GENE_HIGH,
)
from src.sumo_setup.fitness_evaluation import (
    fitness_function,
    build_traffic_fitness_wrapper,
)

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


def run_single_ga(baseline_data, num_genes, baseline_vec, tls_to_genes, out_dir, rng):
    """Run a single PyGAD GA optimization experiment with baseline initialization."""
    global _wrapper, _fitness_history
    _fitness_history = []

    print(f"\n{'='*60}")
    print(f"Simple GA | Strategy: baseline | Pop: {PYGAD_POPULATION_SIZE}")
    print(f"{'='*60}")

    n_workers = NUM_PROCESSORS or os.cpu_count() or 1

    # Initialize population from baseline with Gaussian noise
    initial_pop = init_population("baseline", PYGAD_POPULATION_SIZE, num_genes, baseline_vec, LT_GOMEA_BASELINE_NOISE_STD, rng)
    
    gene_space = [{"low": GENE_LOW, "high": GENE_HIGH} for _ in range(num_genes)]

    ga_instance = pygad.GA(
        num_generations=PYGAD_NUM_GENERATIONS,
        num_parents_mating=PYGAD_NUM_PARENTS_MATING,
        fitness_func=pygad_fitness_func,
        initial_population=initial_pop.tolist(),
        gene_type=float,
        gene_space=gene_space,
        parent_selection_type="tournament",
        K_tournament=2,
        crossover_type="uniform",
        mutation_type="random",
        mutation_by_replacement=False,
        random_mutation_min_val=-5.0,
        random_mutation_max_val=5.0,
        mutation_percent_genes=PYGAD_MUTATION_PERCENT_GENES,
        keep_parents=PYGAD_KEEP_PARENTS,
        on_generation=_on_generation,
        save_best_solutions=True,
        parallel_processing=["process", n_workers],
    )

    t0 = time.time()
    ga_instance.run()
    elapsed = time.time() - t0

    solution, solution_fitness, _ = ga_instance.best_solution()
    best_cost = -float(solution_fitness)
    print(f"Done in {elapsed:.1f}s | Final best: {best_cost:.2f}")

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

    results = {
        "best_configuration": best_json,
        "best_fitness": best_cost,
        "fitness_history": _fitness_history,
        "time_s": round(elapsed, 2),
        "algorithm": "simple_ga",
        "strategy": "baseline",
        "pop_size": PYGAD_POPULATION_SIZE,
        "generations": PYGAD_NUM_GENERATIONS,
        "mutation_percent_genes": PYGAD_MUTATION_PERCENT_GENES,
        "num_parents_mating": PYGAD_NUM_PARENTS_MATING,
        "keep_parents": PYGAD_KEEP_PARENTS,
        "parent_selection_type": "tournament",
        "K_tournament": 2,
        "crossover_type": "uniform",
        "num_genes": num_genes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_file = out_dir / "simple_ga_baseline.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved → {out_file}")

    return best_cost, elapsed


def run_experiment():
    """Run a single GA experiment using baseline initialization."""
    global _wrapper
    
    with open(BASELINE_TRAFFIC_DATA, "r") as f:
        baseline_data = json.load(f)

    _wrapper, num_genes, _, _, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline_data, fitness_function=fitness_function,
    )

    tls_to_genes, _, baseline_vec = build_gene_map(baseline_data)

    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "src" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    try:
        best_cost, elapsed = run_single_ga(
            baseline_data, num_genes, baseline_vec, tls_to_genes, out_dir, rng
        )
        print(f"\n{'='*60}")
        print(f"Result | Best: {best_cost:.2f} | Time: {elapsed:.1f}s")
        print(f"{'='*60}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    run_experiment()
