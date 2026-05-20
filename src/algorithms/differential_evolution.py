"""
Differential Evolution using SciPy for traffic light optimization.

Optimizes the baseline TLS configuration using scipy.optimize.differential_evolution
with parallel fitness evaluation via SUMO.

Runs 3 experiments: random, baseline, and mixed initialization strategies.
Variables are constrained to whole numbers (integers).

Usage:  python -m src.algorithms.differential_evolution
"""

import numpy as np
import json
import copy
import time
import os
import sys
from pathlib import Path
from scipy.optimize import differential_evolution

# Add project root to sys.path to import config and other modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (
    PYGAD_POPULATION_SIZE,
    MAX_EVALS,
    NUM_PROCESSORS,
    BASELINE_TRAFFIC_DATA,
    GAUSSIAN_NOISE,
    GENE_LOW, GENE_HIGH,
)
from src.sumo_setup.fitness_evaluation import (
    fitness_function,
    build_traffic_fitness_wrapper,
)

# ── Module-level state ──────────────────────────────────────────────
_wrapper = None
_fitness_history = []
_num_evals = 0
_best_cost = float("inf")
_gen_costs = []


def _objective(x):
    """Objective function for scipy DE — minimises cost."""
    global _num_evals
    try:
        cost = float(_wrapper(x))
        _num_evals += 1
        return cost
    except Exception as e:
        print(f"Error evaluating fitness: {e}")
        _num_evals += 1
        return 9999999.0


def _de_callback(xk, convergence):
    """
    Called once per generation by scipy DE.
    Logs progress and stops early when the evaluation budget is exhausted.
    """
    global _best_cost
    gen = len(_fitness_history) + 1
    cost = float(_wrapper(xk))
    _num_evals  # don't increment — xk was already evaluated internally
    if cost < _best_cost:
        _best_cost = cost
    _fitness_history.append({
        "gen": gen, "best": _best_cost, "convergence": float(convergence),
    })
    print(
        f"Gen {gen:3d} | Best: {_best_cost:.2f} | "
        f"Convergence: {convergence:.4f} | Evals: {_num_evals}"
    )
    if _num_evals > MAX_EVALS:
        return True  # signal scipy to stop
    return False


# ── Population initialisation strategies ────────────────────────────

def init_population(strategy, n, num_genes, baseline_vec, noise_std,
                    bounds_lo, bounds_hi, rng):
    """Create initial population: 'random', 'baseline', or 'mixed'.

    Returns an (n, num_genes) array with values clipped to [bounds_lo, bounds_hi]
    and rounded to whole numbers.
    """
    if strategy == "random":
        pop = rng.uniform(bounds_lo, bounds_hi, (n, num_genes))

    elif strategy == "baseline":
        pop = np.tile(baseline_vec, (n, 1))
        pop += rng.normal(0, noise_std, pop.shape) * pop
        pop = np.clip(pop, bounds_lo, bounds_hi)

    elif strategy == "mixed":
        half = n // 2
        rand_part = rng.uniform(bounds_lo, bounds_hi, (half, num_genes))
        base_part = np.tile(baseline_vec, (n - half, 1))
        base_part += rng.normal(0, noise_std, base_part.shape) * base_part
        base_part = np.clip(base_part, bounds_lo, bounds_hi)
        pop = np.vstack([rand_part, base_part])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Round to whole numbers
    return np.round(pop).astype(float)


# ── Gene-map helper (identical to simple GA) ────────────────────────

def build_gene_map(baseline_data):
    """Builds a gene map and baseline vector for population initialisation."""
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


# ── Single DE run ───────────────────────────────────────────────────

def run_single_de(strategy, baseline_data, num_genes, baseline_vec,
                  tls_to_genes, bounds_lo, bounds_hi, out_dir, rng):
    """Run a single scipy.optimize.differential_evolution experiment."""
    global _wrapper, _fitness_history, _num_evals, _best_cost

    _fitness_history = []
    _num_evals = 0
    _best_cost = float("inf")

    print(f"\n{'='*60}")
    print(f"Differential Evolution | Strategy: {strategy} | Pop: {PYGAD_POPULATION_SIZE}")
    print(f"{'='*60}")

    n_workers = NUM_PROCESSORS or os.cpu_count() or 1

    # Per-gene bounds as a list of (lo, hi) tuples
    bounds = list(zip(bounds_lo.tolist(), bounds_hi.tolist()))

    # Build initial population
    initial_pop = init_population(
        strategy, PYGAD_POPULATION_SIZE, num_genes, baseline_vec,
        GAUSSIAN_NOISE, bounds_lo, bounds_hi, rng,
    )

    # Mark every variable as integer (whole numbers)
    integrality = np.ones(num_genes)

    t0 = time.time()
    result = differential_evolution(
        func=_objective,
        bounds=bounds,
        maxiter=MAX_EVALS,              # generous upper limit on generations
        maxfev=MAX_EVALS,               # hard cap on total function evaluations
        popsize=PYGAD_POPULATION_SIZE,   # absolute population size via init kwarg
        strategy="best1bin",
        mutation=(0.5, 1.0),             # dithered mutation factor F ∈ [0.5, 1.0]
        recombination=0.7,
        tol=0,                           # don't stop on convergence tolerance
        seed=rng,
        callback=_de_callback,
        init=initial_pop,                # custom initial population
        workers=n_workers,
        updating="deferred",            # required when workers > 1
        integrality=integrality,         # constrain variables to whole numbers
    )
    elapsed = time.time() - t0

    best_vec = result.x
    best_cost = float(result.fun)
    print(f"Done in {elapsed:.1f}s | Final best: {best_cost:.2f} | Evals: {_num_evals}")

    # Reconstruct the full TLS JSON from the flat gene vector
    best_json = copy.deepcopy(baseline_data)
    for tls_id in sorted(best_json["tls_data"]):
        if tls_id not in tls_to_genes:
            continue
        s, e = tls_to_genes[tls_id]
        raw = best_vec[s:e]
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
        "algorithm": "differential_evolution",
        "strategy": strategy,
        "pop_size": PYGAD_POPULATION_SIZE,
        "de_strategy": "best1bin",
        "mutation_range": [0.5, 1.0],
        "recombination": 0.7,
        "integrality": True,
        "num_genes": num_genes,
        "total_evals": _num_evals,
        "scipy_message": result.message,
        "scipy_success": result.success,
        "scipy_nit": int(result.nit),
        "scipy_nfev": int(result.nfev),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_file = out_dir / f"differential_evolution_{strategy}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved → {out_file}")

    return best_cost, elapsed


# ── Run all 3 experiments ───────────────────────────────────────────

def run_all_experiments():
    """Run 3 DE experiments: random, baseline, and mixed initialisation."""
    global _wrapper

    with open(BASELINE_TRAFFIC_DATA, "r") as f:
        baseline_data = json.load(f)

    _wrapper, num_genes, bounds_lo, bounds_hi, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline_data, fitness_function=fitness_function,
    )

    tls_to_genes, _, baseline_vec = build_gene_map(baseline_data)

    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "src" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = ["random", "baseline", "mixed"]
    summary = {}

    rng = np.random.default_rng(42)

    for strat in strategies:
        try:
            best_cost, elapsed = run_single_de(
                strat, baseline_data, num_genes, baseline_vec,
                tls_to_genes, bounds_lo, bounds_hi, out_dir, rng,
            )
            summary[strat] = {"best": best_cost, "time_s": elapsed}
        except Exception as e:
            print(f"ERROR [{strat}]: {e}")
            import traceback; traceback.print_exc()
            summary[strat] = {"error": str(e)}

    # Print results table
    print(f"\n{'Strategy':<10} {'Best':>12} {'Time':>8}")
    print("─" * 32)
    for strat, info in summary.items():
        if "error" in info:
            print(f"{strat:<10} {'ERROR':>12}")
        else:
            print(f"{strat:<10} {info['best']:>12.2f} {info['time_s']:>7.1f}s")


if __name__ == "__main__":
    run_all_experiments()
