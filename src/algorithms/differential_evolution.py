"""
Differential Evolution (SHADE) using EvoX for traffic light optimization.

Optimizes the baseline TLS configuration using the SHADE (Success-History
based Adaptive Differential Evolution) algorithm from the EvoX library
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
from multiprocessing import Pool

import torch
from evox.algorithms import SHADE
from evox.workflows import StdWorkflow, EvalMonitor
from evox.core import Problem

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
_num_evals = 0


def _evaluate_single(vec):
    """Evaluate a single solution vector (module-level for pickling)."""
    global _num_evals
    try:
        cost = float(_wrapper(vec))
        _num_evals += 1
        return cost
    except Exception as e:
        print(f"Error evaluating fitness: {e}")
        _num_evals += 1
        return 9999999.0


# ── Custom EvoX Problem wrapping the SUMO fitness function ─────────

class TLSProblem(Problem):
    """EvoX-compatible problem that evaluates TLS solutions via SUMO.

    Converts the PyTorch population tensor to NumPy, rounds to whole
    numbers, evaluates each individual through the SUMO wrapper (optionally
    in parallel), and returns a fitness tensor.
    """

    def __init__(self, wrapper, n_workers=1):
        super().__init__()
        self.wrapper = wrapper
        self.n_workers = n_workers

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        """Evaluate the entire population.

        Args:
            pop: Tensor of shape (pop_size, dim).

        Returns:
            Tensor of shape (pop_size,) with fitness values (costs).
        """
        global _num_evals

        # Round to whole numbers and convert to numpy
        pop_np = torch.round(pop).cpu().numpy()

        if self.n_workers > 1:
            with Pool(processes=self.n_workers) as pool:
                costs = pool.map(_evaluate_single, [row for row in pop_np])
        else:
            costs = [_evaluate_single(row) for row in pop_np]

        return torch.tensor(costs, dtype=pop.dtype, device=pop.device)


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


# ── Single DE-SHADE run ─────────────────────────────────────────────

def run_single_de(strategy, baseline_data, num_genes, baseline_vec,
                  tls_to_genes, bounds_lo, bounds_hi, out_dir, rng):
    """Run a single EvoX SHADE experiment with the given initialisation strategy."""
    global _wrapper, _num_evals

    _num_evals = 0
    fitness_history = []

    print(f"\n{'='*60}")
    print(f"SHADE (EvoX) | Strategy: {strategy} | Pop: {PYGAD_POPULATION_SIZE}")
    print(f"{'='*60}")

    n_workers = NUM_PROCESSORS or os.cpu_count() or 1

    # Build initial population using our strategy
    initial_pop_np = init_population(
        strategy, PYGAD_POPULATION_SIZE, num_genes, baseline_vec,
        GAUSSIAN_NOISE, bounds_lo, bounds_hi, rng,
    )

    # Convert bounds and initial population to PyTorch tensors
    lb = torch.tensor(bounds_lo, dtype=torch.float64)
    ub = torch.tensor(bounds_hi, dtype=torch.float64)

    # Initialise SHADE algorithm
    algorithm = SHADE(
        pop_size=PYGAD_POPULATION_SIZE,
        lb=lb,
        ub=ub,
    )

    # Create the custom problem and monitor
    problem = TLSProblem(wrapper=_wrapper, n_workers=n_workers)
    monitor = EvalMonitor(full_fit_history=True, topk=1)

    # Assemble the standard workflow
    workflow = StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor,
    )

    # Inject our custom initial population into the algorithm's internal buffer
    workflow.init_step()
    init_pop_tensor = torch.tensor(initial_pop_np, dtype=torch.float64)
    algorithm.pop.data.copy_(init_pop_tensor)

    t0 = time.time()
    gen = 0

    # Run generation-by-generation to enforce MAX_EVALS and log progress
    while _num_evals < MAX_EVALS:
        workflow.step()
        gen += 1

        # Retrieve best fitness so far from the monitor
        best_fitness = monitor.get_best_fitness()
        if best_fitness is not None:
            best_cost = float(best_fitness.item())
        else:
            best_cost = float("inf")

        fitness_history.append({
            "gen": gen, "best": best_cost, "evals": _num_evals,
        })
        print(
            f"Gen {gen:3d} | Best: {best_cost:.2f} | Evals: {_num_evals}"
        )

    elapsed = time.time() - t0

    # Retrieve the final best solution
    best_solution = monitor.get_best_solution()
    if best_solution is not None:
        best_vec = torch.round(best_solution).cpu().numpy()
    else:
        best_vec = initial_pop_np[0]

    best_fitness_final = monitor.get_best_fitness()
    best_cost = float(best_fitness_final.item()) if best_fitness_final is not None else float("inf")

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
        "fitness_history": fitness_history,
        "time_s": round(elapsed, 2),
        "algorithm": "shade_evox",
        "strategy": strategy,
        "pop_size": PYGAD_POPULATION_SIZE,
        "integrality": True,
        "num_genes": num_genes,
        "total_evals": _num_evals,
        "generations_completed": gen,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_file = out_dir / f"differential_evolution_{strategy}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved → {out_file}")

    return best_cost, elapsed


# ── Run all 3 experiments ───────────────────────────────────────────

def run_all_experiments():
    """Run 3 SHADE experiments: random, baseline, and mixed initialisation."""
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
