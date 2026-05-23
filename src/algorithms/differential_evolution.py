"""
Differential Evolution (SHADE) using EvoX for traffic light optimization.

Optimizes the baseline TLS configuration using the SHADE (Success-History
based Adaptive Differential Evolution) algorithm from the EvoX library
with parallel fitness evaluation via SUMO.

When config.NOVEL_MUTATION is True, the end of each SHADE generation
applies a pair-cluster mutation (borrowed from
src.algorithms.custom_optimizer) to a MUTATION_RATE fraction of the
population.  Pair clusters are derived from the Ward linkage tree built
on each distance matrix (shortest / euclidian / fastest).  Mutants are
accepted greedily — only when they improve on their parent's fitness —
so SHADE's adaptive state is never overwritten by a worse individual.
When NOVEL_MUTATION is False, the loop runs as plain SHADE.

Runs 3 experiments: one per Ward distance tree
(shortest / euclidian / fastest), all with random initialization.
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
    GENE_LOW, GENE_HIGH,
    MUTATION_RATE,
    NOVEL_MUTATION,
    CLUSTER_THRESHOLD_FASTEST,
    CLUSTER_THRESHOLD_SHORTEST,
    CLUSTER_THRESHOLD_EUCLIDIAN,
)
from src.sumo_setup.fitness_evaluation import (
    fitness_function,
    build_traffic_fitness_wrapper,
)
from src.algorithms.custom_optimizer import (
    build_all_tree_masks,
    mutate_pair_cluster,
)

THRESHOLDS = {
    "shortest":  CLUSTER_THRESHOLD_SHORTEST,
    "euclidian": CLUSTER_THRESHOLD_EUCLIDIAN,
    "fastest":   CLUSTER_THRESHOLD_FASTEST,
}

# ── Module-level state ──────────────────────────────────────────────
_wrapper = None
_num_evals = 0


def _evaluate_single(vec):
    try:
        cost = float(_wrapper(vec))
        return cost
    except Exception as e:
        print(f"Error evaluating fitness: {e}")
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

        _num_evals += len(pop_np)

        return torch.tensor(costs, dtype=pop.dtype, device=pop.device)


# ── Population initialisation ───────────────────────────────────────

def init_population(n, num_genes, bounds_lo, bounds_hi, rng):
    """Create a random initial population.

    Returns an (n, num_genes) array with values in [bounds_lo, bounds_hi]
    and rounded to whole numbers.
    """
    pop = rng.uniform(bounds_lo, bounds_hi, (n, num_genes))
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

def run_single_de(baseline_data, num_genes, tls_to_genes,
                  bounds_lo, bounds_hi, out_dir, rng,
                  tree_name=None, dist_path=None):
    """Run a single EvoX SHADE experiment with random initialization.

    When NOVEL_MUTATION is True, end-of-generation pair-cluster mutation
    is applied to a MUTATION_RATE fraction of the population, using pairs
    derived from the Ward linkage tree at *dist_path*.  Mutants are
    accepted only if they improve on their parent's fitness.

    When NOVEL_MUTATION is False, *tree_name* and *dist_path* are ignored
    and the run is plain SHADE with random initialization.
    """
    global _wrapper, _num_evals

    _num_evals = 0
    fitness_history = []

    print(f"\n{'='*60}")
    if NOVEL_MUTATION:
        threshold = THRESHOLDS[tree_name]
        print(f"SHADE (EvoX) | Tree: {tree_name} (t={threshold}) | "
              f"Random init | Pop: {PYGAD_POPULATION_SIZE}")
        print(f"{'='*60}")
        _, pair_clusters, _ = build_all_tree_masks(dist_path, threshold)
        valid_pairs = [(a, b) for a, b in pair_clusters
                       if a in tls_to_genes and b in tls_to_genes]
        print(f"Pair-mutation: ENABLED — {len(valid_pairs)} 2-TLS pairs")
    else:
        threshold = None
        print(f"SHADE (EvoX) | Random init | Pop: {PYGAD_POPULATION_SIZE}")
        print(f"{'='*60}")
        valid_pairs = []

    n_workers = NUM_PROCESSORS or os.cpu_count() or 1

    initial_pop_np = init_population(
        PYGAD_POPULATION_SIZE, num_genes, bounds_lo, bounds_hi, rng,
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

        # ── End-of-generation pair-cluster mutation ─────────────────────
        # Pull SHADE's current population/fitness, pair-mutate a
        # MUTATION_RATE subset, then greedily replace parents only when
        # the mutant improves.  problem.evaluate() auto-increments
        # _num_evals via TLSProblem, so the eval budget is respected.
        mut_attempted = 0
        mut_improved  = 0
        if NOVEL_MUTATION and valid_pairs and _num_evals < MAX_EVALS:
            mutant_idxs = [
                i for i in range(PYGAD_POPULATION_SIZE)
                if rng.random() < MUTATION_RATE
            ]
            if mutant_idxs:
                pop_data = algorithm.pop.data
                fit_data = algorithm.fit.data
                pop_np   = torch.round(pop_data).cpu().numpy()

                mutants_np = np.stack([
                    np.clip(
                        np.round(mutate_pair_cluster(
                            pop_np[i], valid_pairs, tls_to_genes, rng
                        )),
                        bounds_lo, bounds_hi,
                    )
                    for i in mutant_idxs
                ])

                mutants_tensor = torch.tensor(
                    mutants_np, dtype=pop_data.dtype, device=pop_data.device,
                )
                mutant_fits = problem.evaluate(mutants_tensor)
                mut_attempted = len(mutant_idxs)

                for j, i in enumerate(mutant_idxs):
                    if mutant_fits[j].item() < fit_data[i].item():
                        pop_data[i] = mutants_tensor[j]
                        fit_data[i] = mutant_fits[j]
                        mut_improved += 1

        # Retrieve best fitness so far from the monitor
        best_fitness = monitor.get_best_fitness()
        if best_fitness is not None:
            best_cost = float(best_fitness.item())
        else:
            best_cost = float("inf")

        history_entry = {"gen": gen, "best": best_cost, "evals": _num_evals}
        if NOVEL_MUTATION:
            history_entry["mut_attempted"] = mut_attempted
            history_entry["mut_improved"] = mut_improved
        fitness_history.append(history_entry)

        if NOVEL_MUTATION:
            print(
                f"Gen {gen:3d} | Best: {best_cost:.2f} | Evals: {_num_evals} "
                f"| Mut+{mut_improved}/{mut_attempted}"
            )
        else:
            print(f"Gen {gen:3d} | Best: {best_cost:.2f} | Evals: {_num_evals}")

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
        "algorithm": "shade_evox_pair_mutation" if NOVEL_MUTATION else "shade_evox",
        "novel_mutation": NOVEL_MUTATION,
        "strategy": "random",
        "pop_size": PYGAD_POPULATION_SIZE,
        "integrality": True,
        "num_genes": num_genes,
        "total_evals": _num_evals,
        "generations_completed": gen,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if NOVEL_MUTATION:
        results.update({
            "tree": tree_name,
            "threshold": threshold,
            "mutation_rate": MUTATION_RATE,
            "num_pair_clusters": len(valid_pairs),
        })
        out_file = out_dir / f"differential_evolution_{tree_name}_mutation.json"
    else:
        out_file = out_dir / "differential_evolution.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved → {out_file}")

    return best_cost, elapsed


# ── Run the experiment ──────────────────────────────────────────────

def run_all_experiments():
    """Run SHADE experiments.

    With NOVEL_MUTATION=True, runs once per Ward distance tree
    (shortest / euclidian / fastest).  With NOVEL_MUTATION=False, runs
    a single plain-SHADE experiment with random initialization.
    """
    global _wrapper

    with open(BASELINE_TRAFFIC_DATA, "r") as f:
        baseline_data = json.load(f)

    _wrapper, num_genes, bounds_lo, bounds_hi, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline_data, fitness_function=fitness_function,
    )

    tls_to_genes, _, _ = build_gene_map(baseline_data)

    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "src" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    if NOVEL_MUTATION:
        trees = {
            "shortest":  out_dir / "tls_distances_shortest.json",
            "euclidian": out_dir / "tls_distances_euclidian.json",
            "fastest":   out_dir / "tls_distances_fastest.json",
        }
        summary = {}

        for tree_name, dist_path in trees.items():
            try:
                best_cost, elapsed = run_single_de(
                    baseline_data, num_genes, tls_to_genes,
                    bounds_lo, bounds_hi, out_dir, rng,
                    tree_name=tree_name, dist_path=str(dist_path),
                )
                summary[tree_name] = {"best": best_cost, "time_s": elapsed}
            except Exception as e:
                print(f"ERROR [{tree_name}]: {e}")
                import traceback; traceback.print_exc()
                summary[tree_name] = {"error": str(e)}

        print(f"\n{'Tree':<12} {'Best':>12} {'Time':>8}")
        print("─" * 34)
        for tree_name, info in summary.items():
            if "error" in info:
                print(f"{tree_name:<12} {'ERROR':>12}")
            else:
                print(f"{tree_name:<12} {info['best']:>12.2f} {info['time_s']:>7.1f}s")
    else:
        try:
            best_cost, elapsed = run_single_de(
                baseline_data, num_genes, tls_to_genes,
                bounds_lo, bounds_hi, out_dir, rng,
            )
            print(f"\n{'Best':>12} {'Time':>8}")
            print("─" * 22)
            print(f"{best_cost:>12.2f} {elapsed:>7.1f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    run_all_experiments()
