"""
Differential Evolution (SHADE) with v3 walk-decomposition cluster
crossover.

Same scaffolding as `differential_evolution_cluster.py`, but the
crossover hook calls `LinkageTree.find_node_decomposition` (from
`node_finder_v3`) instead of `find_node_closest_to_size`.  For each
individual *i*:

    target_size_i = CR_i * num_tls   (float, no rounding)

is decomposed by walking the Ward linkage tree:
  1. Start at the smallest cluster whose size >= target_size_i.
  2. At each node, if a child matches the remainder exactly, take it.
     Otherwise pick a child 50/50, subtract its size, recurse into the
     sibling subtree.
  3. Stop on exact match, leaf, or overshoot.

The resulting trial vector copies the genes belonging to *all* selected
clusters from the mutant vector and inherits the rest from the parent.

Runs 4 experiments, one per Ward distance tree (shortest / euclidian /
fastest / random); each uses its own linkage tree for the cluster lookup.

Usage:  python -m src.algorithms.differential_evolution_cluster_v3
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
from evox.algorithms.so.de_variants import shade as _shade_module

# ── Module-level state used by the cluster-crossover hook ──────────
# Populated per-experiment in run_single_de() before any workflow.step()
# call so the monkey-patched crossover sees the right tree / mapping.
_linkage_tree   = None  # LinkageTree (v3) for the active experiment
_tls_to_genes   = None  # dict[tls_id] -> (start, end) gene slice
_num_tls        = None  # genotype length at the TLS level (36 here)
_xover_rng      = None  # rng for tiebreaks / 50-50 picks in the walk

# ── Step-pairwise-mutation state (only used when SHADE_PAIRWISE_MUTATION) ──
_pair_clusters  = None  # list[(tls_a, tls_b)] size-2 Ward clusters
_phase_split    = None  # dict[tls_id] -> {green/red/yellow/mutable: indices}
_lb_np          = None  # per-gene lower bounds as numpy
_ub_np          = None  # per-gene dynamic upper bounds as numpy

# Captured per-step for printing / JSON logging
_last_cr_vect      = None
_last_target_sizes = None
_last_actual_sizes = None


def _cluster_binary_crossover(mutation_vector, current_vect, CR_vect):
    """SHADE-compatible crossover that operates on TLS clusters via the
    v3 walk-decomposition.

    Drop-in replacement for `DE_binary_crossover`.  Signature and shapes
    match exactly so SHADE.step() is otherwise unmodified.
    """
    global _last_cr_vect, _last_target_sizes, _last_actual_sizes
    _last_cr_vect = CR_vect.detach().clone()

    pop_size = CR_vect.shape[0]
    trial = current_vect.clone()

    cr_np   = CR_vect.detach().cpu().numpy()
    targets = np.rint(cr_np * _num_tls).astype(int)
    actual  = np.empty(pop_size, dtype=int)

    for i in range(pop_size):
        # ── Step-pairwise-mutation (augment-on-top, not separately evaluated) ──
        # With probability MUTATION_RATE, grow the "second" TLS of a random
        # size-2 Ward cluster to (first_green_sum + STEP_SIZE).  Applied to the
        # trial row BEFORE the decomposition copy below, so this individual's
        # trial carries both edits into SHADE's single per-generation eval.
        if (SHADE_PAIRWISE_MUTATION and _pair_clusters
                and _xover_rng.random() < MUTATION_RATE):
            row = trial[i].detach().cpu().numpy()
            mutated = mutate_pair_cluster_step(
                row, _pair_clusters, _tls_to_genes,
                _phase_split, _xover_rng, _ub_np, STEP_SIZE,
            )
            mutated = np.clip(np.round(mutated), _lb_np, _ub_np)
            trial[i].copy_(
                torch.as_tensor(mutated, dtype=trial.dtype, device=trial.device)
            )

        members = _linkage_tree.find_node_decomposition(
            int(targets[i]), rng=_xover_rng,
        )
        # Translate TLS IDs -> flat gene indices, skipping TLS not in
        # the gene map (defensive: linkage tree IDs come from the
        # distance JSON; not all may have phases in baseline_data).
        gene_idxs = []
        kept = 0
        for tls_id in members:
            if tls_id in _tls_to_genes:
                s, e = _tls_to_genes[tls_id]
                gene_idxs.extend(range(s, e))
                kept += 1
        actual[i] = kept

        if gene_idxs:
            idx = torch.tensor(gene_idxs, dtype=torch.long, device=trial.device)
            trial[i].index_copy_(0, idx, mutation_vector[i].index_select(0, idx))

    _last_target_sizes = targets.tolist()
    _last_actual_sizes = actual.tolist()
    return trial


_shade_module.DE_binary_crossover = _cluster_binary_crossover

# Add project root to sys.path before importing config / project modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (
    PYGAD_POPULATION_SIZE,
    MAX_EVALS,
    NUM_PROCESSORS,
    BASELINE_TRAFFIC_DATA,
    MUTATION_RATE,
    SHADE_PAIRWISE_MUTATION,
    STEP_SIZE,
)
from src.sumo_setup.fitness_evaluation import (
    fitness_function,
    build_traffic_fitness_wrapper,
)
from src.novel.node_finder_v3 import LinkageTree
from src.novel.linkage_tree import build_all_tree_masks
from src.novel.pairwise_mutation import build_phase_split
from src.novel.SHADE_mutation import mutate_pair_cluster_step
from src.novel.distance_trees import distance_tree_paths


# ── SUMO wrapper / problem (parallel SUMO evaluation) ──────────────
_wrapper   = None
_num_evals = 0


def _evaluate_single(vec):
    try:
        return float(_wrapper(vec))
    except Exception as e:
        print(f"Error evaluating fitness: {e}")
        return 9999999.0


class TLSProblem(Problem):
    """EvoX problem evaluating TLS solutions via the SUMO wrapper."""

    def __init__(self, wrapper, n_workers=1):
        super().__init__()
        self.wrapper = wrapper
        self.n_workers = n_workers

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        pop_np = torch.round(pop).cpu().numpy()
        if self.n_workers > 1:
            with Pool(processes=self.n_workers) as pool:
                costs = pool.map(_evaluate_single, [row for row in pop_np])
        else:
            costs = [_evaluate_single(row) for row in pop_np]
        return torch.tensor(costs, dtype=pop.dtype, device=pop.device)


# ── Helpers (population init + gene map, mirrors differential_evolution.py)

def init_population(n, num_genes, bounds_lo, bounds_hi, rng):
    pop = rng.uniform(bounds_lo, bounds_hi, (n, num_genes))
    return np.round(pop).astype(float)


def build_gene_map(baseline_data):
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


# ── Single SHADE-with-cluster-crossover run ────────────────────────

def run_single_de(baseline_data, num_genes, tls_to_genes,
                  bounds_lo, bounds_hi, out_dir, rng,
                  tree_name, dist_path):
    """Run one SHADE experiment whose crossover is the v3 cluster variant."""
    global _wrapper, _num_evals
    global _linkage_tree, _tls_to_genes, _num_tls, _xover_rng
    global _pair_clusters, _phase_split, _lb_np, _ub_np

    _num_evals = 0
    fitness_history = []

    # ── Install per-experiment state for the crossover hook ──────
    _linkage_tree = LinkageTree.from_distance_json(dist_path)
    _tls_to_genes = tls_to_genes
    _num_tls      = len(tls_to_genes)
    _xover_rng    = rng

    # ── Optional step-pairwise-mutation state (same pair/phase sourcing as
    #    differential_evolution.py's pair-mutation) ──────────────────────
    _pair_clusters = None
    _phase_split   = None
    _lb_np = np.asarray(bounds_lo)
    _ub_np = np.asarray(bounds_hi)
    if SHADE_PAIRWISE_MUTATION:
        # No threshold needed: only pair_clusters is used here, and that
        # output ignores the threshold (mixing_masks, which the cut-off gates,
        # is discarded).
        _, pair_clusters, _ = build_all_tree_masks(dist_path)
        _pair_clusters = [(a, b) for a, b in pair_clusters
                          if a in tls_to_genes and b in tls_to_genes]
        _phase_split = build_phase_split(baseline_data, tls_to_genes)

    print(f"\n{'='*60}")
    print(f"SHADE (EvoX) | Cluster crossover v3 | Tree: {tree_name} | "
          f"num_tls={_num_tls} | Pop: {PYGAD_POPULATION_SIZE}")
    if SHADE_PAIRWISE_MUTATION:
        print(f"Step-pairwise-mutation: ENABLED — {len(_pair_clusters)} "
              f"2-TLS pairs | step={STEP_SIZE} | prob={MUTATION_RATE}")
    print(f"{'='*60}")

    n_workers = NUM_PROCESSORS or os.cpu_count() or 1

    initial_pop_np = init_population(
        PYGAD_POPULATION_SIZE, num_genes, bounds_lo, bounds_hi, rng,
    )

    lb = torch.tensor(bounds_lo, dtype=torch.float64)
    ub = torch.tensor(bounds_hi, dtype=torch.float64)

    algorithm = SHADE(pop_size=PYGAD_POPULATION_SIZE, lb=lb, ub=ub)
    problem   = TLSProblem(wrapper=_wrapper, n_workers=n_workers)
    monitor   = EvalMonitor(full_fit_history=True, topk=1)

    workflow = StdWorkflow(
        algorithm=algorithm, problem=problem, monitor=monitor,
    )

    # Inject our custom initial population (see differential_evolution.py
    # for the rationale on skipping workflow.init_step()).
    init_pop_tensor = torch.tensor(initial_pop_np, dtype=torch.float64)
    algorithm.pop.data.copy_(init_pop_tensor)
    init_fitness = workflow.algorithm.evaluate(init_pop_tensor)
    algorithm.fit.data.copy_(init_fitness)

    t0 = time.time()
    gen = 0

    # Initialization evaluations are not counted against MAX_EVALS;
    # the budget covers only the generational loop.
    while _num_evals < MAX_EVALS:
        workflow.step()
        _num_evals += PYGAD_POPULATION_SIZE
        gen += 1

        cr_np = _last_cr_vect.cpu().numpy() if _last_cr_vect is not None else None
        if cr_np is not None:
            cr_str = np.array2string(
                cr_np, precision=3, suppress_small=True, max_line_width=200,
            )
            print(f"  CR per individual (gen {gen}): {cr_str}")

        if _last_target_sizes is not None:
            pairs = " ".join(
                f"{t}->{a}" for t, a in zip(_last_target_sizes, _last_actual_sizes)
            )
            print(f"  Cluster size target->actual (gen {gen}): [{pairs}]")

        best_fitness = monitor.get_best_fitness()
        best_cost = float(best_fitness.item()) if best_fitness is not None else float("inf")

        # Per-generation pop fitness (best/worst/mean of current population)
        try:
            current_fit_np = algorithm.fit.detach().cpu().numpy()
            gen_best  = float(np.min(current_fit_np))
            gen_worst = float(np.max(current_fit_np))
            gen_mean  = float(np.mean(current_fit_np))
        except Exception:
            gen_best = gen_worst = gen_mean = float("nan")

        history_entry = {
            "gen": gen,
            "best": best_cost,
            "gen_best": gen_best,
            "gen_worst": gen_worst,
            "mean": gen_mean,
            "evals": _num_evals,
        }
        if cr_np is not None:
            history_entry["cr"] = cr_np.tolist()
        if _last_target_sizes is not None:
            history_entry["target_sizes"] = list(_last_target_sizes)
            history_entry["actual_sizes"] = list(_last_actual_sizes)
        fitness_history.append(history_entry)

        print(f"Gen {gen:3d} | Best: {best_cost:.2f} | Evals: {_num_evals}")

    elapsed = time.time() - t0

    best_solution = monitor.get_best_solution()
    best_vec = (
        torch.round(best_solution).cpu().numpy()
        if best_solution is not None else initial_pop_np[0]
    )
    best_fitness_final = monitor.get_best_fitness()
    best_cost = float(best_fitness_final.item()) if best_fitness_final is not None else float("inf")

    print(f"Done in {elapsed:.1f}s | Final best: {best_cost:.2f} | Evals: {_num_evals}")

    # Reconstruct full TLS JSON from the flat gene vector
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
        "algorithm": (
            "shade_evox_cluster_crossover_v3_stepmut"
            if SHADE_PAIRWISE_MUTATION else "shade_evox_cluster_crossover_v3"
        ),
        "strategy": "random",
        "pop_size": PYGAD_POPULATION_SIZE,
        "integrality": True,
        "num_genes": num_genes,
        "num_tls": _num_tls,
        "total_evals": _num_evals,
        "generations_completed": gen,
        "tree": tree_name,
        "shade_pairwise_mutation": SHADE_PAIRWISE_MUTATION,
        "step_size": STEP_SIZE if SHADE_PAIRWISE_MUTATION else None,
        "num_pair_clusters": len(_pair_clusters) if _pair_clusters else 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_file = out_dir / f"differential_evolution_cluster_v3_{tree_name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved → {out_file}")

    return best_cost, elapsed


def run_all_experiments():
    """Always runs three experiments, one per linkage tree."""
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

    trees = distance_tree_paths(out_dir)
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


if __name__ == "__main__":
    run_all_experiments()
