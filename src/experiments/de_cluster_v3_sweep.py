"""
Greedy hyperparameter sweep for SHADE + v3 cluster (novel bin-crossing)
with step-pairwise mutation.

Drives ``src.algorithms.differential_evolution_cluster_v3`` WITHOUT modifying
it: this script imports the module, overrides the handful of parameters it
reads from ``config`` (population, mutation rate, mutation step, and the
``SHADE_PAIRWISE_MUTATION`` flag) as module-level globals, then calls its
``run_single_de`` once per (tree, pop, prob, step) combination.  Because the
v3 walk-decomposition crossover is always on in that module, this sweep is a
tuning study of the *step-pairwise mutation* layered on top of bin-crossing —
hence ``SHADE_PAIRWISE_MUTATION`` is forced True for every run.

Search procedure (coordinate descent, per tree, eval budget fixed):
  Stage A — population: run pop ∈ {50,100,200} at (prob 0.1, step 1).  Keep P*.
  Stage B — mutation probability: run prob ∈ {0.3,0.5} at (P*, step 1); the
            (P*, 0.1, 1) result is reused from Stage A.  Keep M*.
  Stage C — mutation step: run step ∈ {2,3} at (P*, M*); the (P*, M*, 1)
            result is reused (from Stage A if M*=0.1, else Stage B).  Keep S*.
  → 3 + 2 + 2 = 7 runs per tree, winner = (P*, M*, S*).  × 4 trees = 28 runs.

"Best" = lowest final ``best_fitness`` (composite cost).  A per-(tree,pop,prob,
step) cache guarantees the reused combinations are never re-evaluated.

Each run's JSON (written by run_single_de as
``differential_evolution_cluster_v3_<tree>.json``) is renamed to a unique
``differential_evolution_cluster_v3_<tree>_pop<P>_mp<prob>_ms<step>.json`` so
no run overwrites another.  A combined ``de_cluster_v3_sweep_summary.json``
records every run plus the per-tree winners.

Usage:
  python -m src.experiments.de_cluster_v3_sweep
  python -m src.experiments.de_cluster_v3_sweep --max-evals 200 --trees random
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

import torch  # noqa: E402  (after sys.path setup, mirrors the algorithm modules)

import src.algorithms.differential_evolution_cluster_v3 as v3  # noqa: E402

# ── Sweep grid ──────────────────────────────────────────────────────
POPULATIONS = [50, 100, 200]      # iteration order = tie-break preference
PROBS_DEFAULT = 0.1
PROBS_OTHER = [0.3, 0.5]
STEP_DEFAULT = 1
STEP_OTHER = [2, 3]
SEED = 42                         # re-seeded per run for order-independence


def _best(d, candidates):
    """Return the candidate with the lowest 'best' cost (first on ties)."""
    return min(candidates, key=lambda c: d[c]["best"])


def run_one(ctx, tree_name, dist_path, pop, prob, step):
    """Run a single SHADE-cluster-v3 experiment, cached by (tree,pop,prob,step).

    Overrides cluster_v3's parameter globals, re-seeds RNGs for reproducibility,
    calls run_single_de, then renames the produced JSON to a unique filename.
    Returns a dict {best, time_s, file, reused}.
    """
    key = (tree_name, pop, prob, step)
    if key in ctx["cache"]:
        cached = dict(ctx["cache"][key])
        cached["reused"] = True
        return cached

    # Override the parameters cluster_v3 reads as module globals. run_single_de
    # and the crossover hook resolve these against the module namespace at call
    # time, so setting them here is sufficient.
    v3.PYGAD_POPULATION_SIZE = pop
    v3.MUTATION_RATE = float(prob)
    v3.STEP_SIZE = float(step)
    v3.SHADE_PAIRWISE_MUTATION = True

    # Deterministic, order-independent run. (EvoX SHADE also uses torch's RNG.)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"\n>>> RUN tree={tree_name} pop={pop} prob={prob} step={step}")
    best_cost, elapsed = v3.run_single_de(
        ctx["baseline_data"], ctx["num_genes"], ctx["tls_to_genes"],
        ctx["bounds_lo"], ctx["bounds_hi"], ctx["out_dir"], rng,
        tree_name=tree_name, dist_path=str(dist_path),
    )

    # run_single_de always writes this fixed name; move it aside so the next
    # run for the same tree does not clobber it.
    produced = ctx["out_dir"] / f"differential_evolution_cluster_v3_{tree_name}.json"
    unique = (
        ctx["out_dir"]
        / f"differential_evolution_cluster_v3_{tree_name}"
          f"_pop{pop}_mp{prob}_ms{step}.json"
    )
    if produced.exists():
        produced.replace(unique)
        out_name = unique.name
    else:
        out_name = None  # defensive: should not happen on a successful run

    result = {"best": best_cost, "time_s": elapsed, "file": out_name,
              "pop": pop, "prob": prob, "step": step, "reused": False}
    ctx["cache"][key] = {k: result[k] for k in ("best", "time_s", "file",
                                                 "pop", "prob", "step")}
    return result


def search_tree(ctx, tree_name, dist_path):
    """Run the 7-run coordinate-descent search for one tree."""
    print(f"\n{'#'*64}\n# Tree: {tree_name}\n{'#'*64}")

    # ── Stage A — population (prob 0.1, step 1) ──────────────────────
    stage_a = {
        p: run_one(ctx, tree_name, dist_path, p, PROBS_DEFAULT, STEP_DEFAULT)
        for p in POPULATIONS
    }
    best_pop = _best(stage_a, POPULATIONS)
    print(f"\n[{tree_name}] Stage A winner: pop={best_pop} "
          f"(cost {stage_a[best_pop]['best']:.2f})")

    # ── Stage B — mutation probability (pop=P*, step 1) ──────────────
    stage_b = {PROBS_DEFAULT: stage_a[best_pop]}  # reuse the 0.1 result
    for prob in PROBS_OTHER:
        stage_b[prob] = run_one(ctx, tree_name, dist_path, best_pop, prob,
                                STEP_DEFAULT)
    prob_order = [PROBS_DEFAULT] + PROBS_OTHER
    best_prob = _best(stage_b, prob_order)
    print(f"\n[{tree_name}] Stage B winner: prob={best_prob} "
          f"(cost {stage_b[best_prob]['best']:.2f})")

    # ── Stage C — mutation step (pop=P*, prob=M*) ────────────────────
    # The step-1 run at (P*, M*) is already cached (Stage A if M*=0.1, else
    # Stage B), so run_one returns it without re-evaluating.
    stage_c = {
        STEP_DEFAULT: run_one(ctx, tree_name, dist_path, best_pop, best_prob,
                              STEP_DEFAULT)
    }
    for step in STEP_OTHER:
        stage_c[step] = run_one(ctx, tree_name, dist_path, best_pop, best_prob,
                                step)
    step_order = [STEP_DEFAULT] + STEP_OTHER
    best_step = _best(stage_c, step_order)
    print(f"\n[{tree_name}] Stage C winner: step={best_step} "
          f"(cost {stage_c[best_step]['best']:.2f})")

    final = stage_c[best_step]
    print(f"\n[{tree_name}] FINAL: pop={best_pop} prob={best_prob} "
          f"step={best_step} → cost {final['best']:.2f}")

    return {
        "best_pop": best_pop,
        "best_prob": best_prob,
        "best_step": best_step,
        "best_cost": final["best"],
        "best_file": final["file"],
        "stage_a_population": {str(p): stage_a[p] for p in POPULATIONS},
        "stage_b_probability": {str(pr): stage_b[pr] for pr in prob_order},
        "stage_c_step": {str(s): stage_c[s] for s in step_order},
    }


def run_sweep(max_evals=None, trees=None):
    """Build the SUMO wrapper once, then run the greedy search per tree."""
    if max_evals is not None:
        v3.MAX_EVALS = int(max_evals)
        print(f"[sweep] MAX_EVALS overridden → {v3.MAX_EVALS}")

    with open(v3.BASELINE_TRAFFIC_DATA, "r") as f:
        baseline_data = json.load(f)

    wrapper, num_genes, bounds_lo, bounds_hi, _ = v3.build_traffic_fitness_wrapper(
        baseline_data=baseline_data, fitness_function=v3.fitness_function,
    )
    v3._wrapper = wrapper  # the global TLSProblem / _evaluate_single read

    tls_to_genes, _, _ = v3.build_gene_map(baseline_data)

    out_dir = ROOT / "src" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = {
        "baseline_data": baseline_data,
        "num_genes": num_genes,
        "tls_to_genes": tls_to_genes,
        "bounds_lo": bounds_lo,
        "bounds_hi": bounds_hi,
        "out_dir": out_dir,
        "cache": {},
    }

    all_trees = v3.distance_tree_paths(out_dir)
    if trees:
        selected = {t: all_trees[t] for t in trees if t in all_trees}
        missing = [t for t in trees if t not in all_trees]
        if missing:
            print(f"[sweep] WARNING: unknown trees ignored: {missing}")
    else:
        selected = all_trees

    t_start = time.time()
    summary = {}
    for tree_name, dist_path in selected.items():
        try:
            summary[tree_name] = search_tree(ctx, tree_name, dist_path)
        except Exception as e:  # keep going; record the failure
            print(f"ERROR [{tree_name}]: {e}")
            import traceback
            traceback.print_exc()
            summary[tree_name] = {"error": str(e)}

    total = time.time() - t_start
    n_runs = len(ctx["cache"])  # distinct (tree,pop,prob,step) actually run

    summary_out = {
        "sweep": "shade_cluster_v3_stepmut_coordinate_descent",
        "shade_pairwise_mutation": True,
        "max_evals": v3.MAX_EVALS,
        "seed": SEED,
        "grid": {
            "population": POPULATIONS,
            "mutation_probability": [PROBS_DEFAULT] + PROBS_OTHER,
            "mutation_step": [STEP_DEFAULT] + STEP_OTHER,
            "trees": list(selected.keys()),
        },
        "distinct_runs": n_runs,
        "total_time_s": round(total, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_tree": summary,
    }
    summary_path = out_dir / "de_cluster_v3_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=4)

    # ── Console summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Sweep done: {n_runs} distinct runs in {total:.1f}s")
    print(f"{'='*60}")
    print(f"{'Tree':<12}{'Pop':>6}{'Prob':>7}{'Step':>6}{'Best':>12}")
    print("─" * 43)
    for tree_name, info in summary.items():
        if "error" in info:
            print(f"{tree_name:<12}{'ERROR':>31}")
        else:
            print(f"{tree_name:<12}{info['best_pop']:>6}"
                  f"{info['best_prob']:>7}{info['best_step']:>6}"
                  f"{info['best_cost']:>12.2f}")
    print(f"\nPer-run JSONs + summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Override config.MAX_EVALS (e.g. small value for a smoke test).")
    parser.add_argument("--trees", nargs="*", default=None,
                        help="Subset of tree names to run (default: all in config.TREE_STRATEGIES).")
    args = parser.parse_args()
    run_sweep(max_evals=args.max_evals, trees=args.trees)


if __name__ == "__main__":
    main()
