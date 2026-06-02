"""
Single-repetition DE-SHADE sweep — identical to ``de_experiments.py`` but run
ONCE instead of twice-and-averaged.

This module runs exactly the same hyperparameter sweep as ``de_experiments.py``
— it does NOT use any pre-selected "winning" configuration.  The only
difference is the number of repetitions: ``de_experiments.py`` runs every
configuration twice (seeds 42 / 43) and selects winners on the *averaged* cost,
whereas this module runs every configuration ONCE (a single seed) and selects
winners on that single run's cost.  It is meant to reproduce the full search at
a fraction of the cost when a noisy single-run estimate is acceptable.

Neither algorithm module is modified: this script imports them, overrides the
handful of parameters they read from ``config`` as module-level globals, and
calls their ``run_single_de`` once per configuration.

WHAT IS SWEPT (same grid as de_experiments.py)
----------------------------------------------
* Novel cluster-v3 (SHADE_PAIRWISE_MUTATION=True) — greedy coordinate descent
  per tree:
    Stage A — population: pop ∈ {50,100,200} at (prob 0.1, step 1). Keep P*.
    Stage B — mutation probability: prob ∈ {0.3,0.5} at (P*, step 1). Keep M*.
    Stage C — mutation step: step ∈ {2,3} at (P*, M*). Keep S*.
  → 7 combos per tree × 4 trees, each run ONCE; winner selection uses the
    single run's cost.

* Novel cluster-v3 WITHOUT step-mutation (SHADE_PAIRWISE_MUTATION=False) —
  same walk-decomposition bin-crossing, no step-mutation.  Plain pop × tree
  sweep: 3 pops × 4 trees, each run once.

* Plain SHADE (NOVEL_MUTATION=False) — swept over the SAME 3 populations
  {50,100,200}, each run once.

Both algorithm modules monkeypatch the SAME symbol
(``_shade_module.DE_binary_crossover``) at import time, so whichever is imported
LAST wins globally.  We re-assert the correct crossover before each section (see
``_use_v3_crossover`` / ``_use_plain_crossover``).

OUTPUT FILES (in src/outputs/single_de_repetition/)
---------------------------------------------------
* Per-run algorithm JSONs are renamed uniquely so no run clobbers another:
    ``single_de_cluster_v3_<tree>_pop<P>_mp<prob>_ms<step>.json``  (with mut)
    ``single_de_cluster_v3_nomut_<tree>_pop<P>.json``             (no mut)
    ``single_de_plain_pop<P>.json``                               (plain SHADE)
* Three summary files (one per variant, mirroring de_experiments' families):
    ``single_de_repetition.json``        — cluster-v3 WITH step-mutation sweep
    ``single_de_repetition_nomut.json``  — cluster-v3 NO mutation sweep
    ``single_de_repetition_plain.json``  — plain SHADE sweep

Usage:
  python -m src.experiments.single_de_repetition
  python -m src.experiments.single_de_repetition --max-evals 200 --seed 42
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

# Both algorithm modules monkeypatch the SAME symbol
# (``_shade_module.DE_binary_crossover``); re-assert the right one per section.
import src.algorithms.differential_evolution_cluster_v3 as v3  # noqa: E402
import src.algorithms.differential_evolution as de  # noqa: E402
from evox.algorithms.so.de_variants import shade as _shade_module  # noqa: E402

# Single repetition → one seed.  Distinct from de_experiments.py's rep seeds
# (42 / 43) so this run is an independent draw, not a duplicate of either rep.
DEFAULT_SEED = 44

# ── Sweep grid (identical to de_experiments.py) ───────────────────────
POPULATIONS = [50, 100, 200]      # iteration order = tie-break preference
PROBS_DEFAULT = 0.1
PROBS_OTHER = [0.3, 0.5]
STEP_DEFAULT = 1
STEP_OTHER = [2, 3]


def _use_v3_crossover():
    """Make the v3 cluster crossover the active SHADE crossover."""
    _shade_module.DE_binary_crossover = v3._cluster_binary_crossover


def _use_plain_crossover():
    """Restore the plain (CR-capturing) SHADE crossover."""
    _shade_module.DE_binary_crossover = de._capturing_binary_crossover


def _best(d, candidates):
    """Return the candidate with the lowest cost (first on ties)."""
    return min(candidates, key=lambda c: d[c]["best"])


# ══════════════════════════════════════════════════════════════════════
# Novel cluster-v3 step-mutation sweep (run each combo once)
# ══════════════════════════════════════════════════════════════════════

def run_one(ctx, tree_name, dist_path, pop, prob, step):
    """Run one SHADE-cluster-v3 config ONCE.

    Cached by (tree,pop,prob,step) so reused coordinate-descent combinations
    are never re-evaluated.  Returns a dict::

        {"best", "pop", "prob", "step", "time_s", "file", "reused"}
    """
    key = (tree_name, pop, prob, step)
    if key in ctx["cache"]:
        cached = dict(ctx["cache"][key])
        cached["reused"] = True
        return cached

    seed = ctx["seed"]
    # Override the parameters cluster_v3 reads as module globals.
    v3.PYGAD_POPULATION_SIZE = pop
    v3.MUTATION_RATE = float(prob)
    v3.STEP_SIZE = float(step)
    v3.SHADE_PAIRWISE_MUTATION = True

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    print(f"\n>>> RUN[v3] tree={tree_name} pop={pop} prob={prob} "
          f"step={step} (seed={seed})")
    best_cost, elapsed = v3.run_single_de(
        ctx["baseline_data"], ctx["num_genes"], ctx["tls_to_genes"],
        ctx["bounds_lo"], ctx["bounds_hi"], ctx["out_dir"], rng,
        tree_name=tree_name, dist_path=str(dist_path),
    )

    # run_single_de always writes this fixed name; move it aside so the next
    # run for the same tree does not clobber it.
    produced = (ctx["out_dir"]
                / f"differential_evolution_cluster_v3_{tree_name}.json")
    unique = (ctx["out_dir"]
              / f"single_de_cluster_v3_{tree_name}"
                f"_pop{pop}_mp{prob}_ms{step}.json")
    out_name = None
    if produced.exists():
        produced.replace(unique)
        out_name = unique.name

    result = {"best": best_cost, "pop": pop, "prob": prob, "step": step,
              "time_s": elapsed, "file": out_name, "reused": False}
    ctx["cache"][key] = {k: result[k] for k in
                         ("best", "pop", "prob", "step", "time_s", "file")}
    return result


def search_tree(ctx, tree_name, dist_path):
    """Run the 7-combo coordinate-descent search for one tree (single run each)."""
    print(f"\n{'#'*64}\n# Tree: {tree_name}\n{'#'*64}")

    # ── Stage A — population (prob 0.1, step 1) ──────────────────────
    stage_a = {
        p: run_one(ctx, tree_name, dist_path, p, PROBS_DEFAULT, STEP_DEFAULT)
        for p in POPULATIONS
    }
    best_pop = _best(stage_a, POPULATIONS)
    print(f"\n[{tree_name}] Stage A winner: pop={best_pop} "
          f"(best {stage_a[best_pop]['best']:.2f})")

    # ── Stage B — mutation probability (pop=P*, step 1) ──────────────
    stage_b = {PROBS_DEFAULT: stage_a[best_pop]}  # reuse the 0.1 result
    for prob in PROBS_OTHER:
        stage_b[prob] = run_one(ctx, tree_name, dist_path, best_pop, prob,
                                STEP_DEFAULT)
    prob_order = [PROBS_DEFAULT] + PROBS_OTHER
    best_prob = _best(stage_b, prob_order)
    print(f"\n[{tree_name}] Stage B winner: prob={best_prob} "
          f"(best {stage_b[best_prob]['best']:.2f})")

    # ── Stage C — mutation step (pop=P*, prob=M*) ────────────────────
    # The step-1 run at (P*, M*) is already cached, so run_one returns it
    # without re-evaluating.
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
          f"(best {stage_c[best_step]['best']:.2f})")

    final = stage_c[best_step]
    print(f"\n[{tree_name}] FINAL: pop={best_pop} prob={best_prob} "
          f"step={best_step} → best {final['best']:.2f}")

    return {
        "best_pop": best_pop,
        "best_prob": best_prob,
        "best_step": best_step,
        "winner": final,
        "stage_a_population": {str(p): stage_a[p] for p in POPULATIONS},
        "stage_b_probability": {str(pr): stage_b[pr] for pr in prob_order},
        "stage_c_step": {str(s): stage_c[s] for s in step_order},
    }


# ══════════════════════════════════════════════════════════════════════
# Novel cluster-v3 WITHOUT step-mutation (run each pop×tree once)
# ══════════════════════════════════════════════════════════════════════

def run_one_nomut(ctx, tree_name, dist_path, pop):
    """Run cluster-v3 (novel bin-crossing) with step-mutation OFF, once.

    Same walk-decomposition crossover as the mutation sweep, but
    SHADE_PAIRWISE_MUTATION=False so no step-mutation is applied.  There is no
    prob/step axis here — only the population.  run_single_de's output records
    ``shade_pairwise_mutation=false`` / ``step_size=null`` so the JSON itself
    marks these as mutation-free.  Returns {best, time_s, file}.
    """
    seed = ctx["seed"]
    v3.PYGAD_POPULATION_SIZE = pop
    v3.SHADE_PAIRWISE_MUTATION = False

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    print(f"\n>>> RUN[v3-nomut] tree={tree_name} pop={pop} (seed={seed})")
    best_cost, elapsed = v3.run_single_de(
        ctx["baseline_data"], ctx["num_genes"], ctx["tls_to_genes"],
        ctx["bounds_lo"], ctx["bounds_hi"], ctx["out_dir"], rng,
        tree_name=tree_name, dist_path=str(dist_path),
    )

    produced = (ctx["out_dir"]
                / f"differential_evolution_cluster_v3_{tree_name}.json")
    unique = (ctx["out_dir"]
              / f"single_de_cluster_v3_nomut_{tree_name}_pop{pop}.json")
    out_name = None
    if produced.exists():
        produced.replace(unique)
        out_name = unique.name

    print(f"\n[v3-nomut {tree_name} pop={pop}] best {best_cost:.2f}")
    return {"best": best_cost, "time_s": elapsed, "file": out_name}


def run_nomut_sweep(ctx, selected):
    """Sweep cluster-v3 (no mutation) over POPULATIONS for each tree (once each).

    Returns {tree: {pop_str: {best, time_s, file}}}.
    """
    _use_v3_crossover()
    per_tree = {}
    for tree_name, dist_path in selected.items():
        try:
            per_tree[tree_name] = {
                str(pop): run_one_nomut(ctx, tree_name, dist_path, pop)
                for pop in POPULATIONS
            }
        except Exception as e:  # keep going; record the failure
            print(f"ERROR [v3-nomut:{tree_name}]: {e}")
            import traceback
            traceback.print_exc()
            per_tree[tree_name] = {"error": str(e)}
    return per_tree


# ══════════════════════════════════════════════════════════════════════
# Plain SHADE sweep over the same 3 populations (run each once)
# ══════════════════════════════════════════════════════════════════════

def run_plain_de(ctx):
    """Run plain SHADE (NOVEL_MUTATION=False) over POPULATIONS, once each.

    Plain SHADE ignores the linkage trees, so there is no tree or mutation
    axis — just the 3 populations.  Returns {pop_str: {best, time_s, file}}.
    """
    _use_plain_crossover()
    de.NOVEL_MUTATION = False

    seed = ctx["seed"]
    results = {}
    for pop in POPULATIONS:
        de.PYGAD_POPULATION_SIZE = pop
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        print(f"\n>>> RUN[plain] pop={pop} (seed={seed})")
        best_cost, elapsed = de.run_single_de(
            ctx["baseline_data"], ctx["num_genes"], ctx["tls_to_genes"],
            ctx["bounds_lo"], ctx["bounds_hi"], ctx["out_dir"], rng,
        )

        produced = ctx["out_dir"] / "differential_evolution.json"
        unique = ctx["out_dir"] / f"single_de_plain_pop{pop}.json"
        out_name = None
        if produced.exists():
            produced.replace(unique)
            out_name = unique.name

        results[str(pop)] = {"best": best_cost, "time_s": elapsed,
                             "file": out_name}
        print(f"\n[plain pop={pop}] best {best_cost:.2f}")

    return results


def _pick_best_pop(by_pop):
    """Name the lowest-cost population for a {pop_str: {best, ...}} dict."""
    best_pop = min(by_pop, key=lambda p: by_pop[p]["best"])
    return {"best_pop": int(best_pop), "by_pop": by_pop}


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════

def run_experiments(max_evals=None, trees=None, seed=DEFAULT_SEED):
    """Build the SUMO wrapper once, run all three sweeps once, write 3 files."""
    if max_evals is not None:
        v3.MAX_EVALS = int(max_evals)
        de.MAX_EVALS = int(max_evals)
        print(f"[single_de_repetition] MAX_EVALS overridden → {max_evals}")

    with open(v3.BASELINE_TRAFFIC_DATA, "r") as f:
        baseline_data = json.load(f)

    wrapper, num_genes, bounds_lo, bounds_hi, _ = v3.build_traffic_fitness_wrapper(
        baseline_data=baseline_data, fitness_function=v3.fitness_function,
    )
    # Both modules read their own module-level _wrapper from TLSProblem.
    v3._wrapper = wrapper
    de._wrapper = wrapper

    tls_to_genes, _, _ = v3.build_gene_map(baseline_data)

    # Distance-tree JSONs live in the main outputs dir; results are written to
    # a dedicated subfolder so this experiment's files stay self-contained.
    base_out = ROOT / "src" / "outputs"
    out_dir = base_out / "single_de_repetition"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = {
        "baseline_data": baseline_data,
        "num_genes": num_genes,
        "tls_to_genes": tls_to_genes,
        "bounds_lo": bounds_lo,
        "bounds_hi": bounds_hi,
        "out_dir": out_dir,
        "cache": {},
        "seed": seed,
    }

    all_trees = v3.distance_tree_paths(base_out)
    if trees:
        selected = {t: all_trees[t] for t in trees if t in all_trees}
        missing = [t for t in trees if t not in all_trees]
        if missing:
            print(f"[single_de_repetition] WARNING: unknown trees ignored: {missing}")
    else:
        selected = all_trees

    t_start = time.time()

    # ── Section 1: novel cluster-v3 step-mutation sweep ──────────────
    _use_v3_crossover()
    per_tree = {}
    for tree_name, dist_path in selected.items():
        try:
            per_tree[tree_name] = search_tree(ctx, tree_name, dist_path)
        except Exception as e:  # keep going; record the failure
            print(f"ERROR [v3:{tree_name}]: {e}")
            import traceback
            traceback.print_exc()
            per_tree[tree_name] = {"error": str(e)}

    # ── Section 2: cluster-v3 WITHOUT step-mutation (pop × tree) ─────
    nomut = run_nomut_sweep(ctx, selected)

    # ── Section 3: plain SHADE over the same populations ─────────────
    try:
        plain = run_plain_de(ctx)
    except Exception as e:
        print(f"ERROR [plain]: {e}")
        import traceback
        traceback.print_exc()
        plain = {"error": str(e)}

    total = time.time() - t_start

    # ── Write the three summary files (one per variant) ──────────────
    mut_payload = {
        "variant": "single_de_repetition",
        "seed": seed,
        "max_evals": v3.MAX_EVALS,
        "grid": {
            "population": POPULATIONS,
            "mutation_probability": [PROBS_DEFAULT] + PROBS_OTHER,
            "mutation_step": [STEP_DEFAULT] + STEP_OTHER,
            "trees": list(selected.keys()),
        },
        "total_time_s": round(total, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cluster_v3_novel_mutation": per_tree,
    }
    mut_path = out_dir / "single_de_repetition.json"
    with open(mut_path, "w") as f:
        json.dump(mut_payload, f, indent=4)

    nomut_payload = {
        "variant": "single_de_repetition_no_mutation",
        "seed": seed,
        "max_evals": v3.MAX_EVALS,
        "grid": {"population": POPULATIONS, "trees": list(selected.keys())},
        "total_time_s": round(total, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cluster_v3_no_mutation": {
            t: (by_pop if "error" in by_pop else _pick_best_pop(by_pop))
            for t, by_pop in nomut.items()
        },
    }
    nomut_path = out_dir / "single_de_repetition_nomut.json"
    with open(nomut_path, "w") as f:
        json.dump(nomut_payload, f, indent=4)

    plain_payload = {
        "variant": "single_de_repetition_plain",
        "seed": seed,
        "max_evals": de.MAX_EVALS,
        "grid": {"population": POPULATIONS},
        "total_time_s": round(total, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plain_de_shade": (plain if "error" in plain else _pick_best_pop(plain)),
    }
    plain_path = out_dir / "single_de_repetition_plain.json"
    with open(plain_path, "w") as f:
        json.dump(plain_payload, f, indent=4)

    # ── Console summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"single_de_repetition done in {total:.1f}s")
    print(f"{'='*60}")
    print("Cluster-v3 novel mutation (best):")
    print(f"{'Tree':<12}{'Pop':>6}{'Prob':>7}{'Step':>6}{'Best':>12}")
    print("─" * 43)
    for tree_name, info in per_tree.items():
        if "error" in info:
            print(f"{tree_name:<12}{'ERROR':>31}")
        else:
            print(f"{tree_name:<12}{info['best_pop']:>6}"
                  f"{info['best_prob']:>7}{info['best_step']:>6}"
                  f"{info['winner']['best']:>12.2f}")

    print("\nCluster-v3 NO mutation (best per tree):")
    print(f"{'Tree':<12}{'Pop':>6}{'Best':>12}")
    print("─" * 30)
    for tree_name, by_pop in nomut.items():
        if "error" in by_pop:
            print(f"{tree_name:<12}{'ERROR':>18}")
        else:
            best_pop = min(POPULATIONS, key=lambda p: by_pop[str(p)]["best"])
            print(f"{tree_name:<12}{best_pop:>6}"
                  f"{by_pop[str(best_pop)]['best']:>12.2f}")

    print("\nPlain SHADE (best):")
    print(f"{'Pop':>6}{'Best':>12}")
    print("─" * 18)
    if "error" not in plain:
        best_plain_pop = min(POPULATIONS, key=lambda p: plain[str(p)]["best"])
        for pop in POPULATIONS:
            star = " *" if pop == best_plain_pop else ""
            print(f"{pop:>6}{plain[str(pop)]['best']:>12.2f}{star}")
        print(f"→ best plain pop: {best_plain_pop}")

    print(f"\nWrote:\n  {mut_path}\n  {nomut_path}\n  {plain_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Override config.MAX_EVALS (e.g. small value for a smoke test).")
    parser.add_argument("--trees", nargs="*", default=None,
                        help="Subset of tree names for the cluster-v3 sweeps (default: all).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed for the single repetition (default: {DEFAULT_SEED}).")
    args = parser.parse_args()
    run_experiments(max_evals=args.max_evals, trees=args.trees, seed=args.seed)


if __name__ == "__main__":
    main()
