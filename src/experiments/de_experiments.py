"""
DE-SHADE experiments — novel cluster-v3 step-mutation sweep + plain SHADE,
each repeated twice and averaged.

This module consolidates two things that used to live elsewhere:

  1. The SHADE + v3-cluster (novel bin-crossing) *step-pairwise-mutation*
     greedy hyperparameter sweep (formerly ``de_cluster_v3_sweep.py``).
  2. Plain SHADE (``differential_evolution`` with NOVEL_MUTATION=False),
     formerly launched as a subprocess from ``baseline_experiments.py``.

Neither algorithm module is modified: this script imports them, overrides the
handful of parameters they read from ``config`` as module-level globals, and
calls their ``run_single_de`` once per configuration.

WHY TWO REPETITIONS
-------------------
SHADE — and especially the step-pairwise mutation layered on top of it — is
stochastic.  A single run is a noisy point estimate.  We therefore run every
configuration twice with *distinct* seeds (rep 1 → 42, rep 2 → 43) and average
the two final ``best_fitness`` values.  Re-using the same seed would make the
runs byte-identical and the average meaningless, so the seeds must differ.

WHAT IS SWEPT
-------------
* Novel cluster-v3 (SHADE_PAIRWISE_MUTATION=True) — greedy coordinate descent
  per tree, exactly as the original sweep:
    Stage A — population: pop ∈ {50,100,200} at (prob 0.1, step 1). Keep P*.
    Stage B — mutation probability: prob ∈ {0.3,0.5} at (P*, step 1). Keep M*.
    Stage C — mutation step: step ∈ {2,3} at (P*, M*). Keep S*.
  → 7 combos per tree × 4 trees.  Each combo is run TWICE (one per rep); the
    AVERAGED cost drives winner selection so the chosen config is robust to
    the seed.

* Novel cluster-v3 WITHOUT step-mutation (SHADE_PAIRWISE_MUTATION=False) —
  same walk-decomposition bin-crossing, but no step-mutation.  No prob/step
  axis, so it is a plain pop × tree sweep: 3 pops × 4 trees = 12 configs, each
  run twice.  run_single_de tags these JSONs ``shade_pairwise_mutation=false``
  / ``step_size=null`` so the output marks them mutation-free.

* Plain SHADE (NOVEL_MUTATION=False) — swept over the SAME 3 populations
  {50,100,200}.  Plain SHADE ignores the linkage trees, so there is no tree or
  mutation axis: 3 configs, each run twice.

OUTPUT FILES (in src/outputs/)
------------------------------
* ``de_experiments_rep1.json`` / ``de_experiments_rep2.json`` — per-repetition
  view: every evaluated configuration with that rep's best cost / time / file.
* ``de_experiments_averaged.json`` — averaged best cost per configuration plus
  the per-tree / per-pop winners (selected on the averaged cost).  Sections:
  ``cluster_v3_novel_mutation`` / ``cluster_v3_no_mutation`` / ``plain_de_shade``.
* Per-run algorithm JSONs are renamed uniquely and tagged ``_rep1`` / ``_rep2``
  so no run clobbers another (no-mutation runs carry a ``_nomut`` marker).

Usage:
  python -m src.experiments.de_experiments
  python -m src.experiments.de_experiments --max-evals 200 --trees random
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
# (``_shade_module.DE_binary_crossover``) at import time, so whichever is
# imported LAST wins globally.  We re-assert the correct crossover before each
# section below (see _use_v3_crossover / _use_plain_crossover).
import src.algorithms.differential_evolution_cluster_v3 as v3  # noqa: E402
import src.algorithms.differential_evolution as de  # noqa: E402
from evox.algorithms.so.de_variants import shade as _shade_module  # noqa: E402

# ── Repetition seeds — MUST differ so the stochastic runs actually vary ──
REP_SEEDS = {1: 42, 2: 43}

# ── Sweep grid (unchanged from the original coordinate-descent sweep) ──
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
    """Return the candidate with the lowest averaged cost (first on ties)."""
    return min(candidates, key=lambda c: d[c]["avg"])


# ══════════════════════════════════════════════════════════════════════
# Novel cluster-v3 step-mutation sweep (run each combo twice, average)
# ══════════════════════════════════════════════════════════════════════

def run_one(ctx, tree_name, dist_path, pop, prob, step):
    """Run one SHADE-cluster-v3 config TWICE (one per rep) and average.

    Cached by (tree,pop,prob,step) so reused coordinate-descent combinations
    are never re-evaluated.  Returns a dict::

        {"avg": <mean best>, "pop", "prob", "step", "reused",
         "rep1": {"best", "time_s", "file"},
         "rep2": {"best", "time_s", "file"}}
    """
    key = (tree_name, pop, prob, step)
    if key in ctx["cache"]:
        cached = dict(ctx["cache"][key])
        cached["reused"] = True
        return cached

    reps = {}
    for rep, seed in REP_SEEDS.items():
        # Override the parameters cluster_v3 reads as module globals.
        v3.PYGAD_POPULATION_SIZE = pop
        v3.MUTATION_RATE = float(prob)
        v3.STEP_SIZE = float(step)
        v3.SHADE_PAIRWISE_MUTATION = True

        # Deterministic per (combo, rep); seeds differ across reps so the two
        # runs are genuinely different draws of the same configuration.
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        print(f"\n>>> RUN[v3] tree={tree_name} pop={pop} prob={prob} "
              f"step={step} rep={rep} (seed={seed})")
        best_cost, elapsed = v3.run_single_de(
            ctx["baseline_data"], ctx["num_genes"], ctx["tls_to_genes"],
            ctx["bounds_lo"], ctx["bounds_hi"], ctx["out_dir"], rng,
            tree_name=tree_name, dist_path=str(dist_path),
        )

        # run_single_de always writes this fixed name; move it aside so the
        # next run (or rep) for the same tree does not clobber it.
        produced = (ctx["out_dir"]
                    / f"differential_evolution_cluster_v3_{tree_name}.json")
        unique = (
            ctx["out_dir"]
            / f"differential_evolution_cluster_v3_{tree_name}"
              f"_pop{pop}_mp{prob}_ms{step}_rep{rep}.json"
        )
        out_name = None
        if produced.exists():
            produced.replace(unique)
            out_name = unique.name

        reps[rep] = {"best": best_cost, "time_s": elapsed, "file": out_name}

    avg = float(np.mean([reps[r]["best"] for r in REP_SEEDS]))
    result = {"avg": avg, "pop": pop, "prob": prob, "step": step,
              "rep1": reps[1], "rep2": reps[2], "reused": False}
    ctx["cache"][key] = {k: result[k] for k in
                         ("avg", "pop", "prob", "step", "rep1", "rep2")}
    return result


def search_tree(ctx, tree_name, dist_path):
    """Run the 7-combo coordinate-descent search for one tree.

    Each combo is evaluated twice; winner selection uses the AVERAGED cost.
    """
    print(f"\n{'#'*64}\n# Tree: {tree_name}\n{'#'*64}")

    # ── Stage A — population (prob 0.1, step 1) ──────────────────────
    stage_a = {
        p: run_one(ctx, tree_name, dist_path, p, PROBS_DEFAULT, STEP_DEFAULT)
        for p in POPULATIONS
    }
    best_pop = _best(stage_a, POPULATIONS)
    print(f"\n[{tree_name}] Stage A winner: pop={best_pop} "
          f"(avg {stage_a[best_pop]['avg']:.2f})")

    # ── Stage B — mutation probability (pop=P*, step 1) ──────────────
    stage_b = {PROBS_DEFAULT: stage_a[best_pop]}  # reuse the 0.1 result
    for prob in PROBS_OTHER:
        stage_b[prob] = run_one(ctx, tree_name, dist_path, best_pop, prob,
                                STEP_DEFAULT)
    prob_order = [PROBS_DEFAULT] + PROBS_OTHER
    best_prob = _best(stage_b, prob_order)
    print(f"\n[{tree_name}] Stage B winner: prob={best_prob} "
          f"(avg {stage_b[best_prob]['avg']:.2f})")

    # ── Stage C — mutation step (pop=P*, prob=M*) ────────────────────
    # The step-1 run at (P*, M*) is already cached, so run_one returns it
    # without re-evaluating either rep.
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
          f"(avg {stage_c[best_step]['avg']:.2f})")

    final = stage_c[best_step]
    print(f"\n[{tree_name}] FINAL: pop={best_pop} prob={best_prob} "
          f"step={best_step} → avg {final['avg']:.2f} "
          f"(rep1 {final['rep1']['best']:.2f} / rep2 {final['rep2']['best']:.2f})")

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
# Novel cluster-v3 WITHOUT step-mutation (run each pop×tree twice, average)
# ══════════════════════════════════════════════════════════════════════

def run_one_nomut(ctx, tree_name, dist_path, pop):
    """Run cluster-v3 (novel bin-crossing) with step-mutation OFF, twice.

    Same walk-decomposition crossover as the mutation sweep, but
    SHADE_PAIRWISE_MUTATION=False so no step-mutation is applied.  There is no
    prob/step axis here — only the population.  run_single_de's output records
    ``shade_pairwise_mutation=false`` / ``step_size=null`` so the JSON itself
    marks these as mutation-free.  Returns {avg, rep1, rep2}.
    """
    reps = {}
    for rep, seed in REP_SEEDS.items():
        v3.PYGAD_POPULATION_SIZE = pop
        v3.SHADE_PAIRWISE_MUTATION = False

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        print(f"\n>>> RUN[v3-nomut] tree={tree_name} pop={pop} "
              f"rep={rep} (seed={seed})")
        best_cost, elapsed = v3.run_single_de(
            ctx["baseline_data"], ctx["num_genes"], ctx["tls_to_genes"],
            ctx["bounds_lo"], ctx["bounds_hi"], ctx["out_dir"], rng,
            tree_name=tree_name, dist_path=str(dist_path),
        )

        produced = (ctx["out_dir"]
                    / f"differential_evolution_cluster_v3_{tree_name}.json")
        unique = (ctx["out_dir"]
                  / f"differential_evolution_cluster_v3_nomut_{tree_name}"
                    f"_pop{pop}_rep{rep}.json")
        out_name = None
        if produced.exists():
            produced.replace(unique)
            out_name = unique.name

        reps[rep] = {"best": best_cost, "time_s": elapsed, "file": out_name}

    avg = float(np.mean([reps[r]["best"] for r in REP_SEEDS]))
    print(f"\n[v3-nomut {tree_name} pop={pop}] avg {avg:.2f} "
          f"(rep1 {reps[1]['best']:.2f} / rep2 {reps[2]['best']:.2f})")
    return {"avg": avg, "rep1": reps[1], "rep2": reps[2]}


def run_nomut_sweep(ctx, selected):
    """Sweep cluster-v3 (no mutation) over POPULATIONS for each tree, ×2 reps.

    Returns {tree: {pop_str: {avg, rep1, rep2}}}.  Per-tree dicts share the
    plain-SHADE shape, so _plain_view projects them (best pop named per view).
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
# Plain SHADE sweep over the same 3 populations (run each twice, average)
# ══════════════════════════════════════════════════════════════════════

def run_plain_de(ctx):
    """Run plain SHADE (NOVEL_MUTATION=False) over POPULATIONS, twice each.

    Plain SHADE ignores the linkage trees, so there is no tree or mutation
    axis — just the 3 populations.  Returns {pop: {avg, rep1, rep2}}.
    """
    _use_plain_crossover()
    de.NOVEL_MUTATION = False

    results = {}
    for pop in POPULATIONS:
        reps = {}
        for rep, seed in REP_SEEDS.items():
            de.PYGAD_POPULATION_SIZE = pop
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)

            print(f"\n>>> RUN[plain] pop={pop} rep={rep} (seed={seed})")
            best_cost, elapsed = de.run_single_de(
                ctx["baseline_data"], ctx["num_genes"], ctx["tls_to_genes"],
                ctx["bounds_lo"], ctx["bounds_hi"], ctx["out_dir"], rng,
            )

            produced = ctx["out_dir"] / "differential_evolution.json"
            unique = (ctx["out_dir"]
                      / f"differential_evolution_plain_pop{pop}_rep{rep}.json")
            out_name = None
            if produced.exists():
                produced.replace(unique)
                out_name = unique.name

            reps[rep] = {"best": best_cost, "time_s": elapsed, "file": out_name}

        avg = float(np.mean([reps[r]["best"] for r in REP_SEEDS]))
        results[str(pop)] = {"avg": avg, "rep1": reps[1], "rep2": reps[2]}
        print(f"\n[plain pop={pop}] avg {avg:.2f} "
              f"(rep1 {reps[1]['best']:.2f} / rep2 {reps[2]['best']:.2f})")

    return results


# ══════════════════════════════════════════════════════════════════════
# Projection: turn the rich (rep1/rep2/avg) results into per-file views
# ══════════════════════════════════════════════════════════════════════

def _combo_view(combo, view):
    """Project one combo result for a given view ('rep1', 'rep2', 'avg')."""
    base = {"pop": combo["pop"], "prob": combo["prob"], "step": combo["step"]}
    if view == "avg":
        base.update({"best": combo["avg"],
                     "best_rep1": combo["rep1"]["best"],
                     "best_rep2": combo["rep2"]["best"]})
    else:
        r = combo[view]
        base.update({"best": r["best"], "time_s": r["time_s"], "file": r["file"]})
    return base


def _tree_view(info, view):
    """Project one tree's coordinate-descent summary for a given view."""
    out = {
        "best_pop": info["best_pop"],
        "best_prob": info["best_prob"],
        "best_step": info["best_step"],
        "winner": _combo_view(info["winner"], view),
    }
    for stage in ("stage_a_population", "stage_b_probability", "stage_c_step"):
        out[stage] = {k: _combo_view(c, view) for k, c in info[stage].items()}
    return out


def _plain_view(plain, view):
    """Project the plain-SHADE results for a given view, naming the best pop.

    All 3 populations are always evaluated (no path dependence), so the winner
    is reported per-view: lowest rep-1 cost for the rep1 file, rep-2 cost for
    the rep2 file, and averaged cost for the averaged file.
    """
    by_pop = {}
    for pop, combo in plain.items():
        if view == "avg":
            by_pop[pop] = {"best": combo["avg"],
                           "best_rep1": combo["rep1"]["best"],
                           "best_rep2": combo["rep2"]["best"]}
        else:
            by_pop[pop] = dict(combo[view])
    best_pop = min(by_pop, key=lambda p: by_pop[p]["best"])
    return {"best_pop": int(best_pop), "by_pop": by_pop}


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════

def run_experiments(max_evals=None, trees=None):
    """Build the SUMO wrapper once, run both sweeps (×2 reps), write 3 files."""
    if max_evals is not None:
        v3.MAX_EVALS = int(max_evals)
        de.MAX_EVALS = int(max_evals)
        print(f"[de_experiments] MAX_EVALS overridden → {max_evals}")

    with open(v3.BASELINE_TRAFFIC_DATA, "r") as f:
        baseline_data = json.load(f)

    wrapper, num_genes, bounds_lo, bounds_hi, _ = v3.build_traffic_fitness_wrapper(
        baseline_data=baseline_data, fitness_function=v3.fitness_function,
    )
    # Both modules read their own module-level _wrapper from TLSProblem.
    v3._wrapper = wrapper
    de._wrapper = wrapper

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
            print(f"[de_experiments] WARNING: unknown trees ignored: {missing}")
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

    # ── Write the three views ────────────────────────────────────────
    def _write(view, rep_label, seed_label):
        payload = {
            "repetition": rep_label,
            "seed": seed_label,
            "max_evals": v3.MAX_EVALS,
            "grid": {
                "population": POPULATIONS,
                "mutation_probability": [PROBS_DEFAULT] + PROBS_OTHER,
                "mutation_step": [STEP_DEFAULT] + STEP_OTHER,
                "trees": list(selected.keys()),
            },
            "total_time_s": round(total, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cluster_v3_novel_mutation": {
                t: (info if "error" in info else _tree_view(info, view))
                for t, info in per_tree.items()
            },
            "cluster_v3_no_mutation": {
                t: (by_pop if "error" in by_pop else _plain_view(by_pop, view))
                for t, by_pop in nomut.items()
            },
            "plain_de_shade": (plain if "error" in plain
                               else _plain_view(plain, view)),
        }
        path = out_dir / f"de_experiments_{rep_label}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
        return path

    p1 = _write("rep1", "rep1", REP_SEEDS[1])
    p2 = _write("rep2", "rep2", REP_SEEDS[2])
    pa = _write("avg", "averaged", list(REP_SEEDS.values()))

    # ── Console summary (averaged) ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DE experiments done in {total:.1f}s")
    print(f"{'='*60}")
    print("Cluster-v3 novel mutation (averaged best):")
    print(f"{'Tree':<12}{'Pop':>6}{'Prob':>7}{'Step':>6}{'AvgBest':>12}")
    print("─" * 43)
    for tree_name, info in per_tree.items():
        if "error" in info:
            print(f"{tree_name:<12}{'ERROR':>31}")
        else:
            print(f"{tree_name:<12}{info['best_pop']:>6}"
                  f"{info['best_prob']:>7}{info['best_step']:>6}"
                  f"{info['winner']['avg']:>12.2f}")

    print("\nCluster-v3 NO mutation (averaged best per tree):")
    print(f"{'Tree':<12}{'Pop':>6}{'AvgBest':>12}")
    print("─" * 30)
    for tree_name, by_pop in nomut.items():
        if "error" in by_pop:
            print(f"{tree_name:<12}{'ERROR':>18}")
        else:
            best_pop = min(POPULATIONS, key=lambda p: by_pop[str(p)]["avg"])
            print(f"{tree_name:<12}{best_pop:>6}"
                  f"{by_pop[str(best_pop)]['avg']:>12.2f}")

    print("\nPlain SHADE (averaged best):")
    print(f"{'Pop':>6}{'AvgBest':>12}")
    print("─" * 18)
    if "error" not in plain:
        best_plain_pop = min(POPULATIONS, key=lambda p: plain[str(p)]["avg"])
        for pop in POPULATIONS:
            star = " *" if pop == best_plain_pop else ""
            print(f"{pop:>6}{plain[str(pop)]['avg']:>12.2f}{star}")
        print(f"→ best plain pop (avg): {best_plain_pop}")

    print(f"\nWrote:\n  {p1}\n  {p2}\n  {pa}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Override config.MAX_EVALS (e.g. small value for a smoke test).")
    parser.add_argument("--trees", nargs="*", default=None,
                        help="Subset of tree names for the cluster-v3 sweep (default: all).")
    args = parser.parse_args()
    run_experiments(max_evals=args.max_evals, trees=args.trees)


if __name__ == "__main__":
    main()
