"""
Single-repetition DE-SHADE run at the chosen winning configurations.

Unlike ``de_experiments.py`` (which *sweeps* the hyperparameter grid and
averages two reps per combo), this module runs each algorithm variant ONCE at a
single, pre-selected configuration.  It is meant to reproduce / showcase the
winning settings without the cost of the full coordinate-descent search.

Neither algorithm module is modified: this script imports them, overrides the
handful of parameters they read from ``config`` as module-level globals, and
calls their ``run_single_de`` once per configuration.

THREE VARIANTS
--------------
* Novel cluster-v3 WITH step-mutation + bin-crossing
  (``SHADE_PAIRWISE_MUTATION=True``).  Per-tree (pop, prob, step):
    shortest  → pop 100, prob 0.1, step 3
    euclidian → pop  50, prob 0.3, step 3
    fastest   → pop  50, prob 0.5, step 2
    random    → pop  50, prob 0.3, step 1

* Novel cluster-v3 WITHOUT step-mutation
  (``SHADE_PAIRWISE_MUTATION=False``).  Same walk-decomposition bin-crossing,
  no step-mutation, so only a per-tree population:
    shortest  → pop 50
    euclidian → pop 50
    fastest   → pop 50
    random    → pop 100

* Plain SHADE (``NOVEL_MUTATION=False``) — ignores the linkage trees:
    pop 50

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
* A summary file ``single_de_repetition.json`` collecting every run's best
  cost / time / file under the three variant sections.

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

# ── Chosen winning configurations ─────────────────────────────────────
# Cluster-v3 WITH step-mutation + bin-crossing: per-tree (pop, prob, step).
V3_MUT_CONFIG = {
    "shortest":  {"pop": 100, "prob": 0.1, "step": 3},
    "euclidian": {"pop":  50, "prob": 0.3, "step": 3},
    "fastest":   {"pop":  50, "prob": 0.5, "step": 2},
    "random":    {"pop":  50, "prob": 0.3, "step": 1},
}

# Cluster-v3 WITHOUT step-mutation: per-tree population only.
V3_NOMUT_CONFIG = {
    "shortest":  {"pop": 50},
    "euclidian": {"pop": 50},
    "fastest":   {"pop": 50},
    "random":    {"pop": 100},
}

# Plain SHADE: single population.
PLAIN_CONFIG = {"pop": 50}


def _use_v3_crossover():
    """Make the v3 cluster crossover the active SHADE crossover."""
    _shade_module.DE_binary_crossover = v3._cluster_binary_crossover


def _use_plain_crossover():
    """Restore the plain (CR-capturing) SHADE crossover."""
    _shade_module.DE_binary_crossover = de._capturing_binary_crossover


# ══════════════════════════════════════════════════════════════════════
# Variant 1: cluster-v3 WITH step-mutation + bin-crossing
# ══════════════════════════════════════════════════════════════════════

def run_v3_mut(ctx, tree_name, dist_path, cfg, seed):
    """Run cluster-v3 with step-mutation once at *cfg* = {pop, prob, step}."""
    pop, prob, step = cfg["pop"], cfg["prob"], cfg["step"]

    v3.PYGAD_POPULATION_SIZE = pop
    v3.MUTATION_RATE = float(prob)
    v3.STEP_SIZE = float(step)
    v3.SHADE_PAIRWISE_MUTATION = True

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    print(f"\n>>> RUN[v3-mut] tree={tree_name} pop={pop} prob={prob} "
          f"step={step} (seed={seed})")
    best_cost, elapsed = v3.run_single_de(
        ctx["baseline_data"], ctx["num_genes"], ctx["tls_to_genes"],
        ctx["bounds_lo"], ctx["bounds_hi"], ctx["out_dir"], rng,
        tree_name=tree_name, dist_path=str(dist_path),
    )

    # run_single_de always writes this fixed name; move it aside.
    produced = ctx["out_dir"] / f"differential_evolution_cluster_v3_{tree_name}.json"
    unique = (ctx["out_dir"]
              / f"single_de_cluster_v3_{tree_name}"
                f"_pop{pop}_mp{prob}_ms{step}.json")
    out_name = None
    if produced.exists():
        produced.replace(unique)
        out_name = unique.name

    return {"pop": pop, "prob": prob, "step": step,
            "best": best_cost, "time_s": elapsed, "file": out_name}


# ══════════════════════════════════════════════════════════════════════
# Variant 2: cluster-v3 WITHOUT step-mutation
# ══════════════════════════════════════════════════════════════════════

def run_v3_nomut(ctx, tree_name, dist_path, cfg, seed):
    """Run cluster-v3 bin-crossing with step-mutation OFF, once at *cfg*."""
    pop = cfg["pop"]

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

    produced = ctx["out_dir"] / f"differential_evolution_cluster_v3_{tree_name}.json"
    unique = (ctx["out_dir"]
              / f"single_de_cluster_v3_nomut_{tree_name}_pop{pop}.json")
    out_name = None
    if produced.exists():
        produced.replace(unique)
        out_name = unique.name

    return {"pop": pop, "best": best_cost, "time_s": elapsed, "file": out_name}


# ══════════════════════════════════════════════════════════════════════
# Variant 3: plain SHADE
# ══════════════════════════════════════════════════════════════════════

def run_plain(ctx, cfg, seed):
    """Run plain SHADE (NOVEL_MUTATION=False) once at *cfg* = {pop}."""
    _use_plain_crossover()
    de.NOVEL_MUTATION = False

    pop = cfg["pop"]
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

    return {"pop": pop, "best": best_cost, "time_s": elapsed, "file": out_name}


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════

def run_experiments(max_evals=None, trees=None, seed=DEFAULT_SEED):
    """Build the SUMO wrapper once, run all three variants, write a summary."""
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

    # ── Variant 1: cluster-v3 WITH step-mutation + bin-crossing ──────
    _use_v3_crossover()
    v3_mut = {}
    for tree_name, dist_path in selected.items():
        cfg = V3_MUT_CONFIG.get(tree_name)
        if cfg is None:
            print(f"[v3-mut] no config for tree {tree_name}; skipping")
            continue
        try:
            v3_mut[tree_name] = run_v3_mut(ctx, tree_name, dist_path, cfg, seed)
        except Exception as e:  # keep going; record the failure
            print(f"ERROR [v3-mut:{tree_name}]: {e}")
            import traceback
            traceback.print_exc()
            v3_mut[tree_name] = {"error": str(e)}

    # ── Variant 2: cluster-v3 WITHOUT step-mutation ──────────────────
    _use_v3_crossover()
    v3_nomut = {}
    for tree_name, dist_path in selected.items():
        cfg = V3_NOMUT_CONFIG.get(tree_name)
        if cfg is None:
            print(f"[v3-nomut] no config for tree {tree_name}; skipping")
            continue
        try:
            v3_nomut[tree_name] = run_v3_nomut(ctx, tree_name, dist_path, cfg, seed)
        except Exception as e:
            print(f"ERROR [v3-nomut:{tree_name}]: {e}")
            import traceback
            traceback.print_exc()
            v3_nomut[tree_name] = {"error": str(e)}

    # ── Variant 3: plain SHADE ───────────────────────────────────────
    try:
        plain = run_plain(ctx, PLAIN_CONFIG, seed)
    except Exception as e:
        print(f"ERROR [plain]: {e}")
        import traceback
        traceback.print_exc()
        plain = {"error": str(e)}

    total = time.time() - t_start

    # ── Write the summary ────────────────────────────────────────────
    payload = {
        "variant": "single_de_repetition",
        "seed": seed,
        "max_evals": v3.MAX_EVALS,
        "trees": list(selected.keys()),
        "total_time_s": round(total, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cluster_v3_novel_mutation": v3_mut,
        "cluster_v3_no_mutation": v3_nomut,
        "plain_de_shade": plain,
    }
    summary_path = out_dir / "single_de_repetition.json"
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=4)

    # ── Console summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"single_de_repetition done in {total:.1f}s")
    print(f"{'='*60}")

    print("Cluster-v3 WITH step-mutation:")
    print(f"{'Tree':<12}{'Pop':>6}{'Prob':>7}{'Step':>6}{'Best':>12}")
    print("─" * 43)
    for tree_name, info in v3_mut.items():
        if "error" in info:
            print(f"{tree_name:<12}{'ERROR':>31}")
        else:
            print(f"{tree_name:<12}{info['pop']:>6}{info['prob']:>7}"
                  f"{info['step']:>6}{info['best']:>12.2f}")

    print("\nCluster-v3 WITHOUT step-mutation:")
    print(f"{'Tree':<12}{'Pop':>6}{'Best':>12}")
    print("─" * 30)
    for tree_name, info in v3_nomut.items():
        if "error" in info:
            print(f"{tree_name:<12}{'ERROR':>18}")
        else:
            print(f"{tree_name:<12}{info['pop']:>6}{info['best']:>12.2f}")

    print("\nPlain SHADE:")
    print(f"{'Pop':>6}{'Best':>12}")
    print("─" * 18)
    if "error" not in plain:
        print(f"{plain['pop']:>6}{plain['best']:>12.2f}")

    print(f"\nWrote summary → {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Override config.MAX_EVALS (e.g. small value for a smoke test).")
    parser.add_argument("--trees", nargs="*", default=None,
                        help="Subset of tree names for the cluster-v3 variants (default: all).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed for the single repetition (default: {DEFAULT_SEED}).")
    args = parser.parse_args()
    run_experiments(max_evals=args.max_evals, trees=args.trees, seed=args.seed)


if __name__ == "__main__":
    main()
