"""
Best-configuration experiments across instances — EXTRA REPETITIONS (rep4 + rep5).

This is a duplicate of ``best_config_instance_experiments.py`` that runs TWO MORE
repetitions (seeds 45 and 46) of the same frozen best configurations, so the
per-config averages can be extended beyond the original three reps.  Outputs are
written alongside the originals without clobbering them (see REPETITIONS / OUTPUT).

Best-configuration experiments across instances (Jakarta + Beijing + Kota Kinabalu).

Unlike ``de_experiments.py`` (which *searches* for good hyper-parameters on the
Jakarta instance), this script takes the BEST configurations already found by
that search — frozen and hard-coded below — and re-runs them on the OTHER
instances so we can compare how the three algorithm variants transfer:

  1. SHADE + cluster-v3 bin-crossing + step-mutation   (``cluster_v3_mut``)
  2. SHADE + cluster-v3 bin-crossing, NO step-mutation (``cluster_v3_nomut``)
  3. Plain SHADE (no clustering, no mutation)          (``plain``)

For the two cluster variants we keep the per-tree winners (best config found for
each of the 4 linkage trees), so each instance runs every tree at its own best
hyper-parameters.  Plain SHADE has no tree, just its single best population.

INSTANCE SWITCHING
------------------
``config`` hard-wires SUMO at the Jakarta network (``src/sumo_setup``).  The
``extraction`` and ``fitness_evaluation`` modules read ``SUMO_ARGS`` (and the
latter ``BASELINE_DATA``) as module globals, so we switch instances by pointing
those globals at the per-instance ``osm.sumocfg`` / baseline before each run.

Beijing and Kota Kinabalu ship only SUMO network files — no baseline JSON and no
distance trees.  Both are GENERATED on first use and cached under
``src/outputs/instances/<instance>/`` so reruns skip the heavy SUMO/sumolib work:

* ``baseline_traffic_data.json`` — extracted from the instance network and
  filtered to ``OPTIMIZE_PHASE_COUNTS`` (mirrors ``generation.generate_data``).
* ``tls_distances_<tree>.json``  — all 4 linkage trees, built with sumolib over
  that instance's network (mirrors ``tls_distances.build_distance_matrices``).

REPETITIONS
-----------
Every configuration is stochastic, so each is run TWO more times with distinct
seeds (rep 4 → 45, rep 5 → 46), continuing the sequence after the original
three reps (seeds 42/43/44), and the two final ``best_fitness`` values are
averaged.

OUTPUT (per instance, in src/outputs/best_config_rep4_rep5/<instance>/)
----------------------------------------------------------------------
All results live in their own folder, separate from the original rep1–3 outputs.
The baseline + distance-tree caches are still READ from
src/outputs/instances/<instance>/ so they are not regenerated.

* ``best_config_experiments_rep4_rep5.json`` — every variant/tree with per-rep
  best cost, time and file, plus the averaged best.
* Per-run algorithm JSONs are renamed uniquely (``<instance>_..._repN.json``,
  N ∈ {4, 5}) — one file per rep, so rep4 and rep5 are kept separately.

Usage:
  python -m src.experiments.best_config_instance_experiments_rep3_rep4
  python -m src.experiments.best_config_instance_experiments_rep3_rep4 --instances beijing
  python -m src.experiments.best_config_instance_experiments_rep3_rep4 --max-evals 200
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

import torch  # noqa: E402  (after sys.path setup, mirrors the algorithm modules)

import config  # noqa: E402

# ``fitness_evaluation`` opens config.BASELINE_TRAFFIC_DATA at IMPORT time
# (module-level ``BASELINE_DATA = json.load(...)``), so the file must exist
# before we import the algorithm modules below.  On a fresh checkout the Jakarta
# baseline is absent — write a harmless empty stub if so.  Its contents are
# never used: we override ``fe.BASELINE_DATA`` with each instance's real
# baseline (see activate_instance) before any evaluation runs.  Guarded on
# non-existence so a real Jakarta baseline is never clobbered.
_jakarta_baseline = Path(config.BASELINE_TRAFFIC_DATA)
if not _jakarta_baseline.exists():
    _jakarta_baseline.parent.mkdir(parents=True, exist_ok=True)
    with open(_jakarta_baseline, "w") as _f:
        json.dump({"tls_data": {}, "fitness": 0}, _f)
    print("[best_config_instance_experiments] wrote import-time baseline stub "
          f"→ {_jakarta_baseline} (replaced per-instance at runtime)")

# Both algorithm modules monkeypatch the SAME symbol
# (``_shade_module.DE_binary_crossover``) at import time, so whichever is
# imported LAST wins globally.  We re-assert the correct crossover before each
# section below (see _use_v3_crossover / _use_plain_crossover).
import src.algorithms.differential_evolution_cluster_v3 as v3  # noqa: E402
import src.algorithms.differential_evolution as de  # noqa: E402
from evox.algorithms.so.de_variants import shade as _shade_module  # noqa: E402

# These two modules read SUMO_ARGS / BASELINE_DATA as module globals at call
# time, so re-pointing those globals is what switches the active instance.
import src.sumo_setup.extraction as extraction  # noqa: E402
import src.sumo_setup.fitness_evaluation as fe  # noqa: E402
import src.plot.tls_distances as td  # noqa: E402
from src.novel.distance_trees import distance_tree_paths  # noqa: E402

# ── Instances to optimise (folder under src/ holding each one's SUMO files) ──
# Jakarta is the reference instance these best configs were *discovered* on, so
# re-running it here is circular; it is included only so it CAN be run on demand
# (e.g. ``--instances jakarta``).  Like every other instance, it generates and
# caches its own baseline + distance trees under src/outputs/instances/jakarta/
# rather than reusing the canonical src/outputs/baseline_traffic_data.json.
INSTANCES = {
    "jakarta": ROOT / "src" / "sumo_setup",
    "beijing": ROOT / "src" / "sumo_setup_beijing",
    "kotakinabalu": ROOT / "src" / "sumo_setup_kotakinabalu",
}

# ── Repetition seeds — TWO EXTRA reps continuing after the original 1–3 ──
REP_SEEDS = {4: 45, 5: 46}

# ── Dedicated results folder ─────────────────────────────────────────────
# ALL rep4/rep5 outputs (per-rep JSONs + the summary) are written here, one
# subfolder per instance, isolated from the original rep1–3 outputs in
# src/outputs/instances/<instance>/.  The baseline + distance-tree CACHES are
# still read from that original folder so we don't regenerate them.
RESULTS_DIR = ROOT / "src" / "outputs" / "best_config_rep4_rep5"

# ══════════════════════════════════════════════════════════════════════
# Hard-coded BEST configurations (from
# src/outputs/best_parameter/best_configuration_summary.json :: headline_winners)
# ══════════════════════════════════════════════════════════════════════

# SHADE + cluster-v3 bin-crossing + step-mutation: best (pop, mp, ms) per tree.
CLUSTER_MUT_BEST = {
    "euclidian": {"pop": 50,  "mp": 0.3, "ms": 3.0},
    "fastest":   {"pop": 50,  "mp": 0.5, "ms": 2.0},
    "random":    {"pop": 50,  "mp": 0.3, "ms": 1.0},
    "shortest":  {"pop": 100, "mp": 0.1, "ms": 1.0},
}

# SHADE + cluster-v3 bin-crossing, NO step-mutation: best population per tree.
CLUSTER_NOMUT_BEST = {
    "euclidian": {"pop": 50},
    "fastest":   {"pop": 50},
    "random":    {"pop": 200},
    "shortest":  {"pop": 50},
}

# Plain SHADE: single best population (no tree, no mutation).
PLAIN_BEST = {"pop": 100}


def _use_v3_crossover():
    """Make the v3 cluster crossover the active SHADE crossover."""
    _shade_module.DE_binary_crossover = v3._cluster_binary_crossover


def _use_plain_crossover():
    """Restore the plain (CR-capturing) SHADE crossover."""
    _shade_module.DE_binary_crossover = de._capturing_binary_crossover


# ══════════════════════════════════════════════════════════════════════
# Instance switching + on-demand (cached) instance-data generation
# ══════════════════════════════════════════════════════════════════════

def _instance_sumo_args(instance_dir):
    """config.SUMO_ARGS with the ``-c`` config file swapped to this instance."""
    args = list(config.SUMO_ARGS)
    i = args.index("-c")
    args[i + 1] = str(instance_dir / "osm.sumocfg")
    return args


def ensure_baseline(instance_dir, inst_out):
    """Load (or extract + cache) this instance's filtered baseline traffic data.

    Mirrors ``generation.generate_data``: extract via SUMO, then drop TLSs whose
    phase count is outside ``OPTIMIZE_PHASE_COUNTS`` so every downstream consumer
    sees the same filtered set.  Cached to ``baseline_traffic_data.json``.
    """
    path = inst_out / "baseline_traffic_data.json"
    if path.exists():
        print(f"  [baseline] cache hit → {path.relative_to(ROOT)}")
        with open(path) as f:
            return json.load(f)

    print(f"  [baseline] extracting from {instance_dir.name} network …")
    extraction.SUMO_ARGS = _instance_sumo_args(instance_dir)
    data = extraction.extract_traffic_light_data(detail=False)

    excluded = [tid for tid, phases in data["tls_data"].items()
                if len(phases) not in config.OPTIMIZE_PHASE_COUNTS]
    for tid in excluded:
        del data["tls_data"][tid]
    if excluded:
        print(f"  [baseline] excluded {len(excluded)} TLS(s) outside "
              f"{sorted(config.OPTIMIZE_PHASE_COUNTS)} phase counts")
    print(f"  [baseline] kept {len(data['tls_data'])} TLS(s)")

    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    return data


def ensure_distance_trees(instance_dir, baseline_data, inst_out):
    """Load (or build + cache) all 4 linkage-tree distance JSONs for this instance.

    Mirrors ``tls_distances.build_distance_matrices`` but against this instance's
    network, restricted to the same filtered TLS set as the baseline.  Returns
    the ``{tree_name: Path}`` map the cluster optimisers consume.
    """
    paths = distance_tree_paths(inst_out)
    if all(p.exists() for p in paths.values()):
        print(f"  [trees] cache hit → {inst_out.relative_to(ROOT)}")
        return paths

    net_file = instance_dir / "osm.net.xml.gz"
    print(f"  [trees] loading network {net_file.relative_to(ROOT)} …")
    net = td.sumolib.net.readNet(str(net_file))

    baseline_ids = set(baseline_data["tls_data"].keys())
    nodes = [n for n in net.getNodes()
             if n.getType() == "traffic_light" and n.getID() in baseline_ids]
    print(f"  [trees] baseline lists {len(baseline_ids)} TLS(s); "
          f"{len(nodes)} matched in the network")
    tls_data = [
        {
            "id": n.getID(),
            "lon": net.convertXY2LonLat(*n.getCoord())[0],
            "lat": net.convertXY2LonLat(*n.getCoord())[1],
        }
        for n in nodes
    ]

    # Seed so the "random" tree variant is reproducible (matches the plot tool).
    random.seed(config.SEED_BASE)
    state = td.compute_all(net, nodes)
    for name, cfg in td.VARIANTS.items():
        td.save_variant(name, cfg, tls_data, state[name], inst_out)
    return paths


def activate_instance(instance_dir, baseline_data):
    """Point the fitness evaluator at this instance (SUMO args + baseline)."""
    fe.SUMO_ARGS = _instance_sumo_args(instance_dir)
    fe.BASELINE_DATA = baseline_data


# ══════════════════════════════════════════════════════════════════════
# Single-config runners (each runs one rep per REP_SEEDS and averages)
# ══════════════════════════════════════════════════════════════════════

def run_cluster(ctx, tree_name, dist_path, pop, mutation, mp=None, ms=None):
    """Run one cluster-v3 config (mutation on/off) once per REP_SEEDS and average.

    ``mutation`` toggles SHADE_PAIRWISE_MUTATION; when on, ``mp``/``ms`` set the
    mutation probability/step.  Returns ``{avg, ..., repN}`` (one ``repN`` key
    per seed in REP_SEEDS) where each rep is ``{best, time_s, file}``.
    """
    tag = "mut" if mutation else "nomut"
    reps = {}
    for rep, seed in REP_SEEDS.items():
        v3.PYGAD_POPULATION_SIZE = pop
        v3.SHADE_PAIRWISE_MUTATION = mutation
        if mutation:
            v3.MUTATION_RATE = float(mp)
            v3.STEP_SIZE = float(ms)

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        extra = f" mp={mp} ms={ms}" if mutation else ""
        print(f"\n>>> RUN[{ctx['instance']}/v3-{tag}] tree={tree_name} "
              f"pop={pop}{extra} rep={rep} (seed={seed})")
        best_cost, elapsed = v3.run_single_de(
            ctx["baseline_data"], ctx["num_genes"], ctx["tls_to_genes"],
            ctx["bounds_lo"], ctx["bounds_hi"], ctx["out_dir"], rng,
            tree_name=tree_name, dist_path=str(dist_path),
        )

        # run_single_de always writes this fixed name; move it aside so the
        # next rep / tree / variant does not clobber it.
        produced = (ctx["out_dir"]
                    / f"differential_evolution_cluster_v3_{tree_name}.json")
        suffix = f"_mp{mp}_ms{int(ms)}" if mutation else ""
        unique = (ctx["out_dir"]
                  / f"{ctx['instance']}_cluster_v3_{tag}_{tree_name}"
                    f"_pop{pop}{suffix}_rep{rep}.json")
        out_name = None
        if produced.exists():
            produced.replace(unique)
            out_name = unique.name

        reps[rep] = {"best": best_cost, "time_s": elapsed, "file": out_name}

    avg = float(np.mean([reps[r]["best"] for r in REP_SEEDS]))
    rep_str = " / ".join(f"rep{r} {reps[r]['best']:.2f}" for r in REP_SEEDS)
    print(f"\n[{ctx['instance']}/v3-{tag} {tree_name}] avg {avg:.2f} ({rep_str})")
    result = {"avg": avg, "pop": pop, "mp": mp, "ms": ms}
    result.update({f"rep{r}": reps[r] for r in REP_SEEDS})
    return result


def run_plain(ctx, pop):
    """Run plain SHADE (NOVEL_MUTATION=False) once per REP_SEEDS and average."""
    _use_plain_crossover()
    de.NOVEL_MUTATION = False

    reps = {}
    for rep, seed in REP_SEEDS.items():
        de.PYGAD_POPULATION_SIZE = pop
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        print(f"\n>>> RUN[{ctx['instance']}/plain] pop={pop} "
              f"rep={rep} (seed={seed})")
        best_cost, elapsed = de.run_single_de(
            ctx["baseline_data"], ctx["num_genes"], ctx["tls_to_genes"],
            ctx["bounds_lo"], ctx["bounds_hi"], ctx["out_dir"], rng,
        )

        produced = ctx["out_dir"] / "differential_evolution.json"
        unique = (ctx["out_dir"]
                  / f"{ctx['instance']}_plain_pop{pop}_rep{rep}.json")
        out_name = None
        if produced.exists():
            produced.replace(unique)
            out_name = unique.name

        reps[rep] = {"best": best_cost, "time_s": elapsed, "file": out_name}

    avg = float(np.mean([reps[r]["best"] for r in REP_SEEDS]))
    rep_str = " / ".join(f"rep{r} {reps[r]['best']:.2f}" for r in REP_SEEDS)
    print(f"\n[{ctx['instance']}/plain pop={pop}] avg {avg:.2f} ({rep_str})")
    result = {"avg": avg, "pop": pop}
    result.update({f"rep{r}": reps[r] for r in REP_SEEDS})
    return result


# ══════════════════════════════════════════════════════════════════════
# Per-instance driver
# ══════════════════════════════════════════════════════════════════════

def run_instance(instance, instance_dir):
    """Generate (cached) instance data, then run all 3 variants ×2 reps."""
    print(f"\n{'#'*64}\n# Instance: {instance}\n{'#'*64}")
    # Caches (baseline + distance trees) live in the ORIGINAL instance folder so
    # we reuse them instead of regenerating; results go to the dedicated folder.
    inst_cache = ROOT / "src" / "outputs" / "instances" / instance
    inst_cache.mkdir(parents=True, exist_ok=True)
    inst_out = RESULTS_DIR / instance
    inst_out.mkdir(parents=True, exist_ok=True)

    baseline_data = ensure_baseline(instance_dir, inst_cache)
    activate_instance(instance_dir, baseline_data)
    trees = ensure_distance_trees(instance_dir, baseline_data, inst_cache)

    # Build the SUMO fitness wrapper from THIS instance's baseline.  Both
    # algorithm modules read their own module-level _wrapper from TLSProblem.
    wrapper, num_genes, bounds_lo, bounds_hi, _ = v3.build_traffic_fitness_wrapper(
        baseline_data=baseline_data, fitness_function=v3.fitness_function,
    )
    v3._wrapper = wrapper
    de._wrapper = wrapper

    tls_to_genes, _, _ = v3.build_gene_map(baseline_data)

    ctx = {
        "instance": instance,
        "baseline_data": baseline_data,
        "num_genes": num_genes,
        "tls_to_genes": tls_to_genes,
        "bounds_lo": bounds_lo,
        "bounds_hi": bounds_hi,
        "out_dir": inst_out,
    }

    t_start = time.time()

    # ── Section 1: cluster-v3 WITH step-mutation (per-tree best) ──────
    _use_v3_crossover()
    mut = {}
    for tree_name, p in CLUSTER_MUT_BEST.items():
        try:
            mut[tree_name] = run_cluster(ctx, tree_name, trees[tree_name],
                                         pop=p["pop"], mutation=True,
                                         mp=p["mp"], ms=p["ms"])
        except Exception as e:
            print(f"ERROR [{instance}/v3-mut:{tree_name}]: {e}")
            import traceback
            traceback.print_exc()
            mut[tree_name] = {"error": str(e)}

    # ── Section 2: cluster-v3 WITHOUT step-mutation (per-tree best) ───
    _use_v3_crossover()
    nomut = {}
    for tree_name, p in CLUSTER_NOMUT_BEST.items():
        try:
            nomut[tree_name] = run_cluster(ctx, tree_name, trees[tree_name],
                                           pop=p["pop"], mutation=False)
        except Exception as e:
            print(f"ERROR [{instance}/v3-nomut:{tree_name}]: {e}")
            import traceback
            traceback.print_exc()
            nomut[tree_name] = {"error": str(e)}

    # ── Section 3: plain SHADE (single best population) ───────────────
    try:
        plain = run_plain(ctx, PLAIN_BEST["pop"])
    except Exception as e:
        print(f"ERROR [{instance}/plain]: {e}")
        import traceback
        traceback.print_exc()
        plain = {"error": str(e)}

    total = time.time() - t_start

    payload = {
        "instance": instance,
        "instance_dir": str(instance_dir.relative_to(ROOT)),
        "seeds": list(REP_SEEDS.values()),
        "max_evals": v3.MAX_EVALS,
        "best_configs": {
            "cluster_v3_mut": CLUSTER_MUT_BEST,
            "cluster_v3_nomut": CLUSTER_NOMUT_BEST,
            "plain": PLAIN_BEST,
        },
        "total_time_s": round(total, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": {
            "cluster_v3_mut": mut,
            "cluster_v3_nomut": nomut,
            "plain": plain,
        },
    }
    out_path = inst_out / "best_config_experiments_rep4_rep5.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)

    _print_instance_summary(instance, mut, nomut, plain, total)
    print(f"\nWrote: {out_path}")
    return payload


def _print_instance_summary(instance, mut, nomut, plain, total):
    """Console table of averaged best costs for one instance."""
    print(f"\n{'='*60}")
    print(f"[{instance}] done in {total:.1f}s")
    print(f"{'='*60}")

    print("Cluster-v3 WITH step-mutation (averaged best):")
    print(f"{'Tree':<12}{'Pop':>6}{'Prob':>7}{'Step':>6}{'AvgBest':>12}")
    print("─" * 43)
    for tree_name, info in mut.items():
        if "error" in info:
            print(f"{tree_name:<12}{'ERROR':>31}")
        else:
            print(f"{tree_name:<12}{info['pop']:>6}{info['mp']:>7}"
                  f"{info['ms']:>6.0f}{info['avg']:>12.2f}")

    print("\nCluster-v3 NO step-mutation (averaged best):")
    print(f"{'Tree':<12}{'Pop':>6}{'AvgBest':>12}")
    print("─" * 30)
    for tree_name, info in nomut.items():
        if "error" in info:
            print(f"{tree_name:<12}{'ERROR':>18}")
        else:
            print(f"{tree_name:<12}{info['pop']:>6}{info['avg']:>12.2f}")

    print("\nPlain SHADE (averaged best):")
    print(f"{'Pop':>6}{'AvgBest':>12}")
    print("─" * 18)
    if "error" not in plain:
        print(f"{plain['pop']:>6}{plain['avg']:>12.2f}")
    else:
        print(f"{'ERROR':>18}")


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

def run_experiments(instances=None, max_evals=None):
    if max_evals is not None:
        v3.MAX_EVALS = int(max_evals)
        de.MAX_EVALS = int(max_evals)
        print(f"[best_config_instance_experiments] MAX_EVALS → {max_evals}")

    selected = instances or list(INSTANCES.keys())
    unknown = [i for i in selected if i not in INSTANCES]
    if unknown:
        print(f"WARNING: unknown instances ignored: {unknown}")
    selected = [i for i in selected if i in INSTANCES]

    t0 = time.time()
    for instance in selected:
        try:
            run_instance(instance, INSTANCES[instance])
        except Exception as e:
            print(f"ERROR [instance {instance}]: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}\nAll instances done in {time.time() - t0:.1f}s\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="*", default=None,
                        help=f"Subset of instances to run "
                             f"(default: {list(INSTANCES.keys())}).")
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Override config.MAX_EVALS (e.g. small value for a smoke test).")
    args = parser.parse_args()
    run_experiments(instances=args.instances, max_evals=args.max_evals)


if __name__ == "__main__":
    main()
