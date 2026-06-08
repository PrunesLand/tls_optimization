"""
Random Search experiments — run Random Search ``N`` times per instance and
average the repetitions.

WHY REPETITIONS
---------------
Random Search draws its whole population at random, so a single run is a noisy
point estimate of how good the method is on an instance.  We therefore repeat
it ``N`` times (default 5) with *distinct* seeds and average the per-run best
fitness.  Reusing one seed would make every repetition byte-identical and the
average meaningless, so the seeds must differ (rep i → SEED_BASE + (i-1)).

HOW IT WORKS
------------
The algorithm module ``src.algorithms.random_search`` is left untouched except
for two env hooks it already honours:
  * ``TLS_RS_SEED``    — RNG seed for the initial population.
  * ``TLS_RS_OUTFILE`` — basename to write instead of ``random_search.json``.
Instance switching reuses the same mechanism as ``baseline_experiments``: the
algorithm reads its active SUMO network + baseline traffic data from ``config``,
which honours ``TLS_SUMOCFG`` / ``TLS_BASELINE_DATA``.  Each repetition runs in
its own subprocess (so its *spawned* multiprocessing workers re-import ``config``
and inherit the same instance + seed), and the module already routes its output
into the active instance's ``<out_dir>/random_search`` folder.

OUTPUT FILES — all under one dedicated folder:
``src/outputs/random_search_experiments/<instance>/``
-----------------------------------------------------
* ``random_search_rep1.json`` … ``random_search_repN.json`` — the full result
  of each repetition (evaluations, best configuration, best fitness, time).
* ``random_search_summary.json`` — per-instance summary over the N reps: the
  per-rep best fitness / time, plus the average (and std / min / max) of the
  best fitness and the average run time.

Usage:
  python -m src.experiments.random_search_experiments                      # all instances, 5 reps
  python -m src.experiments.random_search_experiments --instances jakarta beijing
  python -m src.experiments.random_search_experiments --reps 3
  python -m src.experiments.random_search_experiments --instances jakarta --max-evals 50   # smoke test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from config import INSTANCES, SEED_BASE  # noqa: E402

RS_MODULE = "src.algorithms.random_search"

# All repetitions + summaries land here, one subfolder per instance, so the
# results live in their own dedicated folder inside src/outputs.
RESULTS_ROOT = ROOT / "src" / "outputs" / "random_search_experiments"


def _instance_dir(instance):
    return RESULTS_ROOT / instance


def run_repetition(instance, spec, rep, seed, max_evals):
    """Run one Random Search repetition for ``instance`` in its own subprocess.

    Returns the parsed result dict for the produced rep file, or ``None`` on
    failure.
    """
    rs_dir = _instance_dir(instance)
    rs_dir.mkdir(parents=True, exist_ok=True)
    outfile = f"random_search_rep{rep}.json"

    env = os.environ.copy()
    env["TLS_SUMOCFG"] = str(spec["sumocfg"])
    env["TLS_BASELINE_DATA"] = str(spec["baseline_data"])
    env["TLS_RS_SEED"] = str(seed)
    env["TLS_RS_OUTFILE"] = outfile
    env["TLS_RS_OUTDIR"] = str(rs_dir)
    if max_evals is not None:
        # config reads MAX_EVALS at import; override it for every (re-importing)
        # worker via this env hook so smoke tests stay fast.
        env["TLS_RS_MAX_EVALS"] = str(max_evals)

    print(f"\n{'='*60}")
    print(f"# [{instance}] Random Search rep {rep} (seed={seed})")
    print(f"{'='*60}\n")

    t0 = time.time()
    ret = subprocess.run(
        [sys.executable, "-m", RS_MODULE], cwd=str(ROOT), env=env,
    ).returncode
    elapsed = time.time() - t0

    out_path = rs_dir / outfile
    if ret != 0 or not out_path.exists():
        print(f">>> [{instance}] rep {rep} FAILED (ret={ret})")
        return None

    with open(out_path) as fh:
        result = json.load(fh)
    print(f">>> [{instance}] rep {rep} done in {elapsed:.1f}s "
          f"→ best {result.get('best_fitness'):.2f} → {out_path.name}")
    return result


def summarize(instance, reps, rep_results, seeds, max_evals):
    """Build and write the per-instance summary over the completed reps."""
    rs_dir = _instance_dir(instance)

    per_rep = []
    for rep, res in rep_results.items():
        per_rep.append({
            "rep": rep,
            "seed": seeds[rep],
            "best_fitness": res["best_fitness"],
            "total_time_s": res["total_time_s"],
            "file": f"random_search_rep{rep}.json",
        })
    per_rep.sort(key=lambda r: r["rep"])

    bests = np.array([r["best_fitness"] for r in per_rep], dtype=float)
    times = np.array([r["total_time_s"] for r in per_rep], dtype=float)

    summary = {
        "instance": instance,
        "algorithm": "random_search",
        "repetitions_requested": reps,
        "repetitions_completed": len(per_rep),
        "seeds": [seeds[r["rep"]] for r in per_rep],
        "max_evals": max_evals if max_evals is not None else rep_results[
            per_rep[0]["rep"]].get("MAX_EVALS"),
        "per_repetition": per_rep,
        "average_best_fitness": float(bests.mean()) if bests.size else None,
        "std_best_fitness": float(bests.std(ddof=0)) if bests.size else None,
        "min_best_fitness": float(bests.min()) if bests.size else None,
        "max_best_fitness": float(bests.max()) if bests.size else None,
        "average_time_s": float(times.mean()) if times.size else None,
        "total_time_s": float(times.sum()) if times.size else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = rs_dir / "random_search_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary, out_path


def run_instance(instance, reps, max_evals):
    """Run ``reps`` repetitions for one instance and write its summary."""
    spec = INSTANCES[instance]
    if not spec["baseline_data"].exists():
        print(f"\n!! [{instance}] baseline traffic data missing:\n"
              f"   {spec['baseline_data']}\n"
              f"   Generate it first (best_config_instance_experiments) and re-run.")
        return None

    print(f"\n{'#'*64}")
    print(f"# Instance: {instance}  ({reps} repetitions)")
    print(f"#   sumocfg : {spec['sumocfg'].relative_to(ROOT)}")
    print(f"#   baseline: {spec['baseline_data'].relative_to(ROOT)}")
    print(f"#   outputs : {_instance_dir(instance).relative_to(ROOT)}")
    print(f"{'#'*64}")

    seeds = {rep: SEED_BASE + (rep - 1) for rep in range(1, reps + 1)}
    rep_results = {}
    for rep in range(1, reps + 1):
        res = run_repetition(instance, spec, rep, seeds[rep], max_evals)
        if res is not None:
            rep_results[rep] = res

    if not rep_results:
        print(f"\n!! [{instance}] no repetition succeeded — skipping summary.")
        return None

    summary, out_path = summarize(
        instance, reps, rep_results, seeds, max_evals)
    print(f"\n>>> [{instance}] summary ({summary['repetitions_completed']}/"
          f"{reps} reps): avg best {summary['average_best_fitness']:.2f} "
          f"(std {summary['std_best_fitness']:.2f}) → {out_path}")
    return summary


def run_all(instances, reps, max_evals):
    """Run every selected instance and print a final cross-instance table."""
    t_start = time.time()
    summaries = {}
    for instance in instances:
        summaries[instance] = run_instance(instance, reps, max_evals)
    total = time.time() - t_start

    print(f"\n{'='*64}")
    print(f"Random Search experiments done in {total:.1f}s")
    print(f"{'='*64}")
    print(f"{'Instance':<16}{'Reps':>6}{'AvgBest':>14}{'Std':>12}{'AvgTime':>11}")
    print("─" * 59)
    for instance, s in summaries.items():
        if s is None:
            print(f"{instance:<16}{'(skipped)':>6}")
            continue
        print(f"{instance:<16}{s['repetitions_completed']:>6}"
              f"{s['average_best_fitness']:>14.2f}"
              f"{s['std_best_fitness']:>12.2f}"
              f"{s['average_time_s']:>10.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", nargs="*", default=list(INSTANCES),
                        help=f"Instances to run (default: all). "
                             f"Known: {list(INSTANCES)}.")
    parser.add_argument("--reps", type=int, default=5,
                        help="Repetitions per instance (default: 5).")
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Override config.MAX_EVALS (small value = smoke test).")
    args = parser.parse_args()

    unknown = [i for i in args.instances if i not in INSTANCES]
    if unknown:
        print(f"WARNING: unknown instances ignored: {unknown} "
              f"(known: {list(INSTANCES)})")
    instances = [i for i in args.instances if i in INSTANCES]
    if not instances:
        print("No valid instances selected — nothing to do.")
        return

    run_all(instances, args.reps, args.max_evals)


if __name__ == "__main__":
    main()
