"""
Run all baseline / algorithm experiments sequentially, for one or more instances.

Runs each algorithm one after another, each in its own subprocess, while
keeping every algorithm module completely untouched.  Each subprocess
writes its own individual result files to src/outputs/ exactly as it does
when run standalone, so per-algorithm results are preserved.

Algorithms run (each as `python -m <module>`, in order):
  - Random Search             (src.algorithms.random_search)
  - Baseline evaluation       (src.algorithms.evaluate_baseline)  ← current SUMO config
  - Simple Genetic Algorithm  (src.algorithms.simple_genetic_algorithm)

Note: plain DE-SHADE (and the novel cluster-v3 step-mutation sweep) now live
      in src.experiments.de_experiments, which runs them twice and averages;
      they are intentionally NOT launched from this baseline runner.

INSTANCE SWITCHING
------------------
Each algorithm module reads the active SUMO network + baseline traffic data from
``config`` (``CONFIG_FILE`` / ``BASELINE_TRAFFIC_DATA``).  ``config`` honours the
``TLS_SUMOCFG`` and ``TLS_BASELINE_DATA`` environment variables, so this runner
switches instances purely by exporting those before launching each subprocess.
Because the modules' multiprocessing workers are *spawned* (macOS default), they
re-import ``config`` and inherit the same env, so every worker evaluates the
correct instance.

The algorithm modules always write their JSON to ``src/outputs/``.  For non-
Jakarta instances this runner moves the freshly produced files into
``src/outputs/instances/<instance>/`` and restores any Jakarta file the run
overwrote, so per-instance results never clobber each other.

Each subprocess inherits this process's stdout/stderr, so output streams
live to the console in order.  Running sequentially means each algorithm
gets the full CPU for its own parallel fitness evaluations.

Usage:
  python -m src.experiments.baseline_experiments                              # Jakarta (default)
  python -m src.experiments.baseline_experiments --instances beijing kotakinabalu
  python -m src.experiments.baseline_experiments --instances jakarta beijing
  python -m src.experiments.baseline_experiments --instances beijing --only evaluate_baseline
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SHARED_OUT = ROOT / "src" / "outputs"
sys.path.append(str(ROOT))

from config import INSTANCES  # noqa: E402  (per-instance SUMO config / baseline / out_dir)

# name -> module path run via `python -m <module>`, executed in this order
EXPERIMENTS = {
    "random_search": "src.algorithms.random_search",
    "evaluate_baseline": "src.algorithms.evaluate_baseline",
    "simple_genetic_algorithm": "src.algorithms.simple_genetic_algorithm",
}


def _snapshot():
    """Map every top-level src/outputs/*.json to (mtime_ns, bytes)."""
    return {p.name: (p.stat().st_mtime_ns, p.read_bytes())
            for p in SHARED_OUT.glob("*.json")}


def _relocate_outputs(before, dest_dir):
    """Move files this run produced into ``dest_dir``; restore overwritten ones.

    A file is "produced" if it is new or its mtime changed since ``before``.
    Each produced file is moved into ``dest_dir``; if the run overwrote a
    pre-existing Jakarta file, that file's original bytes are restored to
    src/outputs so the default-instance copy survives untouched.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in list(SHARED_OUT.glob("*.json")):
        prev = before.get(p.name)
        produced = prev is None or p.stat().st_mtime_ns != prev[0]
        if not produced:
            continue
        (dest_dir / p.name).write_bytes(p.read_bytes())
        p.unlink()
        moved.append(p.name)
        if prev is not None:  # restore the Jakarta original we just moved aside
            (SHARED_OUT / p.name).write_bytes(prev[1])
    return moved


def run_instance(instance, modules):
    """Run the selected algorithms for one instance and wait for each."""
    spec = INSTANCES[instance]
    baseline = spec["baseline_data"]
    if not baseline.exists():
        print(f"\n!! [{instance}] baseline traffic data missing:\n   {baseline}")
        print(f"   Generate it first, e.g.:\n"
              f"     python -m src.experiments.best_config_instance_experiments "
              f"--instances {instance} --max-evals 1\n"
              f"   then re-run this command.")
        return {}

    env = os.environ.copy()
    env["TLS_SUMOCFG"] = str(spec["sumocfg"])
    env["TLS_BASELINE_DATA"] = str(baseline)

    dest = spec["out_dir"]
    is_default = dest == SHARED_OUT

    print(f"\n{'#'*64}")
    print(f"# Instance: {instance}")
    print(f"#   sumocfg : {spec['sumocfg'].relative_to(ROOT)}")
    print(f"#   baseline: {baseline.relative_to(ROOT)}")
    print(f"#   outputs : {dest.relative_to(ROOT)}")
    print(f"{'#'*64}")

    results, timings = {}, {}
    for name, module in modules.items():
        print(f"\n{'='*60}")
        print(f"# [{instance}] Running {name} ({module})")
        print(f"{'='*60}\n")

        before = None if is_default else _snapshot()
        t0 = time.time()
        ret = subprocess.run(
            [sys.executable, "-m", module], cwd=str(ROOT), env=env,
        ).returncode
        elapsed = time.time() - t0

        moved = [] if is_default else _relocate_outputs(before, dest)

        results[name] = ret
        timings[name] = elapsed
        status = "OK" if ret == 0 else f"FAIL({ret})"
        suffix = f" → {dest.relative_to(ROOT)}/ ({len(moved)} file(s))" if moved else ""
        print(f"\n>>> [{instance}] {name} finished in {elapsed:.1f}s → {status}{suffix}")

    return {"results": results, "timings": timings}


def run_all(instances, only):
    """Run every selected experiment for every selected instance."""
    modules = {k: v for k, v in EXPERIMENTS.items() if only is None or k in only}
    if not modules:
        print(f"No matching experiments for --only {only}. "
              f"Known: {list(EXPERIMENTS)}")
        return

    t_start = time.time()
    summary = {}
    for instance in instances:
        summary[instance] = run_instance(instance, modules)

    total = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"All instances done in {total:.1f}s")
    print(f"{'='*60}")
    print(f"{'Instance':<16}{'Experiment':<28}{'Result':>10}{'Time':>10}")
    print("─" * 64)
    for instance, info in summary.items():
        if not info:
            print(f"{instance:<16}{'(skipped — no baseline)':<48}")
            continue
        for name, ret in info["results"].items():
            status = "OK" if ret == 0 else f"FAIL({ret})"
            print(f"{instance:<16}{name:<28}{status:>10}{info['timings'][name]:>9.1f}s")
    print(f"\nPer-instance JSON results under: {SHARED_OUT}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", nargs="*", default=["jakarta"],
                        help=f"Instances to run (default: jakarta). "
                             f"Known: {list(INSTANCES)}.")
    parser.add_argument("--only", nargs="*", default=None,
                        help=f"Subset of algorithms to run "
                             f"(default: all). Known: {list(EXPERIMENTS)}.")
    args = parser.parse_args()

    unknown = [i for i in args.instances if i not in INSTANCES]
    if unknown:
        print(f"WARNING: unknown instances ignored: {unknown} "
              f"(known: {list(INSTANCES)})")
    instances = [i for i in args.instances if i in INSTANCES]
    if not instances:
        print("No valid instances selected — nothing to do.")
        return

    run_all(instances, args.only)


if __name__ == "__main__":
    main()
