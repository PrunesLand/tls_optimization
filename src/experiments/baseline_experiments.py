"""
Run all baseline / algorithm experiments sequentially.

Runs each algorithm one after another, each in its own subprocess, while
keeping every algorithm module completely untouched.  Each subprocess
writes its own individual result files to src/outputs/ exactly as it does
when run standalone, so per-algorithm results are preserved.

Algorithms run (each as `python -m <module>`, in order):
  - Random Search             (src.algorithms.random_search)
  - Baseline evaluation       (src.algorithms.evaluate_baseline)
  - Simple Genetic Algorithm  (src.algorithms.simple_genetic_algorithm)

Note: plain DE-SHADE (and the novel cluster-v3 step-mutation sweep) now live
      in src.experiments.de_experiments, which runs them twice and averages;
      they are intentionally NOT launched from this baseline runner.

Each subprocess inherits this process's stdout/stderr, so output streams
live to the console in order.  Running sequentially means each algorithm
gets the full CPU for its own parallel fitness evaluations.

Usage:  python -m src.experiments.baseline_experiments
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# name -> module path run via `python -m <module>`, executed in this order
EXPERIMENTS = {
    "random_search": "src.algorithms.random_search",
    "evaluate_baseline": "src.algorithms.evaluate_baseline",
    "simple_genetic_algorithm": "src.algorithms.simple_genetic_algorithm",
}


def run_all():
    """Run every experiment one after another and wait for each to finish."""
    results = {}
    timings = {}
    t_start = time.time()

    for name, module in EXPERIMENTS.items():
        print(f"\n{'#'*60}")
        print(f"# Running {name} ({module})")
        print(f"{'#'*60}\n")

        t0 = time.time()
        ret = subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(ROOT),
        ).returncode
        elapsed = time.time() - t0

        results[name] = ret
        timings[name] = elapsed
        status = "OK" if ret == 0 else f"FAIL({ret})"
        print(f"\n>>> {name} finished in {elapsed:.1f}s → {status}")

    total = time.time() - t_start

    print(f"\n{'='*50}")
    print(f"All experiments done in {total:.1f}s")
    print(f"{'='*50}")
    print(f"{'Experiment':<28}{'Result':>10}{'Time':>10}")
    print("─" * 48)
    for name, ret in results.items():
        status = "OK" if ret == 0 else f"FAIL({ret})"
        print(f"{name:<28}{status:>10}{timings[name]:>9.1f}s")
    print(f"\nIndividual JSON results: {ROOT / 'src' / 'outputs'}")


if __name__ == "__main__":
    run_all()
