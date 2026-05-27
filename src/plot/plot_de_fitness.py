"""
Plot best / worst / mean fitness bands per generation for the SHADE
differential-evolution runs.

For each result JSON, the per-generation `gen_best` (lowest cost) and
`gen_worst` (highest cost) form a shaded band; the `mean` is drawn as a
solid line down the middle.  This gives the "thick line" look where the
band thickness is the spread of the population and the centre line is the
mean.

Covers both algorithms:
  - differential_evolution.json                       (plain SHADE)
  - differential_evolution_cluster_v3_{tree}.json     (cluster crossover v3)

Usage:  python src/plot/plot_de_fitness.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "src" / "outputs"

# (filename, panel title, colour) — one panel each.
RUNS = [
    ("differential_evolution.json",
     "SHADE (plain)", "#1f77b4"),
    ("differential_evolution_cluster_v3_shortest.json",
     "Cluster v3 — shortest", "#d62728"),
    ("differential_evolution_cluster_v3_euclidian.json",
     "Cluster v3 — euclidian", "#2ca02c"),
    ("differential_evolution_cluster_v3_fastest.json",
     "Cluster v3 — fastest", "#9467bd"),
]


def load_history(path):
    with open(path) as f:
        data = json.load(f)
    hist = data["fitness_history"]
    gen   = np.array([h["gen"] for h in hist])
    best  = np.array([h["gen_best"] for h in hist])
    worst = np.array([h["gen_worst"] for h in hist])
    mean  = np.array([h["mean"] for h in hist])
    return gen, best, worst, mean, data.get("best_fitness")


def main():
    runs = [(OUT_DIR / fn, title, colour)
            for fn, title, colour in RUNS if (OUT_DIR / fn).exists()]
    if not runs:
        raise SystemExit("No result JSON files found in src/outputs/")

    n = len(runs)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(13, 5 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes.ravel()

    for ax, (path, title, colour) in zip(axes, runs):
        gen, best, worst, mean, final_best = load_history(path)

        # Shaded band: best (low cost) up to worst (high cost).
        ax.fill_between(gen, best, worst, color=colour, alpha=0.25,
                        label="best–worst spread")
        # Thin band edges.
        ax.plot(gen, best,  color=colour, lw=0.8, alpha=0.6)
        ax.plot(gen, worst, color=colour, lw=0.8, alpha=0.6)
        # Mean line down the middle.
        ax.plot(gen, mean, color=colour, lw=2.2, label="mean")

        subtitle = f"  (final best = {final_best:,.0f})" if final_best else ""
        ax.set_title(f"{title}{subtitle}", fontsize=12)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Cost (fitness)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    # Hide any unused panels.
    for ax in axes[len(runs):]:
        ax.set_visible(False)

    fig.suptitle("SHADE fitness per generation — best / worst / mean",
                 fontsize=15, y=0.995)
    fig.tight_layout()

    out_file = OUT_DIR / "de_fitness_best_worst_mean.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
