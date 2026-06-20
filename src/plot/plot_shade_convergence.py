"""
Convergence plots for SHADE: with mutation vs without mutation vs the plain
SHADE baseline, per instance.

For each instance, metric, and mutation setting we draw one standalone image:

  * SHADE *with* pairwise mutation (cluster_v3_mut), one curve per init
    strategy (euclidian / fastest / random / shortest)
  * SHADE *without* mutation       (cluster_v3_nomut), same strategies

Plain SHADE (no cluster crossover, no mutation) is overlaid on every image as
the dashed black baseline so each is read against the same reference.

Four images are produced per instance (2 metrics × 2 mutation settings):
  * <instance>_best_{mut,nomut}.png — best-so-far cost vs cumulative evals
  * <instance>_avg_{mut,nomut}.png  — population-average cost vs cumulative evals

Each curve is the mean over the 5 repetitions.  Because the methods use
different population sizes (hence a different number of generations for the
same 2000-eval budget), per-run histories are interpolated onto a common
eval grid before averaging.

The result files for the 5 reps live in different directories with different
naming conventions (the early reps came from the hyper-parameter sweep), so
each instance resolves its own per-rep paths.  All three instances share the
same selected best-config hyper-parameters per strategy.

Usage:
    python src/plot/plot_shade_convergence.py            # all instances
    python src/plot/plot_shade_convergence.py jakarta    # one instance
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "src" / "outputs"

STRATEGIES = ["euclidian", "fastest", "random", "shortest"]
N_REPS = 5

# Selected best-config hyper-parameter stems per strategy (shared by all
# instances).  Used to pick the matching file out of the sweep results.
MUT_HP = {
    "euclidian": "pop50_mp0.3_ms3",
    "fastest":   "pop50_mp0.5_ms2",
    "random":    "pop50_mp0.3_ms1",
    "shortest":  "pop100_mp0.1_ms1",
}
NOMUT_HP = {
    "euclidian": "pop50",
    "fastest":   "pop50",
    "random":    "pop200",
    "shortest":  "pop50",
}
PLAIN_HP = "pop100"

# Common eval grid (all runs share a 2000-eval budget).
EVAL_GRID = np.linspace(50, 2000, 80)

# Consistent colour per strategy across both panels.
STRAT_COLOUR = {
    "euclidian": "#2ca02c",
    "fastest":   "#9467bd",
    "random":    "#d62728",
    "shortest":  "#1f77b4",
}



# ── Per-rep file resolution ────────────────────────────────────────────────
def _jakarta_paths(kind, strat):
    """Jakarta's 5 reps are spread across the sweep output dirs."""
    bp = OUT_DIR / "best_parameter"
    j1 = bp / "single_de_repetition_v2_and_jakarta_rep1"   # rep1
    j23 = bp / "jakarta_rep2_rep3"                          # rep2 (_rep1), rep3 (_rep2)
    jplain23 = bp / "de_plain"                              # plain rep2/rep3
    j45 = OUT_DIR / "jakarta_4_5"                           # rep4, rep5

    if kind == "plain":
        return [
            j1 / f"single_de_plain_{PLAIN_HP}.json",
            jplain23 / f"differential_evolution_plain_{PLAIN_HP}_rep1.json",
            jplain23 / f"differential_evolution_plain_{PLAIN_HP}_rep2.json",
            j45 / f"jakarta_plain_{PLAIN_HP}_rep4.json",
            j45 / f"jakarta_plain_{PLAIN_HP}_rep5.json",
        ]

    hp = MUT_HP[strat] if kind == "mut" else NOMUT_HP[strat]
    nm = "" if kind == "mut" else "nomut_"   # early reps drop "mut" token
    return [
        j1 / f"single_de_cluster_v3_{nm}{strat}_{hp}.json",
        j23 / f"differential_evolution_cluster_v3_{nm}{strat}_{hp}_rep1.json",
        j23 / f"differential_evolution_cluster_v3_{nm}{strat}_{hp}_rep2.json",
        j45 / f"jakarta_cluster_v3_{kind}_{strat}_{hp}_rep4.json",
        j45 / f"jakarta_cluster_v3_{kind}_{strat}_{hp}_rep5.json",
    ]


def _instance_paths(name, kind, strat):
    """Beijing / Kota Kinabalu: rep1-3 in instances/, rep4-5 in best_config."""
    early = OUT_DIR / "instances" / name
    late = OUT_DIR / "best_config_rep4_rep5" / name

    if kind == "plain":
        stem = f"{name}_plain_{PLAIN_HP}"
    else:
        hp = MUT_HP[strat] if kind == "mut" else NOMUT_HP[strat]
        stem = f"{name}_cluster_v3_{kind}_{strat}_{hp}"

    return [
        early / f"{stem}_rep1.json",
        early / f"{stem}_rep2.json",
        early / f"{stem}_rep3.json",
        late / f"{stem}_rep4.json",
        late / f"{stem}_rep5.json",
    ]


def rep_paths(name, kind, strat):
    if name == "jakarta":
        return _jakarta_paths(kind, strat)
    return _instance_paths(name, kind, strat)


# ── Curve building ─────────────────────────────────────────────────────────
def _series(path, metric):
    """(evals, values) for one run. metric: 'best' (best-so-far) or 'mean'."""
    with open(path) as f:
        hist = json.load(f)["fitness_history"]
    evals = np.array([h["evals"] for h in hist], dtype=float)
    if metric == "best":
        vals = np.minimum.accumulate(
            np.array([h["best"] for h in hist], dtype=float))
    else:
        vals = np.array([h["mean"] for h in hist], dtype=float)
    return evals, vals


def _avg_curve(paths, metric):
    """Interpolate each existing run onto EVAL_GRID; average. Returns (curve, n)."""
    curves = []
    for p in paths:
        if not p.exists():
            continue
        evals, vals = _series(p, metric)
        curves.append(np.interp(EVAL_GRID, evals, vals))
    if not curves:
        return None, 0
    return np.mean(curves, axis=0), len(curves)


# metric -> (filename suffix, y-axis label, figure-title phrase)
METRICS = {
    "best": ("best", "Best-so-far cost (fitness)", "best-so-far"),
    "mean": ("avg",  "Population average cost (fitness)", "population average"),
}


# kind -> (filename token, panel title)
KINDS = {
    "mut":   ("mut",   "SHADE with pairwise mutation"),
    "nomut": ("nomut", "SHADE without mutation"),
}


def plot_panel(name, metric, kind):
    suffix, ylabel, phrase = METRICS[metric]
    kind_token, title = KINDS[kind]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    plain, _ = _avg_curve(rep_paths(name, "plain", None), metric)

    for strat in STRATEGIES:
        curve, n = _avg_curve(rep_paths(name, kind, strat), metric)
        if curve is None:
            print(f"  [warn] {name} {kind} {strat}: no files found")
            continue
        if n != N_REPS:
            print(f"  [warn] {name} {kind} {strat}: {n}/{N_REPS} reps")
        ax.plot(EVAL_GRID, curve, color=STRAT_COLOUR[strat],
                lw=2.0, label=strat)

    if plain is not None:
        ax.plot(EVAL_GRID, plain, color="black", lw=2.0, ls="--",
                label="plain SHADE (baseline)")

    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"{title}\n{name} — {phrase}, mean over {N_REPS} reps",
        fontsize=12,
    )
    fig.tight_layout()

    out_file = OUT_DIR / f"shade_convergence_{name}_{suffix}_{kind_token}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_file}")


def main():
    instances = ["jakarta", "beijing", "kotakinabalu"]
    wanted = sys.argv[1:] or instances
    for name in wanted:
        if name not in instances:
            raise SystemExit(f"Unknown instance '{name}'. "
                             f"Choose from: {', '.join(instances)}")
        for metric in METRICS:
            for kind in KINDS:
                plot_panel(name, metric, kind)


if __name__ == "__main__":
    main()
