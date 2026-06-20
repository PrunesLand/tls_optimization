"""
Aggregate the best-parameter repetitions and select, for every experiment and
every linkage tree, the configuration with the lowest *mean* cost across the
repetitions.

This is a pure post-processing step over the per-run JSONs already written by
``de_experiments.py`` and ``single_de_repetition.py`` — it runs no SUMO/DE, it
only reads, averages and ranks.

WHERE THE RUNS LIVE
-------------------
All per-run JSONs sit under ``src/outputs/best_parameter/``:

* ``de_plain/``               — plain SHADE, repetitions 1 & 2 (seeds 42 / 43).
* ``individual/``             — cluster-v3 (with and without step-mutation),
                                repetitions 1 & 2 (seeds 42 / 43).
* ``single_de_repetition_v2/``— repetition 3 (seed 44) for *all* variants.

Reps 1 & 2 were produced by ``de_experiments.py`` and carry a ``_rep1`` /
``_rep2`` suffix.  Rep 3 was produced separately by ``single_de_repetition.py``
with a different ``single_de_`` filename prefix and **no** rep suffix — hence it
lives in its own folder and is treated as repetition 3 here.

THE THREE EXPERIMENTS  (read from each JSON's ``algorithm`` field)
-----------------------------------------------------------------
* ``plain``            (``shade_evox``)                             — no tree.
* ``cluster_v3_nomut`` (``shade_evox_cluster_crossover_v3``)        — per tree.
* ``cluster_v3_mut``   (``shade_evox_cluster_crossover_v3_stepmut``)— per tree.

A *configuration* is the hyper-parameter combination swept within an experiment:

* plain            → ``pop``
* cluster_v3_nomut → ``(tree, pop)``
* cluster_v3_mut   → ``(tree, pop, mp, ms)``

``tree``, ``pop`` and ``ms`` (step size) are read from each JSON's own metadata.
The mutation probability ``mp`` is **not** stored in the JSON, so it is parsed
from the filename (``..._mp0.3_...``); the repetition number likewise.

THE METRIC
----------
``best_fitness`` (identical to ``best_configuration.composite_cost``) — a traffic
cost, so **lower is better**.  For each configuration we average ``best_fitness``
over the repetitions it has, mirroring how ``de_experiments.py`` averages its two
reps, then extend that to all three.

SELECTING THE WINNER
--------------------
Because rep 3's greedy coordinate-descent explored a slightly different path than
reps 1 & 2, some step-mutation configurations have fewer than three reps.  A
one-run "average" is not comparable to a three-run one, so the headline winner
per (experiment, tree) is chosen among configurations covered by at least
``--min-reps`` repetitions (default: all three; falls back to fewer only if a
tree has nothing fully covered).  ``best_including_incomplete`` is always also
reported — the lowest-mean config regardless of coverage — and flagged when it
differs, so a promising single-run config is never silently dropped.

OUTPUT
------
* A JSON summary (default ``src/outputs/best_parameter/best_configuration_summary.json``)
  with, per experiment and tree: the selected winner, the winner allowing
  incomplete coverage, and the full ranking of every configuration (mean, std,
  per-rep values and source files, and the representative best single run).
* Human-readable tables printed to stdout.

Usage:
  python -m src.experiments.select_best_configuration
  python -m src.experiments.select_best_configuration --min-reps 1
  python -m src.experiments.select_best_configuration --embed-config
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Where the per-run JSONs live ──────────────────────────────────────
DEFAULT_BASE = ROOT / "src" / "outputs" / "best_parameter"

# Folders holding reps 1 & 2 (files carry _rep1 / _rep2) and rep 3 (no suffix).
REP12_DIRS = ["individual", "de_plain"]
REP3_DIR = "single_de_repetition_v2"

# The JSON ``algorithm`` field is authoritative for the experiment family.
ALGORITHM_TO_EXPERIMENT = {
    "shade_evox": "plain",
    "shade_evox_cluster_crossover_v3": "cluster_v3_nomut",
    "shade_evox_cluster_crossover_v3_stepmut": "cluster_v3_mut",
}

EXPERIMENT_LABELS = {
    "plain": "Plain SHADE (no clustering, no step-mutation)",
    "cluster_v3_nomut": "Cluster-v3 bin-crossing, step-mutation OFF",
    "cluster_v3_mut": "Cluster-v3 bin-crossing, step-mutation ON",
}

# Order experiments are reported in.
EXPERIMENT_ORDER = ["cluster_v3_mut", "cluster_v3_nomut", "plain"]

# Plain SHADE ignores the linkage trees; the cluster variants sweep over them.
EXPERIMENTS_WITH_TREE = {"cluster_v3_nomut", "cluster_v3_mut"}

# Preferred display/tie-break order for trees; unknown trees sort after these.
TREE_ORDER = ["euclidian", "fastest", "random", "shortest"]
NO_TREE_KEY = "(plain — no linkage tree)"

METRIC = "best_fitness"          # == best_configuration.composite_cost (lower better)
EXPECTED_REPS = 3

_REP_RE = re.compile(r"_rep(\d+)")
_MP_RE = re.compile(r"_mp([0-9.]+)")


# ══════════════════════════════════════════════════════════════════════
# Loading & classification
# ══════════════════════════════════════════════════════════════════════

def _rep_of(fname, is_rep3_dir):
    """Repetition number from the filename, or 3 for the rep-3 folder."""
    m = _REP_RE.search(fname)
    if m:
        return int(m.group(1))
    return 3 if is_rep3_dir else None


def _mp_of(fname):
    """Mutation probability parsed from the filename (not stored in the JSON)."""
    m = _MP_RE.search(fname)
    return float(m.group(1)) if m else None


def load_runs(base):
    """Read every per-run JSON under ``base`` into flat run records.

    Returns ``(runs, skipped)`` where each run is::

        {experiment, tree, pop, mp, ms, rep, best_fitness, file,
         composite_cost, time_s}

    and ``skipped`` is a list of ``(filename, reason)`` for files that are not
    per-run results (summaries, unknown algorithms, missing rep, …).
    """
    runs, skipped = [], []
    dirs = [(d, False) for d in REP12_DIRS] + [(REP3_DIR, True)]
    for sub, is_rep3 in dirs:
        folder = base / sub
        if not folder.is_dir():
            skipped.append((sub + "/", "folder not found"))
            continue
        for path in sorted(folder.glob("*.json")):
            fname = path.name
            try:
                with open(path) as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                skipped.append((fname, f"unreadable: {e}"))
                continue

            # The summary files lack these keys; only per-run files have both.
            if METRIC not in d or "algorithm" not in d:
                skipped.append((fname, "not a per-run file (summary?)"))
                continue

            exp = ALGORITHM_TO_EXPERIMENT.get(d.get("algorithm"))
            if exp is None:
                skipped.append((fname, f"unknown algorithm {d.get('algorithm')!r}"))
                continue

            rep = _rep_of(fname, is_rep3)
            if rep is None:
                skipped.append((fname, "could not determine repetition"))
                continue

            tree = d.get("tree") if exp in EXPERIMENTS_WITH_TREE else None
            mp = _mp_of(fname) if exp == "cluster_v3_mut" else None
            ms = (float(d["step_size"])
                  if exp == "cluster_v3_mut" and d.get("step_size") is not None
                  else None)
            if exp == "cluster_v3_mut" and mp is None:
                skipped.append((fname, "step-mutation run without mp in filename"))
                continue

            runs.append({
                "experiment": exp,
                "tree": tree,
                "pop": int(d["pop_size"]),
                "mp": mp,
                "ms": ms,
                "rep": rep,
                "best_fitness": float(d[METRIC]),
                "composite_cost": d.get("best_configuration", {}).get("composite_cost"),
                "time_s": d.get("time_s"),
                "file": str(path.relative_to(ROOT)),
            })
    return runs, skipped


def group_configs(runs):
    """Group runs by configuration, collecting one entry per repetition.

    Returns ``{config_key: {meta..., "reps": {rep: {best_fitness, file, ...}}}}``
    plus a list of duplicate ``(file, reason)`` collisions.
    """
    configs, dupes = {}, []
    for r in runs:
        key = (r["experiment"], r["tree"], r["pop"], r["mp"], r["ms"])
        entry = configs.setdefault(key, {
            "experiment": r["experiment"], "tree": r["tree"], "pop": r["pop"],
            "mp": r["mp"], "ms": r["ms"], "reps": {},
        })
        if r["rep"] in entry["reps"]:
            dupes.append((r["file"], f"duplicate rep {r['rep']} for {key}"))
            continue
        entry["reps"][r["rep"]] = {
            "best_fitness": r["best_fitness"],
            "composite_cost": r["composite_cost"],
            "time_s": r["time_s"],
            "file": r["file"],
        }
    return configs, dupes


# ══════════════════════════════════════════════════════════════════════
# Averaging, ranking & selection
# ══════════════════════════════════════════════════════════════════════

def summarize_config(cfg):
    """Average a configuration's ``best_fitness`` over the reps it has."""
    reps = cfg["reps"]
    order = sorted(reps)
    values = [reps[r]["best_fitness"] for r in order]
    n = len(values)
    best_rep = min(order, key=lambda r: reps[r]["best_fitness"])
    return {
        "pop": cfg["pop"], "mp": cfg["mp"], "ms": cfg["ms"],
        "n_reps": n,
        "reps_present": order,
        "mean_best_fitness": statistics.fmean(values),
        "std_best_fitness": statistics.stdev(values) if n >= 2 else 0.0,
        "min_best_fitness": min(values),
        "max_best_fitness": max(values),
        "per_rep": {
            str(r): {"best_fitness": reps[r]["best_fitness"],
                     "time_s": reps[r]["time_s"],
                     "file": reps[r]["file"]}
            for r in order
        },
        # The concrete deployable solution: the single best run for this config.
        "representative_run": {
            "rep": best_rep,
            "best_fitness": reps[best_rep]["best_fitness"],
            "file": reps[best_rep]["file"],
        },
    }


def _rank_sort(stats_list):
    """Configurations sorted by mean cost (ties broken by best single run)."""
    ranked = sorted(stats_list,
                    key=lambda s: (s["mean_best_fitness"], s["min_best_fitness"]))
    for i, s in enumerate(ranked, 1):
        s["rank"] = i
    return ranked


def pick_winner(stats_list, min_reps):
    """Lowest-mean config covered by >= ``min_reps`` reps (with fallback).

    Returns ``(winner, fell_back)``; ``fell_back`` is True when no config met
    ``min_reps`` and the choice was made over all configs instead.
    """
    eligible = [s for s in stats_list if s["n_reps"] >= min_reps]
    fell_back = not eligible
    if fell_back:
        eligible = stats_list
    winner = min(eligible, key=lambda s: (s["mean_best_fitness"],
                                          s["min_best_fitness"], s["pop"]))
    return winner, fell_back


def _tree_sort_key(tree):
    if tree in TREE_ORDER:
        return (0, TREE_ORDER.index(tree))
    return (1, str(tree))


def build_report(configs, min_reps, embed_config, base):
    """Assemble the nested experiment → tree → ranking report."""
    # experiment -> tree_display -> [config stats]
    grouped = {}
    for key, cfg in configs.items():
        exp = cfg["experiment"]
        tree_disp = cfg["tree"] if exp in EXPERIMENTS_WITH_TREE else NO_TREE_KEY
        grouped.setdefault(exp, {}).setdefault(tree_disp, []).append(
            summarize_config(cfg))

    experiments_out = {}
    headline = []
    for exp in EXPERIMENT_ORDER:
        if exp not in grouped:
            continue
        trees_out = {}
        tree_keys = sorted(grouped[exp], key=_tree_sort_key)
        for tree in tree_keys:
            ranking = _rank_sort(grouped[exp][tree])
            winner, fell_back = pick_winner(ranking, min_reps)
            best_any, _ = pick_winner(ranking, 1)
            incomplete_differs = (best_any is not winner)

            if embed_config:
                _embed_representative(winner, base)
                if incomplete_differs:
                    _embed_representative(best_any, base)

            trees_out[tree] = {
                "best": winner,
                "selection_min_reps": min_reps,
                "selection_fell_back": fell_back,
                "best_including_incomplete": (best_any if incomplete_differs
                                              else None),
                "n_configs": len(ranking),
                "ranking": ranking,
            }
            headline.append({
                "experiment": exp,
                "tree": tree,
                "pop": winner["pop"], "mp": winner["mp"], "ms": winner["ms"],
                "mean_best_fitness": winner["mean_best_fitness"],
                "n_reps": winner["n_reps"],
                "representative_file": winner["representative_run"]["file"],
            })
        experiments_out[exp] = {
            "label": EXPERIMENT_LABELS[exp],
            "has_tree": exp in EXPERIMENTS_WITH_TREE,
            "trees": trees_out,
        }

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metric": f"{METRIC} (== best_configuration.composite_cost); LOWER is better",
        "averaging": "arithmetic mean of best_fitness over the repetitions a "
                     "configuration has",
        "selection_rule": (
            f"per (experiment, tree): lowest mean best_fitness among configs "
            f"with >= {min_reps} repetition(s); falls back to fewer only if a "
            f"tree has none that qualify. 'best_including_incomplete' reports "
            f"the lowest-mean config ignoring coverage when it differs."),
        "rep_sources": {
            "1": "individual/ + de_plain/  (de_experiments.py, seed 42)",
            "2": "individual/ + de_plain/  (de_experiments.py, seed 43)",
            "3": f"{REP3_DIR}/  (single_de_repetition.py, seed 44)",
        },
        "experiments": experiments_out,
        "headline_winners": headline,
    }


def _embed_representative(stats, base):
    """Attach the winning run's full ``best_configuration`` (tls_data)."""
    rel = stats["representative_run"]["file"]
    try:
        with open(ROOT / rel) as f:
            stats["representative_run"]["best_configuration"] = \
                json.load(f).get("best_configuration")
    except (OSError, json.JSONDecodeError):
        stats["representative_run"]["best_configuration"] = None


# ══════════════════════════════════════════════════════════════════════
# Console rendering
# ══════════════════════════════════════════════════════════════════════

def _fmt(v, width=0, prec=2):
    return f"{v:>{width}.{prec}f}" if isinstance(v, (int, float)) else f"{str(v):>{width}}"


def print_report(report):
    print(f"\n{'='*78}")
    print("BEST CONFIGURATION PER EXPERIMENT PER TREE")
    print(f"metric: {report['metric']}")
    print(f"{'='*78}")

    for exp, exp_block in report["experiments"].items():
        print(f"\n### {exp_block['label']}  [{exp}]")
        for tree, t in exp_block["trees"].items():
            print(f"\n  Tree: {tree}    ({t['n_configs']} configs)")
            print(f"    {'rank':>4} {'pop':>4} {'mp':>5} {'ms':>4} {'n':>2} "
                  f"{'mean':>11} {'std':>9} {'min':>11}")
            print(f"    {'-'*4} {'-'*4} {'-'*5} {'-'*4} {'-'*2} "
                  f"{'-'*11} {'-'*9} {'-'*11}")
            for s in t["ranking"]:
                star = "  <= best" if s is t["best"] else ""
                if t["best_including_incomplete"] is s:
                    star = "  <= best (incomplete coverage)"
                mp = "-" if s["mp"] is None else f"{s['mp']:.1f}"
                ms = "-" if s["ms"] is None else f"{s['ms']:.0f}"
                print(f"    {s['rank']:>4} {s['pop']:>4} {mp:>5} {ms:>4} "
                      f"{s['n_reps']:>2} {s['mean_best_fitness']:>11.2f} "
                      f"{s['std_best_fitness']:>9.2f} "
                      f"{s['min_best_fitness']:>11.2f}{star}")
            if t["selection_fell_back"]:
                print(f"    ! no config had >= {t['selection_min_reps']} reps; "
                      f"winner chosen over all configs.")

    print(f"\n{'='*78}")
    print("HEADLINE WINNERS")
    print(f"{'='*78}")
    print(f"{'experiment':<18} {'tree':<26} {'pop':>4} {'mp':>5} {'ms':>4} "
          f"{'n':>2} {'mean best_fitness':>18}")
    print("-" * 78)
    for w in report["headline_winners"]:
        mp = "-" if w["mp"] is None else f"{w['mp']:.1f}"
        ms = "-" if w["ms"] is None else f"{w['ms']:.0f}"
        print(f"{w['experiment']:<18} {str(w['tree']):<26} {w['pop']:>4} "
              f"{mp:>5} {ms:>4} {w['n_reps']:>2} {w['mean_best_fitness']:>18.2f}")


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE,
                        help=f"Folder holding the rep sub-folders "
                             f"(default: {DEFAULT_BASE}).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path (default: "
                             "<base-dir>/best_configuration_summary.json).")
    parser.add_argument("--min-reps", type=int, default=EXPECTED_REPS,
                        help="Min repetitions a config needs to be eligible as "
                             f"the headline winner (default: {EXPECTED_REPS}).")
    parser.add_argument("--embed-config", action="store_true",
                        help="Embed the winning run's full best_configuration "
                             "(tls_data) in the JSON output.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print the tables to stdout.")
    args = parser.parse_args()

    base = args.base_dir
    out_path = args.out or (base / "best_configuration_summary.json")

    runs, skipped = load_runs(base)
    if not runs:
        raise SystemExit(f"No per-run JSONs found under {base}. "
                         f"Skipped: {skipped}")
    configs, dupes = group_configs(runs)
    report = build_report(configs, args.min_reps, args.embed_config, base)
    report["files_skipped"] = [{"file": f, "reason": r} for f, r in skipped]
    report["duplicate_runs"] = [{"file": f, "reason": r} for f, r in dupes]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)

    if not args.quiet:
        print_report(report)
        n_configs = sum(len(t["ranking"])
                        for e in report["experiments"].values()
                        for t in e["trees"].values())
        print(f"\nParsed {len(runs)} runs into {n_configs} configurations "
              f"across {len(report['experiments'])} experiments.")
        if skipped:
            print(f"Skipped {len(skipped)} non-run file(s) "
                  f"(summaries etc.) — see 'files_skipped' in the JSON.")
        if dupes:
            print(f"WARNING: {len(dupes)} duplicate run(s) ignored "
                  f"— see 'duplicate_runs' in the JSON.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
