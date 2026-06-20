"""
Wilcoxon analysis: does the novel SHADE step-pairwise-mutation help?

For every instance we hold the DE cluster-v3 bin-crossing machinery fixed and
toggle the novel mutation ON ("mut", SHADE step-pairwise-mutation) vs OFF
("nomut").  Both variants were run for 5 repetitions (shared seeds per rep:
rep1->42, rep2->43, rep3->44, rep4->45, rep5->46) across 4 distance-tree
strategies (euclidian / fastest / random / shortest), at each strategy's
best-found hyper-parameters.  Lower composite_cost (best_fitness) is better.

Three paired Wilcoxon signed-rank views per instance:

  1. per-tree      : mut vs nomut, paired by rep        (n = 5)
  2. pooled        : mut vs nomut, paired by (tree,rep) (n = 20, well-powered)
  3. best-vs-best  : the lowest-mean mut config vs the lowest-mean nomut
                     config, paired by rep              (n = 5)

NOTE on power: with n = 5 the exact two-sided signed-rank test cannot drop
below p = 0.0625, so per-tree results are directional evidence only; the
pooled n = 20 test is the one that can reach conventional significance.

Usage:  python3.11 -m src.experiments.wilcoxon_mutation_analysis
"""
import json
import glob
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "src", "outputs", "wilcoxon_mutation")

INSTANCES = ["jakarta", "beijing", "kotakinabalu"]
TREES = ["euclidian", "fastest", "random", "shortest"]
REP_SEEDS = {1: 42, 2: 43, 3: 44, 4: 45, 5: 46}


# ── 1. Collect every single-run result file ─────────────────────────────────
# Reuse the config-exact loader so Jakarta's hyper-parameter sweep folder does
# not contaminate the per-tree series (keying on tree alone silently mixes the
# many pop/mp/ms configs that exist for Jakarta reps 1-3).
from src.experiments.wilcoxon_method_comparison import collect  # noqa: E402


def baseline_costs():
    """Per-instance deterministic baseline composite cost, if available."""
    out = {}
    cand = {
        "jakarta": os.path.join(ROOT, "src", "outputs", "baseline_results.json"),
        "beijing": os.path.join(ROOT, "src", "outputs", "beijing_baseline"),
        "kotakinabalu": os.path.join(ROOT, "src", "outputs", "kinabalu_baseline"),
    }
    for inst, p in cand.items():
        try:
            if os.path.isdir(p):
                fs = glob.glob(os.path.join(p, "**", "*.json"), recursive=True)
                p = next((f for f in fs if "baseline" in f.lower()), fs[0])
            out[inst] = float(json.load(open(p))["best_fitness"])
        except Exception:
            pass
    return out


# ── 2. Paired Wilcoxon helper ───────────────────────────────────────────────
def paired_test(mut, nomut):
    """mut, nomut: equal-length arrays paired element-wise. Lower is better."""
    mut, nomut = np.asarray(mut, float), np.asarray(nomut, float)
    diff = mut - nomut                       # <0 => mut better (lower cost)
    n = len(diff)
    res = {
        "n": n,
        "mut_mean": float(mut.mean()), "nomut_mean": float(nomut.mean()),
        "mut_median": float(np.median(mut)), "nomut_median": float(np.median(nomut)),
        "mean_diff": float(diff.mean()),
        "mut_wins": int((diff < 0).sum()), "nomut_wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
    }
    if np.allclose(diff, 0):
        res.update(stat=None, p_two=1.0, p_mut_better=1.0, verdict="identical")
        return res
    try:
        stat, p_two = wilcoxon(mut, nomut)                       # two-sided
        _, p_mut = wilcoxon(mut, nomut, alternative="less")      # mut < nomut
    except ValueError:
        res.update(stat=None, p_two=None, p_mut_better=None, verdict="n/a")
        return res
    better = "mut" if diff.mean() < 0 else "nomut"
    res.update(stat=float(stat), p_two=float(p_two), p_mut_better=float(p_mut),
               better_on_mean=better)
    return res


# ── 3. Run per-instance analysis ────────────────────────────────────────────
def analyse():
    data = collect()
    base = baseline_costs()
    report = {}

    for inst in INSTANCES:
        reps = sorted(REP_SEEDS)
        # per-tree
        per_tree = {}
        pooled_mut, pooled_nomut = [], []
        means = {}                       # (method,tree) -> mean
        series = {}                      # (method,tree) -> [vals by rep]
        for tree in TREES:
            mut = data.get((inst, "mut", tree), {})
            nom = data.get((inst, "nomut", tree), {})
            common = [r for r in reps if r in mut and r in nom]
            mv = [mut[r] for r in common]
            nv = [nom[r] for r in common]
            series[("mut", tree)] = [mut.get(r) for r in reps]
            series[("nomut", tree)] = [nom.get(r) for r in reps]
            if mv:
                means[("mut", tree)] = float(np.mean(mv))
            if nv:
                means[("nomut", tree)] = float(np.mean(nv))
            if len(common) >= 2:
                per_tree[tree] = paired_test(mv, nv)
                pooled_mut += mv
                pooled_nomut += nv

        pooled = paired_test(pooled_mut, pooled_nomut) if pooled_mut else None

        # best-vs-best (lowest mean config per method)
        best_mut_tree = min((t for t in TREES if ("mut", t) in means),
                            key=lambda t: means[("mut", t)])
        best_nom_tree = min((t for t in TREES if ("nomut", t) in means),
                            key=lambda t: means[("nomut", t)])
        bm = [data[(inst, "mut", best_mut_tree)][r] for r in reps]
        bn = [data[(inst, "nomut", best_nom_tree)][r] for r in reps]
        bvb = paired_test(bm, bn)
        bvb["mut_tree"] = best_mut_tree
        bvb["nomut_tree"] = best_nom_tree

        report[inst] = {
            "baseline_cost": base.get(inst),
            "tree_means": {f"{m}/{t}": v for (m, t), v in sorted(means.items())},
            "best_mut": {"tree": best_mut_tree, "mean": means[("mut", best_mut_tree)]},
            "best_nomut": {"tree": best_nom_tree, "mean": means[("nomut", best_nom_tree)]},
            "per_tree": per_tree,
            "pooled": pooled,
            "best_vs_best": bvb,
        }
    return report


# ── 4. Pretty print ─────────────────────────────────────────────────────────
def sig(p):
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    rep = analyse()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "wilcoxon_mutation_results.json"), "w") as f:
        json.dump(rep, f, indent=2)

    for inst, d in rep.items():
        print("\n" + "=" * 74)
        print(f"  {inst.upper()}   (lower composite_cost = better)")
        print("=" * 74)
        b = d["baseline_cost"]
        bm, bn = d["best_mut"], d["best_nomut"]
        if b:
            print(f"  baseline cost            : {b:,.0f}")
        print(f"  best WITH-mut strategy   : {bm['tree']:<10} mean={bm['mean']:,.0f}"
              + (f"  ({100*(b-bm['mean'])/b:+.1f}% vs baseline)" if b else ""))
        print(f"  best WITHOUT-mut strategy: {bn['tree']:<10} mean={bn['mean']:,.0f}"
              + (f"  ({100*(b-bn['mean'])/b:+.1f}% vs baseline)" if b else ""))

        print("\n  Per-tree paired Wilcoxon (mut vs nomut, n=5):")
        print(f"  {'tree':<10} {'mut_mean':>10} {'nomut_mean':>11} {'mut_wins':>9}"
              f" {'p(2-sided)':>11} {'p(mut<nom)':>11}")
        for t, r in d["per_tree"].items():
            print(f"  {t:<10} {r['mut_mean']:>10,.0f} {r['nomut_mean']:>11,.0f}"
                  f" {r['mut_wins']:>6}/{r['n']:<2}"
                  f" {('%.4f'%r['p_two']) if r['p_two'] is not None else 'n/a':>11}"
                  f" {('%.4f'%r['p_mut_better']) if r['p_mut_better'] is not None else 'n/a':>9} "
                  f"{sig(r['p_two'])}")

        p = d["pooled"]
        print(f"\n  POOLED over all trees (paired by tree,rep, n={p['n']}):")
        print(f"    mut mean   = {p['mut_mean']:,.0f}   nomut mean = {p['nomut_mean']:,.0f}")
        print(f"    mut wins   = {p['mut_wins']}/{p['n']}   (ties {p['ties']})")
        print(f"    Wilcoxon two-sided p = {p['p_two']:.4g}  {sig(p['p_two'])}")
        print(f"    one-sided  p(mut<nomut) = {p['p_mut_better']:.4g}  "
              f"-> {'MUT' if p['mean_diff']<0 else 'NOMUT'} better on mean")

        v = d["best_vs_best"]
        print(f"\n  BEST-vs-BEST ({v['mut_tree']}-mut vs {v['nomut_tree']}-nomut, n={v['n']}):")
        print(f"    means {v['mut_mean']:,.0f} vs {v['nomut_mean']:,.0f}; "
              f"mut wins {v['mut_wins']}/{v['n']}; "
              f"two-sided p={v['p_two']:.4g} {sig(v['p_two'])}; "
              f"one-sided p(mut<nom)={v['p_mut_better']:.4g}")

    print("\nSaved -> " + os.path.join(OUT, "wilcoxon_mutation_results.json"))


if __name__ == "__main__":
    main()
