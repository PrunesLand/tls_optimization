"""
Pairwise Wilcoxon signed-rank comparisons between the three DE variants, per
instance.  Lower composite_cost (best_fitness) is better.

  plain  : plain DE / "plain SHADE" (no cluster-v3 bin-crossing, no mutation)
  nomut  : cluster-v3 bin-crossing novelty, step-mutation OFF
  mut    : cluster-v3 bin-crossing novelty, SHADE step-mutation ON

Two requested questions:
  (A) nomut vs plain  -> does the bin-crossing/clustering novelty help on its
                         own, before any mutation is added?
  (B) mut   vs nomut  -> on top of the novelty, does the SHADE mutation help?

All variants have 5 reps (shared seeds rep1->42 ... rep5->46).  nomut/mut span
4 distance-tree strategies at each strategy's best hyper-parameters; plain has
a single best config (pop100).  Pairing is by repetition.

Views per comparison:  per-tree (n=5), pooled by (tree,rep) (n=20), and
best-config-vs-best-config (n=5).  With n=5 the exact two-sided signed-rank
test floors at p=0.0625, so the pooled n=20 view is the one that can reach
conventional significance.

Usage:  python3.11 -m src.experiments.wilcoxon_method_comparison
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
REPS = [1, 2, 3, 4, 5]
PLAIN_POP = 100          # best plain config

# Selected best hyper-parameter config per (method, tree), tuned on Jakarta and
# applied to every instance.  CRITICAL: Jakarta reps 1-3 live in the full
# hyper-parameter sweep folder where each tree has many pop/mp/ms configs, so
# without exact matching the loader silently mixes configs.  (pop, mp, ms) for
# mut; (pop,) for nomut.
BEST_MUT = {"euclidian": (50, 0.3, 3.0), "fastest": (50, 0.5, 2.0),
            "random": (50, 0.3, 1.0), "shortest": (100, 0.1, 1.0)}
BEST_NOMUT = {"euclidian": 50, "fastest": 50, "random": 200, "shortest": 50}


def _params(b):
    pop = re.search(r"pop(\d+)", b)
    mp = re.search(r"mp([\d.]+)", b)
    ms = re.search(r"ms([\d.]+)", b)
    return (int(pop.group(1)) if pop else None,
            float(mp.group(1)) if mp else None,
            float(ms.group(1).rstrip(".")) if ms else None)


def collect():
    """{(instance, method, tree-or-None): {rep: best_fitness}}.

    Only the selected best config per (method, tree) is kept, so Jakarta's
    sweep folder does not contaminate the series."""
    data = defaultdict(dict)
    for f in glob.glob(os.path.join(ROOT, "src", "outputs", "**", "*.json"),
                       recursive=True):
        b = os.path.basename(f)
        is_cluster = "cluster_v3" in b or ("single_de" in b and "plain" not in b)
        is_plain = "plain" in b
        if not (is_cluster or is_plain):
            continue
        if any(s in b for s in ("summary", "best_config_experiments",
                                "experiments", "averaged",
                                "best_configuration_summary")):
            continue

        inst = next((c for c in INSTANCES if c in f.lower()), None)
        if inst is None and ("best_parameter" in f or "baseline" in f):
            inst = "jakarta"
        if inst is None:
            continue

        m = re.search(r"rep(\d)", b)
        rep = int(m.group(1)) if m else (3 if "single_de" in b else None)
        if rep is None:
            continue

        pop, mp, ms = _params(b)
        if is_plain:
            if pop != PLAIN_POP:
                continue
            method, tree = "plain", None
        else:
            method = "nomut" if "nomut" in b else "mut"
            tree = next((t for t in TREES if t in b), None)
            if tree is None:
                continue
            if method == "mut" and (pop, mp, ms) != BEST_MUT[tree]:
                continue
            if method == "nomut" and pop != BEST_NOMUT[tree]:
                continue

        try:
            bf = json.load(open(f)).get("best_fitness")
        except Exception:
            continue
        if bf is not None:
            data[(inst, method, tree)][rep] = float(bf)
    return data


def wtest(a, b):
    """Paired signed-rank.  a,b lower=better.  diff=a-b (<0 => a better)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {"n": len(d), "a_mean": float(a.mean()), "b_mean": float(b.mean()),
           "mean_diff": float(d.mean()),
           "a_wins": int((d < 0).sum()), "b_wins": int((d > 0).sum()),
           "ties": int((d == 0).sum())}
    if np.allclose(d, 0):
        out.update(p_two=1.0, p_a_better=1.0)
        return out
    try:
        out["p_two"] = float(wilcoxon(a, b)[1])
        out["p_a_better"] = float(wilcoxon(a, b, alternative="less")[1])
    except ValueError:
        out.update(p_two=None, p_a_better=None)
    return out


def sig(p):
    return "" if p is None else ("***" if p < .001 else "**" if p < .01
                                 else "*" if p < .05 else "ns")


def compare(data, inst, A, B):
    """Compare method A vs method B for one instance.  Returns a result dict.

    A is the 'focus' method; p_a_better tests A < B (A is better)."""
    def series(method, tree):
        return data.get((inst, method, tree), {})

    def best_tree(method):
        cand = {t: np.mean([series(method, t)[r] for r in REPS])
                for t in TREES if all(r in series(method, t) for r in REPS)}
        return min(cand, key=cand.get) if cand else None

    per_tree, pooled_a, pooled_b = {}, [], []
    A_trees = TREES if A != "plain" else [None]
    # plain shares one series across trees; cluster methods have per-tree series
    for t in (TREES if "plain" not in (A, B) else TREES):
        a_tree = t if A != "plain" else None
        b_tree = t if B != "plain" else None
        sa, sb = series(A, a_tree), series(B, b_tree)
        common = [r for r in REPS if r in sa and r in sb]
        if len(common) < 2:
            continue
        av, bv = [sa[r] for r in common], [sb[r] for r in common]
        per_tree[t] = wtest(av, bv)
        pooled_a += av
        pooled_b += bv

    pooled = wtest(pooled_a, pooled_b) if pooled_a else None

    # best-config vs best-config (or vs plain)
    if A == "plain":
        bt_a, av = None, [series("plain", None)[r] for r in REPS]
    else:
        bt_a = best_tree(A); av = [series(A, bt_a)[r] for r in REPS]
    if B == "plain":
        bt_b, bv = None, [series("plain", None)[r] for r in REPS]
    else:
        bt_b = best_tree(B); bv = [series(B, bt_b)[r] for r in REPS]
    bvb = wtest(av, bv)
    bvb["a_tree"], bvb["b_tree"] = bt_a, bt_b
    return {"per_tree": per_tree, "pooled": pooled, "best_vs_best": bvb}


def show(title, A, B, res):
    print("\n  " + title)
    print(f"  {'tree':<10} {A+'_mean':>11} {B+'_mean':>11} {A+'_wins':>9}"
          f" {'p(2-sided)':>11} {'p('+A+'<'+B+')':>13}")
    for t, r in res["per_tree"].items():
        p2 = "%.4f" % r["p_two"] if r["p_two"] is not None else "n/a"
        pa = "%.4f" % r["p_a_better"] if r["p_a_better"] is not None else "n/a"
        print(f"  {t:<10} {r['a_mean']:>11,.0f} {r['b_mean']:>11,.0f}"
              f" {r['a_wins']:>6}/{r['n']:<2} {p2:>11} {pa:>11} {sig(r['p_two'])}")
    p = res["pooled"]
    print(f"  POOLED n={p['n']:<2} : {A} {p['a_mean']:,.0f} vs {B} {p['b_mean']:,.0f}"
          f" | {A} wins {p['a_wins']}/{p['n']} | two-sided p={p['p_two']:.4g} {sig(p['p_two'])}"
          f" | one-sided p({A}<{B})={p['p_a_better']:.4g}")
    v = res["best_vs_best"]
    ta = v["a_tree"] or "pop100"; tb = v["b_tree"] or "pop100"
    print(f"  BEST  n={v['n']:<2} : {A}/{ta} {v['a_mean']:,.0f} vs {B}/{tb} {v['b_mean']:,.0f}"
          f" | {A} wins {v['a_wins']}/{v['n']} | two-sided p={v['p_two']:.4g} {sig(v['p_two'])}"
          f" | one-sided p({A}<{B})={v['p_a_better']:.4g}")


def main():
    data = collect()
    allres = {}
    for inst in INSTANCES:
        print("\n" + "=" * 78)
        print(f"  {inst.upper()}   (lower composite_cost = better)")
        print("=" * 78)
        rA = compare(data, inst, "nomut", "plain")
        rB = compare(data, inst, "mut", "nomut")
        show("(A) NOVELTY no-mutation  vs  PLAIN SHADE", "nomut", "plain", rA)
        show("(B) WITH-mutation  vs  WITHOUT-mutation", "mut", "nomut", rB)
        allres[inst] = {"nomut_vs_plain": rA, "mut_vs_nomut": rB}

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "wilcoxon_method_comparison.json")
    with open(path, "w") as f:
        json.dump(allres, f, indent=2)
    print("\nSaved -> " + path)


if __name__ == "__main__":
    main()
