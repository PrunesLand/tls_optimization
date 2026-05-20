"""
IRRG — Incremental Recursive Ranking Grouping for Traffic Light Optimization

Discovers which traffic light phase durations interact with each other using
monotonicity checking, then groups them for cooperative co-evolution.

Loads data from build_traffic_fitness_wrapper() (same as DG2_grouping.py).
Parallel fitness evaluations via ProcessPoolExecutor — each worker runs its
own SUMO instance. Results saved to src/outputs/irrg_decomposition.json.
"""

import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import BASELINE_TRAFFIC_DATA, NUM_PROCESSORS
from src.genetic_algorithm.fitness_evaluation import fitness_function as _traffic_fitness
from src.decomposition.DG2_grouping import build_traffic_fitness_wrapper

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MU_M = float(np.finfo(np.float64).eps)


@dataclass
class IRRGConfig:
    ns: int = 10          # samples per ranking (higher = more sensitive, more FFEs)
    sti: int = 15         # stop after this many stale RRG iterations
    s: int = 100          # max size of separable variable groups
    shade_fes: int = 5000
    mts_fes: int = 15000
    seed: int = 42
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# FITNESS WRAPPER
# ---------------------------------------------------------------------------

def _eval_single(args):
    """Module-level worker function — required for ProcessPoolExecutor pickling."""
    wrapper, vector = args
    return float(wrapper(vector))


class IRRGFitnessWrapper:
    """Wraps TrafficFitnessWrapper with FFE tracking and parallel batch eval."""

    def __init__(self, traffic_wrapper, lb, ub):
        self.traffic_wrapper = traffic_wrapper
        self.lb, self.ub = lb.copy(), ub.copy()
        self.n = len(lb)
        self.fes = 0

    def __call__(self, x):
        self.fes += 1
        return float(self.traffic_wrapper(np.clip(x, self.lb, self.ub)))

    def batch_evaluate(self, vectors, pool):
        """Evaluate a list of vectors in parallel. Increments FFE counter."""
        tasks = [(self.traffic_wrapper, np.clip(v, self.lb, self.ub)) for v in vectors]
        results = list(pool.map(_eval_single, tasks))
        self.fes += len(results)
        return results


# ---------------------------------------------------------------------------
# MATH HELPERS
# ---------------------------------------------------------------------------

def _eps(y1, y2, n):
    """Scale-relative epsilon for floating-point comparison between two fitness values."""
    k = math.sqrt(n) + 1.0
    gamma = (k * MU_M) / (1.0 - k * MU_M)
    return gamma * (abs(y1) + abs(y2))


def _sgn(x, eps):
    """Epsilon-aware sign: -1, 0, or +1."""
    return 0 if abs(x) <= eps else (1 if x > eps else -1)


def enforce_transitivity(theta):
    """Close the interaction matrix under transitivity (boolean OR-multiply until stable)."""
    t = theta.astype(bool)
    while True:
        t_new = t | (t @ t).astype(bool)
        if np.array_equal(t_new, t):
            break
        t = t_new
    theta[:] = t.astype(theta.dtype)
    return theta


# ---------------------------------------------------------------------------
# RANKING FUNCTIONS  (parallelized via pool)
# ---------------------------------------------------------------------------

def _build_samples(lb, ub, ns, n, rng):
    """ns × n matrix: ns evenly-spaced, independently shuffled values per variable."""
    X = np.zeros((ns, n))
    for j in range(n):
        vals = np.linspace(lb[j], ub[j], ns)
        rng.shuffle(vals)
        X[:, j] = vals
    return X


def _first_ranking(idx, x_hq, X_bar, f, ns, pool=None):
    """
    Sweep ns samples over variables `idx`, holding everything else at x_hq.
    Returns (y1_bar, r1) — fitness values and their ascending sort order.
    Evaluations run in parallel if pool is provided.
    """
    vecs = []
    for i in range(ns):
        x = x_hq.copy()
        x[idx] = X_bar[i, idx]
        vecs.append(x)
    y = np.array(f.batch_evaluate(vecs, pool) if pool else [f(v) for v in vecs])
    return y, np.argsort(y)


def _second_ranking_check(idx1, idx2, x_hq, X_bar, x2_bar, y1, r1, f, ns, n, pool=None):
    """
    Re-evaluate the ns X1 samples with X2 perturbed to x2_bar.
    Returns True if the relative ordering of any consecutive pair flipped
    (= interaction detected between idx1 and idx2).
    """
    x_base = x_hq.copy()
    x_base[idx2] = x2_bar[idx2]
    vecs = []
    for i in range(ns):
        x = x_base.copy()
        x[idx1] = X_bar[r1[i], idx1]
        vecs.append(x)
    # y2 is built in r1-sorted order: y2[0] corresponds to r1[0], y2[1] to r1[1], etc.
    y2 = f.batch_evaluate(vecs, pool) if pool else [f(v) for v in vecs]

    # Track the last y2 value for which the y1 gap was distinguishable.
    # We must not compare y2[i] against a y2[prev] that was part of an
    # indistinguishable y1 pair — that comparison would be meaningless.
    # Paper Pseudocode 3 handles this by computing y2 lazily and only
    # advancing y2_prev when the y1 gap is real. We replicate that logic here.
    y2_prev = y2[0]
    y2_prev_valid = False   # becomes True once we have one real y1 gap to anchor against

    for i in range(1, ns):
        e1 = _eps(y1[r1[i]], y1[r1[i - 1]], n)
        if _sgn(y1[r1[i]] - y1[r1[i - 1]], e1) == 0:
            # y1 gap is indistinguishable — advance y2_prev but do not flip-check
            y2_prev = y2[i]
            continue

        if not y2_prev_valid:
            # First real y1 gap: anchor y2_prev and move on without checking
            y2_prev = y2[i - 1]
            y2_prev_valid = True

        e2 = _eps(y2[i], y2_prev, n)
        if _sgn(y2[i] - y2_prev, e2) < 0:
            return True                       # ordering flipped — interaction!

        y2_prev = y2[i]
    return False


# ---------------------------------------------------------------------------
# INTERACTION DETECTION
# ---------------------------------------------------------------------------

def _consider_V(V, G, x_hq, X_bar, x2_bar, f, ns, n, rng, pool):
    """
    Quick check: should unassigned variables V be included this RRG pass?
    Returns True if any of four conditions holds (see module docstring).
    """
    if not G:       return True   # A: no groups yet
    if not V:       return False
    if len(V) == 1: return True   # B: single variable

    Vs = list(V); rng.shuffle(Vs)
    V1, V2 = Vs[:len(Vs)//2], Vs[len(Vs)//2:]
    for a, b in [(V1, V2), (V2, V1)]:          # C: V vs itself
        y, r = _first_ranking(a, x_hq, X_bar, f, ns, pool)
        if _second_ranking_check(a, b, x_hq, X_bar, x2_bar, y, r, f, ns, n, pool):
            return True

    y_V, r_V = _first_ranking(V, x_hq, X_bar, f, ns, pool)
    for g in G:                                  # D: V vs each existing group
        if _second_ranking_check(V, g, x_hq, X_bar, x2_bar, y_V, r_V, f, ns, n, pool):
            return True
        y_g, r_g = _first_ranking(g, x_hq, X_bar, f, ns, pool)
        if _second_ranking_check(g, V, x_hq, X_bar, x2_bar, y_g, r_g, f, ns, n, pool):
            return True
    return False


def _interact(G1, G2, x_hq, X_bar, x2_bar, y1, r1, f, ns, n, pool):
    """
    Recursively bisect G2 to find which sub-groups interact with G1.
    Returns extended G1 containing all interacting groups from G2.
    """
    X1 = [i for g in G1 for i in g]
    X2 = [i for g in G2 for i in g]
    if not _second_ranking_check(X1, X2, x_hq, X_bar, x2_bar, y1, r1, f, ns, n, pool):
        return list(G1)
    if len(G2) == 1:
        return list(G1) + [G2[0]]
    mid = len(G2) // 2
    G1 = _interact(G1, G2[:mid], x_hq, X_bar, x2_bar, y1, r1, f, ns, n, pool)
    G1 = _interact(G1, G2[mid:], x_hq, X_bar, x2_bar, y1, r1, f, ns, n, pool)
    return G1


# ---------------------------------------------------------------------------
# RRG  — one full sweep
# ---------------------------------------------------------------------------

def rrg(x_hq, x2_bar, theta, f, n, lb, ub, ns, rng, log, pool=None):
    """
    One Recursive Ranking Grouping pass.
    Returns a list of newly discovered interacting variable groups.
    """
    NonSeps = []

    # Build G (known groups) and V (unassigned) from current theta
    G, assigned = [], set()
    for v in range(n):
        if v in assigned: continue
        inters = [j for j in range(n) if theta[v, j] and j != v]
        if inters:
            grp = list(set([v] + inters)); rng.shuffle(grp)
            G.append(grp); assigned.update(grp)
    V = [v for v in range(n) if v not in assigned]

    X_bar = _build_samples(lb, ub, ns, n, rng)

    if _consider_V(V, G, x_hq, X_bar, x2_bar, f, ns, n, rng, pool):
        G += [[v] for v in V]

    if not G: return NonSeps

    rng.shuffle(G)
    G1, G2 = [G[0]], G[1:]

    while G2:
        X1 = [i for g in G1 for i in g]
        y1, r1 = _first_ranking(X1, x_hq, X_bar, f, ns, pool)
        G1_new = _interact(G1, G2, x_hq, X_bar, x2_bar, y1, r1, f, ns, n, pool)

        if len(G1_new) == len(G1):          # G1 not extended
            if len(G1) == 1:
                # Missing-linkage mitigation: shrink G1 so weaker signals emerge
                min_g2 = min(len(g) for g in G2)
                if len(G1[0]) >= max(min_g2, 2):
                    keep = len(G1[0]) // 2
                    G1 = [list(rng.choice(G1[0], size=keep, replace=False))]
                else:
                    G1, G2 = [G2[0]], G2[1:]
            else:
                NonSeps.append([i for g in G1 for i in g])
                G1, G2 = [G2[0]], G2[1:]
        else:
            added = G1_new[len(G1):]
            # Use frozenset comparison so groups are matched by membership,
            # not by list order — groups may have been shuffled since they
            # were added to G2, making plain list equality unreliable.
            added_sets = {frozenset(g) for g in added}
            G2 = [g for g in G2 if frozenset(g) not in added_sets]
            G1 = G1_new

    if len(G1) > 1:
        NonSeps.append([i for g in G1 for i in g])
    return NonSeps


# ---------------------------------------------------------------------------
# IRRG  — incremental outer loop
# ---------------------------------------------------------------------------

def irrg(f, n, lb, ub, theta_init, config, rng, log, n_workers=None):
    """
    Runs RRG repeatedly with fresh random x2_bar until sti consecutive passes
    find nothing new. Returns (Seps, NonSeps) variable groupings.
    """
    n_workers = n_workers or os.cpu_count() or 1
    log.info(f"Workers: {n_workers}")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:

        # Step 1 — find high-quality starting point
        log.info("SHADE search..."); fes0 = f.fes
        x_hq = _run_shade(f, n, lb, ub, config.shade_fes, rng, pool)
        log.info(f"  done ({f.fes - fes0} FFEs)")

        log.info("MTS-LS1 refinement..."); fes0 = f.fes
        x_hq = _run_mts(f, x_hq, lb, ub, config.mts_fes, rng)
        log.info(f"  done ({f.fes - fes0} FFEs)")

        # Step 2 — initialise theta
        theta = theta_init.copy().astype(np.int32)
        np.fill_diagonal(theta, 1)
        enforce_transitivity(theta)

        # Step 3 — incremental RRG
        stale, first, it = 0, True, 0
        while True:
            it += 1
            before = theta.sum()
            x2_bar = rng.uniform(lb, ub)
            for grp in rrg(x_hq, x2_bar, theta, f, n, lb, ub, config.ns, rng, log, pool):
                for i in grp:
                    for j in grp:
                        theta[i, j] = theta[j, i] = 1
            enforce_transitivity(theta)
            new = theta.sum() - before
            log.info(f"Iter {it:3d} | new interactions: {new:4d} | FFEs: {f.fes} | stale: {stale}")
            if new == 0:
                stale += 1
                if first or stale >= config.sti: break
            else:
                stale = 0
            first = False

    # Step 4 — build output groups
    assigned, NonSeps, sep_vars = set(), [], []
    for v in range(n):
        if v in assigned: continue
        grp = [j for j in range(n) if theta[v, j] and j != v]
        if grp:
            full = sorted(set([v] + grp)); NonSeps.append(full); assigned.update(full)
        else:
            sep_vars.append(v)
    Seps = [sep_vars[i:i + config.s] for i in range(0, len(sep_vars), config.s)]
    log.info(f"Done: {len(NonSeps)} interacting groups, {len(Seps)} separable groups")
    return Seps, NonSeps


# ---------------------------------------------------------------------------
# OPTIMISER STUBS  (replace with real SHADE / MTS-LS1 for production)
# ---------------------------------------------------------------------------

def _run_shade(f, n, lb, ub, max_fes, rng, pool):
    """
    Placeholder: batched random search.
    Replace with full SHADE (success-history DE with current-to-pbest mutation).
    """
    best_x = rng.uniform(lb, ub); best_fit = f(best_x); used = 1
    batch = pool._max_workers * 2
    while used < max_fes:
        cands = [rng.uniform(lb, ub) for _ in range(min(batch, max_fes - used))]
        for c, fit in zip(cands, f.batch_evaluate(cands, pool)):
            if fit < best_fit: best_fit, best_x = fit, c.copy()
        used += len(cands)
    return best_x


def _run_mts(f, x_start, lb, ub, max_fes, rng):
    """
    Placeholder: coordinate-wise local search with step halving.
    Replace with full MTS-LS1 (per-variable adaptive step sizes).
    """
    n = len(x_start); x = x_start.copy(); best = f(x); used = 1
    SR = 0.2 * (ub - lb)
    while used < max_fes:
        improved = False
        for i in rng.permutation(n):
            if used >= max_fes: break
            for sign in (1, -1):
                trial = x.copy(); trial[i] = np.clip(x[i] + sign * SR[i], lb[i], ub[i])
                fit = f(trial); used += 1
                if fit < best: x, best, improved = trial, fit, True; break
            else:
                SR[i] = max(SR[i] * 0.5, 1e-15)
        if not improved: break
    return x


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("IRRG — traffic light interaction detection")
    print("=" * 55)

    with open(BASELINE_TRAFFIC_DATA) as fh:
        baseline_data = json.load(fh)

    traffic_wrapper, n, lb, ub, labels = build_traffic_fitness_wrapper(
        baseline_data=baseline_data,
        fitness_function=_traffic_fitness,
    )

    config = IRRGConfig()
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("IRRG")
    rng = np.random.default_rng(config.seed)
    n_workers = NUM_PROCESSORS or os.cpu_count()

    f = IRRGFitnessWrapper(traffic_wrapper, lb, ub)
    theta_init = np.eye(n, dtype=np.int32)

    log.info(f"n={n} variables | ns={config.ns} | sti={config.sti} | workers={n_workers}")
    t0 = time.time()
    Seps, NonSeps = irrg(f, n, lb, ub, theta_init, config, rng, log, n_workers)
    elapsed = time.time() - t0

    print(f"\nFFEs: {f.fes} | time: {elapsed:.1f}s | workers: {n_workers}")
    print(f"Separable groups  : {len(Seps)}  (sizes: {[len(g) for g in Seps[:8]]}{'...' if len(Seps)>8 else ''})")
    print(f"Interacting groups: {len(NonSeps)}  (sizes: {[len(g) for g in NonSeps[:8]]}{'...' if len(NonSeps)>8 else ''})")
    print(f"Variables covered : {sum(len(g) for g in Seps+NonSeps)} / {n}")

    out_dir = Path("src/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "n": n, "total_fes": f.fes, "elapsed_seconds": round(elapsed, 3),
        "n_workers": n_workers, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": vars(config), "gene_labels": labels,
        "separable_groups": Seps, "interacting_groups": NonSeps,
    }
    out_path = out_dir / "irrg_decomposition.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved → {out_path}")