"""
LT-GOMEA Paired — Optimal Mixing restricted to size-2 clusters only.

For each pair: the downstream TLS ("second") may get a slightly higher
duration; the upstream ("first") keeps baseline. All unpaired TLS stay unchanged.

Usage:  python -m src.pygad.lt_gomea_paired_optimizer
"""
import json, copy, time, os, sys
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (
    BASELINE_TRAFFIC_DATA, NUM_PROCESSORS,
    LT_GOMEA_POPULATION_SIZE, LT_GOMEA_NUM_GENERATIONS,
    LT_GOMEA_BASELINE_NOISE_STD,
    CLUSTER_THRESHOLD_FASTEST, CLUSTER_THRESHOLD_SHORTEST, CLUSTER_THRESHOLD_EUCLIDIAN,
)
from src.genetic_algorithm.fitness_evaluation import fitness_function as _traffic_fitness
from src.decomposition.DG2_grouping import build_traffic_fitness_wrapper

THRESHOLDS = {
    "shortest":  CLUSTER_THRESHOLD_SHORTEST,
    "euclidian": CLUSTER_THRESHOLD_EUCLIDIAN,
    "fastest":   CLUSTER_THRESHOLD_FASTEST,
}

GENE_LOW, GENE_HIGH = 5.0, 85.0
MAX_DOWNSTREAM_OFFSET = 5.0   # max extra seconds the downstream TLS may add


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_pairs(distance_json, threshold):
    """Cluster with Ward, return only size-2 clusters with upstream/downstream roles."""
    with open(distance_json) as f:
        data = json.load(f)

    key = 'distance_matrix' if 'distance_matrix' in data else 'travel_time_matrix'
    matrix = data[key]
    ids = [t['id'] for t in data['traffic_lights']]
    n = len(ids)

    vals = [v for row in matrix.values() for v in row.values() if v is not None]
    penalty = max(vals) * 1.5 if vals else 1e6

    arr = np.zeros((n, n))
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            arr[i, j] = matrix[a].get(b) if matrix[a].get(b) is not None else penalty
    arr = (arr + arr.T) / 2
    np.fill_diagonal(arr, 0)

    labels = fcluster(linkage(squareform(arr), method='ward'),
                      t=threshold, criterion='distance')

    # Group IDs by cluster, keep only pairs
    clusters = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(ids[idx])

    pairs = []
    for members in clusters.values():
        if len(members) != 2:
            continue
        a, b = members
        d_ab = matrix.get(a, {}).get(b) or 0.0
        d_ba = matrix.get(b, {}).get(a) or 0.0
        # downstream = harder to reach (higher travel time TO it)
        if d_ab >= d_ba:
            pairs.append({"first": a, "second": b})
        else:
            pairs.append({"first": b, "second": a})

    return pairs, len(clusters)


def _build_gene_map(baseline_data):
    """Map each TLS to its gene indices. Returns (tls_to_genes, num_genes, baseline_vec)."""
    tls_to_genes, idx, baseline = {}, 0, []
    for tls_id in sorted(baseline_data["tls_data"]):
        phases = sorted(baseline_data["tls_data"][tls_id])
        tls_to_genes[tls_id] = (idx, idx + len(phases))
        for pk in phases:
            baseline.append(float(baseline_data["tls_data"][tls_id][pk]["duration"]))
        idx += len(phases)
    return tls_to_genes, idx, np.array(baseline)


def _gene_indices(tls_ids, tls_to_genes):
    """TLS IDs → flat list of gene indices."""
    out = []
    for tid in tls_ids:
        if tid in tls_to_genes:
            s, e = tls_to_genes[tid]
            out.extend(range(s, e))
    return out


# ── Fitness (parallel) ─────────────────────────────────────────────────────

def _eval(args):
    wrapper, sol, i = args
    return i, float(wrapper(sol))

def _eval_pop(wrapper, pop, n_workers):
    fit = np.full(len(pop), np.inf)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_eval, (wrapper, pop[i], i)): i for i in range(len(pop))}
        for f in as_completed(futs):
            i, v = f.result()
            fit[i] = v
    return fit


def _mix(args):
    """Optimal mixing: swap paired genes from donor, enforce constraints."""
    wrapper, src, src_fit, donor, mask, first_idx, second_idx, bvec = args
    child = src.copy()
    for i in mask:
        child[i] = donor[i]

    # First (upstream) stays at baseline
    for i in first_idx:
        child[i] = bvec[i]
    # Second (downstream) may be equal or slightly above baseline
    for i in second_idx:
        child[i] = np.clip(child[i], bvec[i], bvec[i] + MAX_DOWNSTREAM_OFFSET)

    child = np.clip(child, GENE_LOW, GENE_HIGH)
    child_fit = float(wrapper(child))
    return (child, child_fit, True) if child_fit < src_fit else (src, src_fit, False)


# ── Core loop ──────────────────────────────────────────────────────────────

def run_paired_gomea(tree_name, dist_path, strategy, baseline_data,
                     wrapper, num_genes, baseline_vec, tls_to_genes,
                     pop_size, num_gen, noise_std, n_workers, seed=42):
    rng = np.random.default_rng(seed)
    threshold = THRESHOLDS[tree_name]

    print(f"\n{'='*60}")
    print(f"PAIRED | {tree_name} (t={threshold}) | {strategy} | pop={pop_size}")
    print(f"{'='*60}")

    pairs, total_clusters = _load_pairs(dist_path, threshold)
    print(f"Clusters: {total_clusters} | Pairs: {len(pairs)}")

    if not pairs:
        print("WARNING: No size-2 clusters — returning baseline.")
        out = copy.deepcopy(baseline_data)
        out["composite_cost"] = float("inf")
        return {"best_configuration": out, "best_fitness": float("inf"),
                "fitness_history": [], "time_s": 0.0, "tree": tree_name,
                "threshold": threshold, "strategy": strategy,
                "pop_size": pop_size, "generations": num_gen,
                "num_pairs": 0, "seed": seed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    for p in pairs:
        print(f"  {p['first']} (upstream) ↔ {p['second']} (downstream)")

    # Gene indices per role
    first_idx = _gene_indices([p["first"] for p in pairs], tls_to_genes)
    second_idx = _gene_indices([p["second"] for p in pairs], tls_to_genes)
    gene_masks = [_gene_indices([p["first"], p["second"]], tls_to_genes) for p in pairs]
    gene_masks = [m for m in gene_masks if m]

    # Init population (all baseline, then perturb paired genes)
    pop = np.tile(baseline_vec, (pop_size, 1))
    for idx in second_idx:
        pop[:, idx] = baseline_vec[idx] + rng.uniform(0, MAX_DOWNSTREAM_OFFSET, pop_size)
    pop = np.clip(pop, GENE_LOW, GENE_HIGH)

    fit = _eval_pop(wrapper, pop, n_workers)
    best_i = np.argmin(fit)
    best_sol, best_fit = pop[best_i].copy(), fit[best_i]
    print(f"Gen 0 | Best: {best_fit:.2f} | Mean: {np.mean(fit):.2f}")
    history = [{"gen": 0, "best": float(best_fit), "mean": float(np.mean(fit))}]

    t0 = time.time()
    for gen in range(1, num_gen + 1):
        tasks = []
        for i in range(pop_size):
            di = i
            while di == i:
                di = rng.integers(0, pop_size)
            mask = gene_masks[rng.integers(0, len(gene_masks))]
            tasks.append((wrapper, pop[i].copy(), fit[i], pop[di].copy(),
                          mask, first_idx, second_idx, baseline_vec))

        improvements = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_mix, t): idx for idx, t in enumerate(tasks)}
            for f in as_completed(futs):
                idx = futs[f]
                new_sol, new_fit, improved = f.result()
                if improved:
                    pop[idx], fit[idx] = new_sol, new_fit
                    improvements += 1

        gi = np.argmin(fit)
        if fit[gi] < best_fit:
            best_fit, best_sol = fit[gi], pop[gi].copy()

        history.append({"gen": gen, "best": float(best_fit),
                        "mean": float(np.mean(fit)), "improved": improvements})
        print(f"Gen {gen:2d} | Best: {best_fit:.2f} | Mean: {np.mean(fit):.2f} | +{improvements}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s | Final best: {best_fit:.2f}")

    # Rebuild JSON — only paired TLS get new durations
    best_json = _rebuild_json(best_sol, baseline_data, tls_to_genes, pairs)
    best_json["composite_cost"] = float(best_fit)

    return {
        "best_configuration": best_json, "best_fitness": float(best_fit),
        "fitness_history": history, "time_s": round(elapsed, 2),
        "tree": tree_name, "threshold": threshold, "strategy": strategy,
        "pop_size": pop_size, "generations": num_gen,
        "num_pairs": len(pairs), "pairs": pairs,
        "seed": seed, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _rebuild_json(sol, baseline, tls_to_genes, pairs):
    """Gene vector → TLS JSON. Only paired TLS are updated; the rest stay baseline."""
    out = copy.deepcopy(baseline)
    paired_ids = {p["first"] for p in pairs} | {p["second"] for p in pairs}

    for tls_id in sorted(out["tls_data"]):
        if tls_id not in paired_ids or tls_id not in tls_to_genes:
            continue

        s, e = tls_to_genes[tls_id]
        raw = sol[s:e]
        keys = sorted(out["tls_data"][tls_id])
        n = len(keys)

        total = sum(raw)
        if total <= 0:
            dur = [90 // n] * n
            dur[-1] += 90 - sum(dur)
        else:
            dur = [max(1, int(round(d * 90 / total))) for d in raw]
            diff = 90 - sum(dur)
            if diff:
                dur[int(np.argmax(dur))] += diff

        for i, pk in enumerate(keys):
            out["tls_data"][tls_id][pk]["duration"] = int(dur[i])
    return out


# ── Experiment runner ──────────────────────────────────────────────────────

def run_all_experiments():
    with open(BASELINE_TRAFFIC_DATA) as f:
        baseline = json.load(f)

    wrapper, num_genes, _, _, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline, fitness_function=_traffic_fitness)
    tls_to_genes, _, baseline_vec = _build_gene_map(baseline)
    n_workers = NUM_PROCESSORS or os.cpu_count() or 1

    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "src" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = {
        "shortest":  out_dir / "tls_distances_shortest.json",
        "euclidian": out_dir / "tls_distances_euclidian.json",
        "fastest":   out_dir / "tls_distances_fastest.json",
    }
    summary = {}

    for tree_name, path in trees.items():
        for strat in ["random", "baseline", "mixed"]:
            label = f"{tree_name}_{strat}"
            try:
                res = run_paired_gomea(
                    tree_name, str(path), strat, baseline,
                    wrapper, num_genes, baseline_vec, tls_to_genes,
                    LT_GOMEA_POPULATION_SIZE, LT_GOMEA_NUM_GENERATIONS,
                    LT_GOMEA_BASELINE_NOISE_STD, n_workers)

                out_file = out_dir / f"lt_gomea_paired_{label}.json"
                with open(out_file, "w") as f:
                    json.dump(res, f, indent=4)
                print(f"Saved → {out_file}")
                summary[label] = {"best": res["best_fitness"],
                                  "time_s": res["time_s"],
                                  "pairs": res["num_pairs"]}
            except Exception as e:
                print(f"ERROR [{label}]: {e}")
                import traceback; traceback.print_exc()
                summary[label] = {"error": str(e)}

    # Results table
    print(f"\n{'Tree':<15} {'Strategy':<10} {'Pairs':>6} {'Best':>12} {'Time':>8}")
    print("─" * 55)
    for label, info in summary.items():
        t, s = label.rsplit("_", 1)
        if "error" in info:
            print(f"{t:<15} {s:<10} {'':>6} {'ERROR':>12}")
        else:
            print(f"{t:<15} {s:<10} {info['pairs']:>6} {info['best']:>12.2f} {info['time_s']:>7.1f}s")


if __name__ == "__main__":
    run_all_experiments()
