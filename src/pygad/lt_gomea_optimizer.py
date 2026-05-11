"""
LT-GOMEA — Linkage Tree Optimal Mixing for traffic light optimization.

Uses threshold-based clusters from the distance dendrograms as masks.
Runs 9 experiments: 3 linkage trees × 3 population strategies.

Usage:  python -m src.pygad.lt_gomea_optimizer
"""
from config import CLUSTER_THRESHOLD_FASTEST
from config import CLUSTER_THRESHOLD_SHORTEST
from config import CLUSTER_THRESHOLD_EUCLIDIAN
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
)
from src.genetic_algorithm.fitness_evaluation import fitness_function as _traffic_fitness
from src.decomposition.DG2_grouping import build_traffic_fitness_wrapper

# Thresholds matching the dendrogram plots (from plot_dendrograms.py)
THRESHOLDS = {
    "shortest":  CLUSTER_THRESHOLD_SHORTEST,
    "euclidian": CLUSTER_THRESHOLD_EUCLIDIAN,
    "fastest":   CLUSTER_THRESHOLD_FASTEST,
}

GENE_LOW, GENE_HIGH = 5.0, 85.0


# ── Linkage tree → threshold masks ──────────────────────────────────────────

def load_threshold_masks(distance_json, threshold):
    """
    Load a distance matrix, cluster with Ward's method, cut at the given
    threshold, and return a list of masks. Each mask is a list of TLS IDs
    that belong to the same cluster.
    """
    with open(distance_json) as f:
        data = json.load(f)

    key = 'distance_matrix' if 'distance_matrix' in data else 'travel_time_matrix'
    matrix = data[key]
    ids = [t['id'] for t in data['traffic_lights']]
    n = len(ids)

    # Build symmetric distance array
    vals = [v for row in matrix.values() for v in row.values() if v is not None]
    penalty = max(vals) * 1.5 if vals else 1e6

    arr = np.zeros((n, n))
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            arr[i, j] = matrix[a].get(b) if matrix[a].get(b) is not None else penalty
    arr = (arr + arr.T) / 2
    np.fill_diagonal(arr, 0)

    Z = linkage(squareform(arr), method='ward')
    labels = fcluster(Z, t=threshold, criterion='distance')

    # Group TLS IDs by cluster label
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(ids[idx])

    masks = list(clusters.values())
    return masks


# ── Gene mapping ────────────────────────────────────────────────────────────

def build_gene_map(baseline_data):
    """
    Returns:
        tls_to_genes: {tls_id: (start, end)} in the flat gene vector
        num_genes:    total gene count
        baseline_vec: numpy array of baseline durations
    """
    tls_to_genes = {}
    idx = 0
    baseline = []

    for tls_id in sorted(baseline_data["tls_data"]):
        phases = sorted(baseline_data["tls_data"][tls_id])
        tls_to_genes[tls_id] = (idx, idx + len(phases))
        for pk in phases:
            baseline.append(float(baseline_data["tls_data"][tls_id][pk]["duration"]))
        idx += len(phases)

    return tls_to_genes, idx, np.array(baseline)


def mask_to_gene_indices(mask_tls_ids, tls_to_genes):
    """Convert a list of TLS IDs to a flat list of gene indices."""
    out = []
    for tls_id in mask_tls_ids:
        if tls_id in tls_to_genes:
            s, e = tls_to_genes[tls_id]
            out.extend(range(s, e))
    return out


# ── Population init ─────────────────────────────────────────────────────────

def init_population(strategy, n, num_genes, baseline_vec, noise_std, rng):
    """Create initial population: 'random', 'baseline', or 'mixed'."""
    if strategy == "random":
        return rng.uniform(GENE_LOW, GENE_HIGH, (n, num_genes))

    elif strategy == "baseline":
        pop = np.tile(baseline_vec, (n, 1))
        pop += rng.normal(0, noise_std, pop.shape) * pop
        return np.clip(pop, GENE_LOW, GENE_HIGH)

    elif strategy == "mixed":
        half = n // 2
        rand = rng.uniform(GENE_LOW, GENE_HIGH, (half, num_genes))
        base = np.tile(baseline_vec, (n - half, 1))
        base += rng.normal(0, noise_std, base.shape) * base
        return np.vstack([rand, np.clip(base, GENE_LOW, GENE_HIGH)])

    raise ValueError(f"Unknown strategy: {strategy}")


# ── Fitness helpers (parallel) ──────────────────────────────────────────────

def _eval(args):
    wrapper, sol, i = args
    return i, float(wrapper(sol))


def eval_pop(wrapper, pop, n_workers):
    """Evaluate entire population in parallel. Returns fitness array."""
    fit = np.full(len(pop), np.inf)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_eval, (wrapper, pop[i], i)): i for i in range(len(pop))}
        for f in as_completed(futs):
            i, v = f.result()
            fit[i] = v
    return fit


def _mix(args):
    """Optimal mixing for one individual."""
    wrapper, src, src_fit, donor, mask = args
    child = src.copy()
    for i in mask:
        child[i] = donor[i]
    child_fit = float(wrapper(child))
    if child_fit < src_fit:
        return child, child_fit, True
    return src, src_fit, False


# ── Core GOMEA loop ─────────────────────────────────────────────────────────

def run_lt_gomea(tree_name, dist_path, strategy, baseline_data,
                 wrapper, num_genes, baseline_vec, tls_to_genes,
                 pop_size, num_gen, noise_std, n_workers, seed=42):
    """Run one LT-GOMEA experiment. Returns results dict."""
    rng = np.random.default_rng(seed)
    threshold = THRESHOLDS[tree_name]

    print(f"\n{'='*60}")
    print(f"Tree: {tree_name} (t={threshold}) | Strategy: {strategy} | Pop: {pop_size}")
    print(f"{'='*60}")

    # 1. Build masks from threshold clusters
    tls_masks = load_threshold_masks(dist_path, threshold)
    gene_masks = [mask_to_gene_indices(m, tls_to_genes) for m in tls_masks]
    gene_masks = [m for m in gene_masks if m]  # drop empty
    print(f"Masks: {len(gene_masks)} clusters, sizes {[len(m) for m in gene_masks]}")

    # 2. Init & evaluate population
    pop = init_population(strategy, pop_size, num_genes, baseline_vec, noise_std, rng)
    fit = eval_pop(wrapper, pop, n_workers)

    best_i = np.argmin(fit)
    best_sol, best_fit = pop[best_i].copy(), fit[best_i]
    print(f"Gen 0 | Best: {best_fit:.2f} | Mean: {np.mean(fit):.2f}")

    history = [{"gen": 0, "best": float(best_fit), "mean": float(np.mean(fit))}]

    # 3. Generational loop
    t0 = time.time()
    for gen in range(1, num_gen + 1):
        # Build mixing tasks
        tasks = []
        for i in range(pop_size):
            donor_i = i
            while donor_i == i:
                donor_i = rng.integers(0, pop_size)
            mask = gene_masks[rng.integers(0, len(gene_masks))]
            tasks.append((wrapper, pop[i].copy(), fit[i], pop[donor_i].copy(), mask))

        # Run mixing in parallel
        improvements = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_mix, t): idx for idx, t in enumerate(tasks)}
            for f in as_completed(futs):
                idx = futs[f]
                new_sol, new_fit, improved = f.result()
                if improved:
                    pop[idx] = new_sol
                    fit[idx] = new_fit
                    improvements += 1

        # Track best
        gi = np.argmin(fit)
        if fit[gi] < best_fit:
            best_fit, best_sol = fit[gi], pop[gi].copy()

        history.append({"gen": gen, "best": float(best_fit),
                        "mean": float(np.mean(fit)), "improved": improvements})
        print(f"Gen {gen:2d} | Best: {best_fit:.2f} | Mean: {np.mean(fit):.2f} | +{improvements}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s | Final best: {best_fit:.2f}")

    # 4. Reconstruct result JSON
    best_json = _rebuild_json(best_sol, baseline_data, tls_to_genes)
    best_json["composite_cost"] = float(best_fit)

    return {
        "best_configuration": best_json,
        "best_fitness": float(best_fit),
        "fitness_history": history,
        "time_s": round(elapsed, 2),
        "tree": tree_name, "threshold": threshold,
        "strategy": strategy, "pop_size": pop_size,
        "generations": num_gen, "num_masks": len(gene_masks),
        "seed": seed, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _rebuild_json(sol, baseline, tls_to_genes):
    """Convert flat gene vector back to the full TLS JSON format."""
    out = copy.deepcopy(baseline)
    for tls_id in sorted(out["tls_data"]):
        if tls_id not in tls_to_genes:
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


# ── Experiment runner ───────────────────────────────────────────────────────

def run_all_experiments():
    with open(BASELINE_TRAFFIC_DATA) as f:
        baseline = json.load(f)

    wrapper, num_genes, _, _, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline, fitness_function=_traffic_fitness)
    tls_to_genes, _, baseline_vec = build_gene_map(baseline)
    n_workers = NUM_PROCESSORS or os.cpu_count() or 1

    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "src" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = {
        "shortest":  out_dir / "tls_distances_shortest.json",
        "euclidian": out_dir / "tls_distances_euclidian.json",
        "fastest":   out_dir / "tls_distances_fastest.json",
    }
    strategies = ["random", "baseline", "mixed"]
    summary = {}

    for tree_name, path in trees.items():
        for strat in strategies:
            label = f"{tree_name}_{strat}"
            try:
                res = run_lt_gomea(
                    tree_name, str(path), strat, baseline,
                    wrapper, num_genes, baseline_vec, tls_to_genes,
                    LT_GOMEA_POPULATION_SIZE, LT_GOMEA_NUM_GENERATIONS,
                    LT_GOMEA_BASELINE_NOISE_STD, n_workers)

                out_file = out_dir / f"lt_gomea_{label}.json"
                with open(out_file, "w") as f:
                    json.dump(res, f, indent=4)
                print(f"Saved → {out_file}")
                summary[label] = {"best": res["best_fitness"], "time_s": res["time_s"]}

            except Exception as e:
                print(f"ERROR [{label}]: {e}")
                import traceback; traceback.print_exc()
                summary[label] = {"error": str(e)}

    # Print results table
    print(f"\n{'Tree':<15} {'Strategy':<10} {'Best':>12} {'Time':>8}")
    print("─" * 47)
    for label, info in summary.items():
        t, s = label.rsplit("_", 1)
        if "error" in info:
            print(f"{t:<15} {s:<10} {'ERROR':>12}")
        else:
            print(f"{t:<15} {s:<10} {info['best']:>12.2f} {info['time_s']:>7.1f}s")


if __name__ == "__main__":
    run_all_experiments()
