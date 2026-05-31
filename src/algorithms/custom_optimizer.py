"""
Custom Optimier — Linkage Tree Optimal Mixing for traffic light optimization.

1. Optimal-mixing masks now include EVERY internal Ward node whose merge
   distance ≤ threshold (not just the top-level fcluster partition).
   The root node is always excluded.  Example: if the tree contains

       root  (1,2,3,4)   ← always excluded
       node  (1,2,3)     ← included when merge_dist ≤ threshold
       node  (1,2)       ← included
       node  (3,4)       ← included

   All three sub-root nodes become candidate masks, so parent and
   children compete equally during optimal mixing.  Single-gene masks
   are rejected (clusters only).

2. Novel pair-cluster mutation selects a random 2-TLS cluster from the
   FULL Ward tree (no threshold filter), randomly assigns "first" and
   "second" roles, keeps the first TLS intact, then grows the second TLS's
   GREEN time so it strictly exceeds the first's green — enforcing the
   assumption that the second TLS has more available green in gene-space.
   Red phases are pinned at their minimum and yellows stay frozen, so the
   green budget is capped at CYCLE_LENGTH − Σyellows − Σred_mins.
   normalize_to_cycle restores the exact 90 s cycle; because pinned red is
   the smallest phase, the remainder lands in red and the green is kept.

3. Tree-walk mutation traverses the Ward linkage tree starting from a
   randomly selected TLS gene:
   - If the gene is in a pair cluster (size 2), apply paired mutation.
   - If the gene is standalone (leaf, not inside any cluster), mutate it.
   - If the gene is in a cluster of size > 2, enter the cluster and
     recursively mutate individual genes while skipping pair sub-clusters.
   Example: cluster (2, [5, [3,4]]) → mutate 2, enter sub-cluster,
   mutate 5, enter [3,4] which is a pair → skip both.
   But if gene 3 was selected, its smallest cluster is [3,4] (pair) →
   apply paired mutation on (3,4).

9 experiments: 3 linkage trees × 3 population strategies.

Usage:  python -m src.pygad.custom_optimizer
"""
from config import (
    CLUSTER_THRESHOLD_FASTEST,
    CLUSTER_THRESHOLD_SHORTEST,
    CLUSTER_THRESHOLD_EUCLIDIAN,
    MAX_EVALS, BASELINE_TRAFFIC_DATA, NUM_PROCESSORS,
    POPULATION_SIZE, NUM_GENERATIONS,
    GAUSSIAN_NOISE, NOVEL_MUTATION,
    GREEN_FLOOR, MUTATION_RATE,
)
import json, copy, time, os, sys
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.sumo_setup.fitness_evaluation import (
    fitness_function as _traffic_fitness,
    build_traffic_fitness_wrapper,
    normalize_to_cycle,
    phase_type,
)
from src.novel.linkage_tree import build_all_tree_masks
from src.novel.optimal_mixing import mask_to_gene_indices, mix
from src.novel.pairwise_mutation import mutate_tree_walk, build_phase_split

THRESHOLDS = {
    "shortest":  CLUSTER_THRESHOLD_SHORTEST,
    "euclidian": CLUSTER_THRESHOLD_EUCLIDIAN,
    "fastest":   CLUSTER_THRESHOLD_FASTEST,
}  # fraction of individuals mutated per generation


# ── Gene mapping ─────────────────────────────────────────────────────────────

def build_gene_map(baseline_data: dict):
    """
    Returns
    -------
    tls_to_genes : {tls_id: (start, end)} in the flat gene vector
    num_genes    : total gene count
    baseline_vec : np.ndarray of baseline phase durations
    """
    tls_to_genes: dict[str, tuple[int, int]] = {}
    idx      = 0
    baseline: list[float] = []

    for tls_id in sorted(baseline_data["tls_data"]):
        phases = sorted(baseline_data["tls_data"][tls_id])
        tls_to_genes[tls_id] = (idx, idx + len(phases))
        for pk in phases:
            baseline.append(float(baseline_data["tls_data"][tls_id][pk]["duration"]))
        idx += len(phases)

    return tls_to_genes, idx, np.array(baseline)


# ── Population init ──────────────────────────────────────────────────────────

def init_population(
    strategy: str,
    n: int,
    num_genes: int,
    baseline_vec: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
    ub: np.ndarray,
) -> np.ndarray:
    """Create initial population via 'random', 'baseline', or 'mixed' strategy.

    ``ub`` is the per-gene upper bound (dynamic per-TLS green/red ceiling).
    """
    # Per-gene lower bound: yellow phases have ub=6 < GREEN_FLOOR, so cap the
    # lower at ub to keep uniform()/clip() valid (yellow collapses to 6).
    lo = np.minimum(GREEN_FLOOR, ub)
    if strategy == "random":
        return rng.uniform(lo, ub, (n, num_genes))

    elif strategy == "baseline":
        pop  = np.tile(baseline_vec, (n, 1))
        pop += rng.normal(0, noise_std, pop.shape) * pop
        return np.clip(pop, lo, ub)

    elif strategy == "mixed":
        half = n // 2
        rand = rng.uniform(lo, ub, (half, num_genes))
        base = np.tile(baseline_vec, (n - half, 1))
        base += rng.normal(0, noise_std, base.shape) * base
        return np.vstack([rand, np.clip(base, lo, ub)])

    raise ValueError(f"Unknown strategy: {strategy}")


# ── Fitness helpers (parallel) ───────────────────────────────────────────────

def _eval(args):
    wrapper, sol, i = args
    return i, float(wrapper(sol))


def eval_pop(wrapper, pop: np.ndarray, n_workers: int) -> np.ndarray:
    """Evaluate entire population in parallel. Returns fitness array."""
    fit = np.full(len(pop), np.inf)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_eval, (wrapper, pop[i], i)): i for i in range(len(pop))}
        for f in as_completed(futs):
            i, v = f.result()
            fit[i] = v
    return fit


# ── Core GOMEA loop ──────────────────────────────────────────────────────────

def run_custom_optimizer(
    tree_name: str,
    dist_path: str,
    strategy: str,
    baseline_data: dict,
    wrapper,
    num_genes: int,
    baseline_vec: np.ndarray,
    tls_to_genes: dict[str, tuple[int, int]],
    ub: np.ndarray,
    pop_size: int,
    num_gen: int,
    noise_std: float,
    n_workers: int,
    seed: int = 42,
) -> dict:
    """Run one custom optimization experiment. Returns a results dict."""
    rng       = np.random.default_rng(seed)
    threshold = THRESHOLDS[tree_name]

    print(f"\n{'='*60}")
    print(f"Tree: {tree_name} (t={threshold}) | Strategy: {strategy} | Pop: {pop_size}")
    print(f"{'='*60}")

    # 1. Build masks ─────────────────────────────────────────────────────────
    #    mixing_masks : all sub-threshold Ward clusters (parent + children),
    #                   root excluded, singletons excluded.
    #    pair_clusters: all 2-TLS nodes in the full tree (for mutation).
    tls_masks, pair_clusters, tree_structure = build_all_tree_masks(
        dist_path, threshold
    )

    # Convert TLS-ID masks → gene-index masks; require ≥ 2 gene indices
    gene_masks = [mask_to_gene_indices(m, tls_to_genes) for m in tls_masks]
    gene_masks = [m for m in gene_masks if len(m) >= 2]

    # Filter pair_clusters to those where both TLS exist in the gene map
    valid_pairs = [(a, b) for a, b in pair_clusters
                   if a in tls_to_genes and b in tls_to_genes]

    # Per-TLS phase-type index split (green grown, red pinned, yellow frozen).
    phase_split = build_phase_split(baseline_data, tls_to_genes)

    cluster_sizes = sorted(set(len(m) for m in gene_masks))
    print(f"Mixing masks : {len(gene_masks)} clusters "
          f"(gene-group sizes: {cluster_sizes})")
    print(f"Pair-mutation: {len(valid_pairs)} 2-TLS pairs available")
    n_walk_clusters = sum(
        1 for k, v in tree_structure.items()
        if not isinstance(k, str) and v["size"] > 2
    )
    print(f"Tree-walk   : {n_walk_clusters} clusters with size > 2")

    if not gene_masks:
        raise RuntimeError(
            "No valid mixing masks found — check threshold / distance file."
        )

    # 2. Init & evaluate population ─────────────────────────────────────────
    pop = init_population(strategy, pop_size, num_genes, baseline_vec, noise_std, rng, ub)
    fit = eval_pop(wrapper, pop, n_workers)

    best_i              = int(np.argmin(fit))
    best_sol, best_fit  = pop[best_i].copy(), float(fit[best_i])
    print(f"Gen 0 | Best: {best_fit:.2f} | Mean: {np.mean(fit):.2f}")

    history = [{
        "gen":       0,
        "best":      float(best_fit),
        "gen_best":  float(np.min(fit)),
        "gen_worst": float(np.max(fit)),
        "mean":      float(np.mean(fit)),
    }]

    # 3. Generational loop ───────────────────────────────────────────────────
    t0 = time.time() 
    num_evals = pop_size
    gen = 1
    while num_evals < MAX_EVALS: 

        # ── Optimal mixing ───────────────────────────────────────────────────
        # Each individual is paired with a random donor; a random mask from the
        # full sub-threshold cluster set (parents AND children) is applied.
        mix_tasks = []
        for i in range(pop_size):
            donor_i = i
            while donor_i == i:
                donor_i = int(rng.integers(0, pop_size))
            mask = gene_masks[int(rng.integers(0, len(gene_masks)))]
            mix_tasks.append(
                (wrapper, pop[i].copy(), fit[i], pop[donor_i].copy(), mask)
            )

        mix_improved = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(mix, t): idx for idx, t in enumerate(mix_tasks)}
            for f in as_completed(futs):
                idx = futs[f]
                new_sol, new_fit, improved = f.result()
                pop[idx] = new_sol
                fit[idx] = new_fit
                num_evals += 1
                if improved:
                    mix_improved += 1

        # ── Mutation ──────────────────────────────────────────────────────────
        # A random subset (≈ MUTATION_RATE) of individuals are mutated.
        # Each mutant randomly selects a TLS gene; the mutation method is
        # determined by that gene's position in the Ward tree:
        #   - smallest cluster is a pair (size 2) → paired mutation
        #   - smallest cluster has size > 2       → tree-walk mutation
        #   - standalone gene (no cluster)        → direct re-sampling
        # Each mutant is evaluated; the mutation is accepted only if fitness
        # improves, preserving the greedy quality of LT-GOMEA.
        mut_improved = 0
        mutant_idxs = []

        if NOVEL_MUTATION: 
            mutant_idxs = [i for i in range(pop_size) if rng.random() < MUTATION_RATE]

            if mutant_idxs:
                mutants = [
                    mutate_tree_walk(
                        pop[i], tree_structure, tls_to_genes,
                        valid_pairs, phase_split, rng, ub,
                    )
                    for i in mutant_idxs
                ]

                mut_fit_map: dict[int, float] = {}
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futs = {
                        pool.submit(_eval, (wrapper, mutants[j], j)): j
                        for j in range(len(mutant_idxs))
                    }
                    for f in as_completed(futs):
                        j, v = f.result()
                        mut_fit_map[j] = v

                for j, i in enumerate(mutant_idxs):
                    new_fit = mut_fit_map[j]
                    num_evals += 1
                    if new_fit < fit[i]:          # accept only improvements
                        pop[i] = mutants[j]
                        fit[i] = new_fit
                        mut_improved += 1

        # ── Track global best ────────────────────────────────────────────────
        gi = int(np.argmin(fit))
        if fit[gi] < best_fit:
            best_fit = float(fit[gi])
            best_sol = pop[gi].copy()

        history.append({
            "gen":          gen,
            "best":         float(best_fit),
            "gen_best":     float(np.min(fit)),
            "gen_worst":    float(np.max(fit)),
            "mean":         float(np.mean(fit)),
            "mix_improved": mix_improved,
            "mut_improved": mut_improved,
            "mutants":      len(mutant_idxs),
        })
        print(
            f"Gen {gen:2d} | Best: {best_fit:.2f} | Mean: {np.mean(fit):.2f} "
            f"| Mix+{mix_improved} | Mut+{mut_improved}/{len(mutant_idxs)}"
        )
        gen += 1

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s | Final best: {best_fit:.2f}")

    best_json = _rebuild_json(best_sol, baseline_data, tls_to_genes, ub)
    best_json["composite_cost"] = float(best_fit)

    return {
        "best_configuration":  best_json,
        "best_fitness":        float(best_fit),
        "fitness_history":     history,
        "time_s":              round(elapsed, 2),
        "tree":                tree_name,
        "threshold":           threshold,
        "strategy":            strategy,
        "pop_size":            pop_size,
        "generations":         num_gen,
        "num_mixing_masks":    len(gene_masks),
        "num_pair_clusters":   len(valid_pairs),
        "seed":                seed,
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _rebuild_json(sol: np.ndarray, baseline: dict, tls_to_genes: dict,
                  ub: np.ndarray) -> dict:
    """
    Convert flat gene vector back to the full TLS JSON format.

    Uses the SAME normalisation that the fitness wrapper applies before
    SUMO sees the durations, so the saved JSON matches what was actually
    evaluated.
    """
    out = copy.deepcopy(baseline)
    for tls_id in sorted(out["tls_data"]):
        if tls_id not in tls_to_genes:
            continue
        s, e        = tls_to_genes[tls_id]
        keys        = sorted(out["tls_data"][tls_id])
        phase_types = [phase_type(out["tls_data"][tls_id][pk]["state"]) for pk in keys]
        dur         = normalize_to_cycle(list(sol[s:e]), phase_types,
                                         upper_bounds=ub[s:e])

        for i, pk in enumerate(keys):
            out["tls_data"][tls_id][pk]["duration"] = int(dur[i])
    return out


# ── Experiment runner ────────────────────────────────────────────────────────

def run_all_experiments():
    with open(BASELINE_TRAFFIC_DATA) as f:
        baseline = json.load(f)

    wrapper, num_genes, _, ub, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline, fitness_function=_traffic_fitness
    )
    tls_to_genes, _, baseline_vec = build_gene_map(baseline)
    n_workers = NUM_PROCESSORS or os.cpu_count() or 1

    root    = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "src" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = {
        "shortest":  out_dir / "tls_distances_shortest.json",
        "euclidian": out_dir / "tls_distances_euclidian.json",
        "fastest":   out_dir / "tls_distances_fastest.json",
    }
    strategies = ["random", "baseline", "mixed"]
    summary: dict[str, dict] = {}

    for tree_name, path in trees.items():
        for strat in strategies:
            label = f"{tree_name}_{strat}"
            try:
                res = run_custom_optimizer(
                    tree_name, str(path), strat, baseline,
                    wrapper, num_genes, baseline_vec, tls_to_genes, ub,
                    POPULATION_SIZE, NUM_GENERATIONS,
                    GAUSSIAN_NOISE, n_workers,
                )
                mutation_suffix = "_mutation" if NOVEL_MUTATION else ""
                out_file = out_dir / f"custom_optimizer_{label}{mutation_suffix}.json"
                with open(out_file, "w") as f:
                    json.dump(res, f, indent=4)
                print(f"Saved → {out_file}")
                summary[label] = {"best": res["best_fitness"], "time_s": res["time_s"]}

            except Exception as e:
                print(f"ERROR [{label}]: {e}")
                import traceback; traceback.print_exc()
                summary[label] = {"error": str(e)}

    # ── Results table ────────────────────────────────────────────────────────
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