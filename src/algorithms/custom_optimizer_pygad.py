"""
Custom Optimizer (GOMEA library) — continuous GOMEA with custom linkage trees.

Uses the real-valued GOMEA library and feeds it custom Family-of-Subsets (FOS)
files built from Ward linkage trees over three pre-computed distance / travel-
time matrices (shortest, euclidian, fastest). Continuous gene values are
rounded to whole numbers before being passed to the SUMO simulator, matching
the rounding parity used by differential_evolution.py.

FOS file format (one subset per line, space-separated gene indices):
    Lines 0 .. num_genes-1 : every gene as its own univariate subset.
    Subsequent lines       : multi-gene clusters from the sub-threshold tree
                             (size ≥ 2, root excluded).

3 experiments: 3 linkage trees × random initialisation.

Usage:  python -m src.algorithms.custom_optimizer_pygad
"""
import os, sys, json, copy, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

import gomea

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (
    CLUSTER_THRESHOLD_FASTEST,
    CLUSTER_THRESHOLD_SHORTEST,
    CLUSTER_THRESHOLD_EUCLIDIAN,
    MAX_EVALS, BASELINE_TRAFFIC_DATA,
    POPULATION_SIZE,
    GENE_LOW, GENE_HIGH,
    NUM_PROCESSORS,
    NUM_SEEDS_PER_TREE,
    SEED_BASE,
)
from src.sumo_setup.fitness_evaluation import (
    fitness_function as _traffic_fitness,
    build_traffic_fitness_wrapper,
)

THRESHOLDS = {
    "shortest":  CLUSTER_THRESHOLD_SHORTEST,
    "euclidian": CLUSTER_THRESHOLD_EUCLIDIAN,
    "fastest":   CLUSTER_THRESHOLD_FASTEST,
}


def _available_cores() -> int:
    """Cores this process is actually allowed to use.

    Prefers os.sched_getaffinity (respects taskset / cgroups / SLURM
    pinning on Linux); falls back to os.cpu_count() on platforms without
    it (e.g. macOS). Always returns at least 1.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


class StopOptimization(Exception):
    """Raised by the fitness function to hard-stop GOMEA at MAX_EVALS."""
    pass


# ── Custom GOMEA fitness function ────────────────────────────────────────────

class TLSFitness(gomea.fitness.BBOFitnessFunctionRealValued):
    """BBO fitness wrapping the SUMO simulator with integer rounding parity."""

    def __init__(self, number_of_variables, max_evals):
        super().__init__(number_of_variables)
        self.max_evals = max_evals
        self.wrapper = None
        self.best_fit = float("inf")
        self.best_sol = None
        self.history = []
        self.eval_count = 0

    def objective_function(self, objective_index, variables):
        if self.eval_count >= self.max_evals:
            raise StopOptimization(f"Hit MAX_EVALS={self.max_evals}")

        self.eval_count += 1

        # Round continuous variables to whole numbers for the simulator,
        # matching the rounding parity used by differential_evolution.py.
        rounded = np.round(variables).astype(int)
        cost = float(self.wrapper(rounded))

        if cost < self.best_fit:
            self.best_fit = cost
            self.best_sol = np.array(variables)
            self.history.append({
                "gen":  self.eval_count // POPULATION_SIZE,
                "best": cost,
                "mean": cost,
            })

        return cost


# ── Distance matrix helpers ──────────────────────────────────────────────────

def _load_distance_array(distance_json: str):
    """Return (symmetric_np_array, ordered_tls_id_list)."""
    with open(distance_json) as f:
        data = json.load(f)

    key    = "distance_matrix" if "distance_matrix" in data else "travel_time_matrix"
    matrix = data[key]
    tls_ids = [t["id"] for t in data["traffic_lights"]]
    n       = len(tls_ids)

    vals    = [v for row in matrix.values() for v in row.values() if v is not None]
    penalty = max(vals) * 1.5 if vals else 1e6

    arr = np.zeros((n, n))
    for i, a in enumerate(tls_ids):
        for j, b in enumerate(tls_ids):
            v = matrix[a].get(b)
            arr[i, j] = v if v is not None else penalty
    arr = (arr + arr.T) / 2
    np.fill_diagonal(arr, 0)
    return arr, tls_ids


# ── Gene mapping ─────────────────────────────────────────────────────────────

def build_gene_map(baseline_data: dict):
    """Returns (tls_to_genes, num_genes)."""
    tls_to_genes: dict[str, tuple[int, int]] = {}
    idx = 0
    for tls_id in sorted(baseline_data["tls_data"]):
        phases = sorted(baseline_data["tls_data"][tls_id])
        tls_to_genes[tls_id] = (idx, idx + len(phases))
        idx += len(phases)
    return tls_to_genes, idx


# ── FOS file builder ─────────────────────────────────────────────────────────

def build_fos_file(
    distance_json: str,
    threshold: float,
    tls_to_genes: dict,
    num_genes: int,
    out_path: str,
) -> int:
    """
    Build a FOS file from Ward linkage of the distance matrix.

    Univariate genes are written first (one per line), followed by the
    multi-gene clusters from every internal Ward node whose merge distance
    is ≤ ``threshold`` (root excluded, singletons skipped).

    Returns the number of multi-gene clusters written.
    """
    arr, tls_ids = _load_distance_array(distance_json)
    n = len(tls_ids)
    Z = linkage(squareform(arr), method="ward")

    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    for i, row in enumerate(Z):
        left, right    = int(row[0]), int(row[1])
        members[n + i] = members[left] + members[right]

    root_id   = n + len(Z) - 1
    tls_masks = []
    for i, row in enumerate(Z):
        node_id = n + i
        if node_id == root_id:
            continue
        if float(row[2]) > threshold:
            continue
        tls_masks.append([tls_ids[m] for m in members[node_id]])

    gene_subsets: list[list[int]] = []
    for mask in tls_masks:
        subset: list[int] = []
        for tls_id in mask:
            if tls_id in tls_to_genes:
                s, e = tls_to_genes[tls_id]
                subset.extend(range(s, e))
        if len(subset) >= 2:
            gene_subsets.append(subset)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i in range(num_genes):
            f.write(f"{i}\n")
        for subset in gene_subsets:
            f.write(" ".join(str(idx) for idx in subset) + "\n")

    return len(gene_subsets)


# ── JSON reconstruction ─────────────────────────────────────────────────────

def _rebuild_json(sol: np.ndarray, baseline: dict, tls_to_genes: dict) -> dict:
    """Convert flat gene vector back to the TLS JSON format (90 s per TLS)."""
    # Round to integers first so the saved JSON matches what the simulator scored.
    sol_rounded = np.round(sol).astype(int)

    out = copy.deepcopy(baseline)
    for tls_id in sorted(out["tls_data"]):
        if tls_id not in tls_to_genes:
            continue
        s, e  = tls_to_genes[tls_id]
        raw   = sol_rounded[s:e]
        keys  = sorted(out["tls_data"][tls_id])
        n     = len(keys)
        total = float(sum(raw))

        if total <= 0:
            dur       = [90 // n] * n
            dur[-1]  += 90 - sum(dur)
        else:
            dur  = [max(1, int(round(d * 90 / total))) for d in raw]
            diff = 90 - sum(dur)
            if diff:
                dur[int(np.argmax(dur))] += diff

        for i, pk in enumerate(keys):
            out["tls_data"][tls_id][pk]["duration"] = int(dur[i])
    return out


# ── Single experiment runner ─────────────────────────────────────────────────

def run_single_experiment(
    tree_name: str,
    baseline_data: dict,
    num_genes: int,
    tls_to_genes: dict,
    pop_size: int,
    max_evals: int,
    fos_path: str,
    num_clusters: int,
    seed: int,
) -> dict:
    """Run one continuous-GOMEA experiment with a pre-built FOS file."""
    threshold = THRESHOLDS[tree_name]
    tag = f"{tree_name} seed={seed}"
    print(f"[{tag}] start | t={threshold} | pop={pop_size} | clusters={num_clusters}")

    wrapper, _, _, _, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline_data,
        fitness_function=_traffic_fitness,
    )

    fitness_inst         = TLSFitness(num_genes, max_evals)
    fitness_inst.wrapper = wrapper

    linkage_model = gomea.linkage.Custom(file=fos_path)

    rvgom = gomea.RealValuedGOMEA(
        fitness                       = fitness_inst,
        linkage_model                 = linkage_model,
        lower_init_range              = GENE_LOW,
        upper_init_range              = GENE_HIGH,
        random_seed                   = seed,
        base_population_size          = pop_size,
        max_number_of_populations     = 1,
        max_number_of_evaluations     = max_evals,
        verbose                       = False,
    )

    t0 = time.time()
    try:
        res            = rvgom.run()
        best_sol_str   = res.final_metrics_dict["best_solution"]
        solution       = np.array(json.loads(best_sol_str))
        best_cost      = float(res.final_metrics_dict["best_obj_val"])
        evals_done     = int(res.final_metrics_dict["evaluations"])
        fitness_history = fitness_inst.history

    except StopOptimization as e:
        print(f"[{tree_name}] Stopped at MAX_EVALS: {e}")
        solution        = fitness_inst.best_sol
        best_cost       = fitness_inst.best_fit
        evals_done      = fitness_inst.eval_count
        fitness_history = fitness_inst.history

    except RuntimeError as e:
        if "evaluations" in str(e) or "generations" in str(e):
            print(f"[{tree_name}] GOMEA native limit reached: {e}")
            solution        = fitness_inst.best_sol
            best_cost       = fitness_inst.best_fit
            evals_done      = fitness_inst.eval_count
            fitness_history = fitness_inst.history
        else:
            raise

    elapsed = time.time() - t0
    print(f"[{tag}] done in {elapsed:.1f}s | best={best_cost:.2f} | evals={evals_done}")

    best_json                   = _rebuild_json(solution, baseline_data, tls_to_genes)
    best_json["composite_cost"] = float(best_cost)

    return {
        "best_configuration":  best_json,
        "best_fitness":        float(best_cost),
        "fitness_history":     fitness_history,
        "time_s":              round(elapsed, 2),
        "algorithm":           "gomea_library",
        "tree":                tree_name,
        "threshold":           threshold,
        "strategy":            "random",
        "pop_size":            pop_size,
        "generations":         evals_done // pop_size if pop_size else 0,
        "num_mixing_masks":    num_clusters,
        "num_evals":           evals_done,
        "seed":                seed,
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ── Experiment runner ────────────────────────────────────────────────────────

def run_all_experiments():
    with open(BASELINE_TRAFFIC_DATA) as f:
        baseline = json.load(f)

    tls_to_genes, num_genes = build_gene_map(baseline)

    root    = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "src" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = {
        "shortest":  out_dir / "tls_distances_shortest.json",
        "euclidian": out_dir / "tls_distances_euclidian.json",
        "fastest":   out_dir / "tls_distances_fastest.json",
    }

    # Build each FOS file once in the parent. All seeds for a given tree
    # share the same FOS file, so building per-worker would cause races
    # on the same output path.
    fos_paths: dict[str, str] = {}
    fos_clusters: dict[str, int] = {}
    for tree_name, dist_path in trees.items():
        fos_path = out_dir / f"custom_optimizer_pygad_fos_{tree_name}.txt"
        n_clusters = build_fos_file(
            str(dist_path), THRESHOLDS[tree_name],
            tls_to_genes, num_genes, str(fos_path),
        )
        fos_paths[tree_name]    = str(fos_path)
        fos_clusters[tree_name] = n_clusters
        print(f"FOS [{tree_name}] → {fos_path} ({n_clusters} clusters + {num_genes} univariate)")

    seeds     = [SEED_BASE + i for i in range(NUM_SEEDS_PER_TREE)]
    jobs      = [(t, s) for t in trees for s in seeds]
    detected  = _available_cores()
    requested = NUM_PROCESSORS or detected
    workers   = max(1, min(requested, len(jobs)))
    cap_note  = f" (capped from NUM_PROCESSORS={NUM_PROCESSORS})" if NUM_PROCESSORS else ""
    print(
        f"\nLaunching {len(jobs)} runs ({len(trees)} trees × {NUM_SEEDS_PER_TREE} seeds) | "
        f"detected {detected} cores → using {workers} workers{cap_note}\n"
    )

    per_tree_runs: dict[str, list[dict]] = {t: [] for t in trees}
    per_tree_errors: dict[str, list[dict]] = {t: [] for t in trees}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_single_experiment,
                tree_name, baseline,
                num_genes, tls_to_genes,
                POPULATION_SIZE, MAX_EVALS,
                fos_paths[tree_name], fos_clusters[tree_name],
                seed,
            ): (tree_name, seed)
            for tree_name, seed in jobs
        }

        for future in as_completed(futures):
            tree_name, seed = futures[future]
            try:
                per_tree_runs[tree_name].append(future.result())
            except Exception as e:
                print(f"ERROR [{tree_name} seed={seed}]: {e}")
                import traceback; traceback.print_exc()
                per_tree_errors[tree_name].append({"seed": seed, "error": str(e)})

    # ── Aggregate best-of-N per tree, save canonical JSON ───────────────────
    summary: dict[str, dict] = {}
    for tree_name, runs in per_tree_runs.items():
        if not runs:
            summary[tree_name] = {"error": "all seeds failed"}
            continue

        best_run = min(runs, key=lambda r: r["best_fitness"])
        fits     = [r["best_fitness"] for r in runs]
        times    = [r["time_s"]        for r in runs]

        best_run["num_seeds_run"]      = len(runs)
        best_run["num_seeds_failed"]   = len(per_tree_errors[tree_name])
        best_run["best_seed"]          = best_run["seed"]
        best_run["mean_best_fitness"]  = float(np.mean(fits))
        best_run["std_best_fitness"]   = float(np.std(fits))
        best_run["min_best_fitness"]   = float(np.min(fits))
        best_run["max_best_fitness"]   = float(np.max(fits))
        best_run["per_seed_stats"]     = [
            {"seed": r["seed"], "best_fitness": r["best_fitness"],
             "time_s": r["time_s"], "num_evals": r["num_evals"]}
            for r in sorted(runs, key=lambda r: r["seed"])
        ]
        if per_tree_errors[tree_name]:
            best_run["seed_errors"] = per_tree_errors[tree_name]

        out_file = out_dir / f"custom_optimizer_pygad_{tree_name}.json"
        with open(out_file, "w") as f:
            json.dump(best_run, f, indent=4)
        print(f"Saved → {out_file}  (best-of-{len(runs)} | best_seed={best_run['seed']})")

        summary[tree_name] = {
            "best":   best_run["best_fitness"],
            "mean":   best_run["mean_best_fitness"],
            "std":    best_run["std_best_fitness"],
            "time_s": best_run["time_s"],
            "n":      len(runs),
            "max_time": float(np.max(times)),
        }

    # ── Results table ────────────────────────────────────────────────────────
    print(f"\n{'Tree':<12} {'Best':>10} {'Mean':>10} {'Std':>8} {'Seeds':>6} {'BestTime':>10}")
    print("─" * 60)
    for tree_name, info in summary.items():
        if "error" in info:
            print(f"{tree_name:<12} {'ERROR':>10}")
        else:
            print(f"{tree_name:<12} {info['best']:>10.2f} {info['mean']:>10.2f} "
                  f"{info['std']:>8.2f} {info['n']:>6} {info['time_s']:>9.1f}s")


if __name__ == "__main__":
    run_all_experiments()