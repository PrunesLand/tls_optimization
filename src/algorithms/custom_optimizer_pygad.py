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
    dist_path: str,
    baseline_data: dict,
    num_genes: int,
    tls_to_genes: dict,
    pop_size: int,
    max_evals: int,
    out_dir: Path,
    seed: int = 42,
) -> dict:
    """Run one continuous-GOMEA experiment with a custom linkage tree."""
    threshold = THRESHOLDS[tree_name]

    print(f"\n{'='*60}")
    print(f"GOMEA (library) | Tree: {tree_name} (t={threshold}) | Pop: {pop_size}")
    print(f"{'='*60}")

    fos_path     = out_dir / f"custom_optimizer_pygad_fos_{tree_name}.txt"
    num_clusters = build_fos_file(
        dist_path, threshold, tls_to_genes, num_genes, str(fos_path),
    )
    print(f"FOS file → {fos_path} ({num_clusters} multi-gene clusters + {num_genes} univariate)")

    wrapper, _, _, _, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline_data,
        fitness_function=_traffic_fitness,
    )

    fitness_inst         = TLSFitness(num_genes, max_evals)
    fitness_inst.wrapper = wrapper

    linkage_model = gomea.linkage.Custom(file=str(fos_path))

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
    print(f"Done in {elapsed:.1f}s | Final best: {best_cost:.2f} | Evals: {evals_done}")

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
    summary: dict[str, dict] = {}

    for tree_name, dist_path in trees.items():
        try:
            res = run_single_experiment(
                tree_name, str(dist_path), baseline,
                num_genes, tls_to_genes,
                POPULATION_SIZE, MAX_EVALS, out_dir,
            )
            out_file = out_dir / f"custom_optimizer_pygad_{tree_name}.json"
            with open(out_file, "w") as f:
                json.dump(res, f, indent=4)
            print(f"Saved → {out_file}")
            summary[tree_name] = {"best": res["best_fitness"], "time_s": res["time_s"]}

        except Exception as e:
            print(f"ERROR [{tree_name}]: {e}")
            import traceback; traceback.print_exc()
            summary[tree_name] = {"error": str(e)}

    # ── Results table ────────────────────────────────────────────────────────
    print(f"\n{'Tree':<15} {'Best':>12} {'Time':>8}")
    print("─" * 36)
    for tree_name, info in summary.items():
        if "error" in info:
            print(f"{tree_name:<15} {'ERROR':>12}")
        else:
            print(f"{tree_name:<15} {info['best']:>12.2f} {info['time_s']:>7.1f}s")


if __name__ == "__main__":
    run_all_experiments()