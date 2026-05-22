"""
GOMEA Optimizer using the official GOMEA Python library for traffic light optimization.

Optimizes the baseline traffic light configuration using the real-valued GOMEA library.
Runs 3 experiments using custom linkage tree FOS files loaded from precalculated distance JSONs.

Usage:  python3 -m src.algorithms.gomea_optimizer
"""

import os
import sys
import json
import time
import copy
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import gomea

# Add project root to sys.path to import config and other modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (
    CLUSTER_THRESHOLD_FASTEST,
    CLUSTER_THRESHOLD_SHORTEST,
    CLUSTER_THRESHOLD_EUCLIDIAN,
    MAX_EVALS,
    BASELINE_TRAFFIC_DATA,
    GENE_LOW,
    GENE_HIGH,
    POPULATION_SIZE,
)

from src.sumo_setup.fitness_evaluation import (
    fitness_function as _traffic_fitness,
    build_traffic_fitness_wrapper,
)

# Threshold mapping
THRESHOLDS = {
    "shortest":  CLUSTER_THRESHOLD_SHORTEST,
    "euclidian": CLUSTER_THRESHOLD_EUCLIDIAN,
    "fastest":   CLUSTER_THRESHOLD_FASTEST,
}

# ── Custom GOMEA Fitness Function ──────────────────────────────────────────

class TLSFitness(gomea.fitness.BBOFitnessFunctionRealValued):
    """Custom BBO fitness function wrapping the SUMO simulator wrapper.
    
    Tracks the best solution and fitness history inside Python to ensure we can
    recover results even if GOMEA terminates with a RuntimeError (e.g. evaluations limit).
    """
    def __init__(self, number_of_variables, wrapper):
        super().__init__(number_of_variables)
        self.wrapper = wrapper
        self.best_fit = float("inf")
        self.best_sol = None
        self.history = []
        self.eval_count = 0

    def objective_function(self, objective_index, variables):
        self.eval_count += 1
        
        # Round variables to integers (whole numbers) to match differential_evolution
        rounded_vars = np.round(variables).astype(int)
        
        # Evaluate fitness via the SUMO wrapper
        cost = float(self.wrapper(rounded_vars))
        
        # Track best solution
        if cost < self.best_fit:
            self.best_fit = cost
            self.best_sol = np.array(variables)  # store continuous representation
            
            # Record history (gen estimated using pop_size = 100)
            gen = self.eval_count // 100
            self.history.append({
                "gen": gen,
                "best": cost,
                "mean": cost
            })
            
        return cost


# ── Distance Matrix loading ──────────────────────────────────────────────────

def _load_distance_array(distance_json: str):
    """Return (symmetric_np_array, ordered_tls_id_list) from distance JSON."""
    with open(distance_json) as f:
        data = json.load(f)

    key = "distance_matrix" if "distance_matrix" in data else "travel_time_matrix"
    matrix = data[key]
    tls_ids = [t["id"] for t in data["traffic_lights"]]
    n = len(tls_ids)

    vals = [v for row in matrix.values() for v in row.values() if v is not None]
    penalty = max(vals) * 1.5 if vals else 1e6

    arr = np.zeros((n, n))
    for i, a in enumerate(tls_ids):
        for j, b in enumerate(tls_ids):
            v = matrix[a].get(b)
            arr[i, j] = v if v is not None else penalty
            
    arr = (arr + arr.T) / 2
    np.fill_diagonal(arr, 0)
    return arr, tls_ids


# ── FOS File Generation ──────────────────────────────────────────────────────

def build_fos_file(distance_json: str, threshold: float, tls_to_genes: dict, num_genes: int, output_fos_path: str):
    """Build a Family of Subsets (FOS) file from Ward linkage of the distance matrix.
    
    Includes univariate singletons for all genes, and subsets representing clusters of TLSs.
    """
    # 1. Load distance matrix
    arr, tls_ids = _load_distance_array(distance_json)
    n = len(tls_ids)
    
    # 2. Compute Ward linkage
    condensed = squareform(arr)
    Z = linkage(condensed, method="ward")
    
    # 3. Build leaf membership for all nodes in the tree
    members = {i: [i] for i in range(n)}
    for i, row in enumerate(Z):
        left, right = int(row[0]), int(row[1])
        members[n + i] = members[left] + members[right]
        
    root_id = n + len(Z) - 1
    
    # 4. Extract sub-threshold internal nodes
    tls_masks = []
    for i, row in enumerate(Z):
        node_id = n + i
        merge_dist = float(row[2])
        tls_group = [tls_ids[m] for m in members[node_id]]
        
        if node_id == root_id:
            continue  # Exclude root cluster containing all TLSs
        if merge_dist > threshold:
            continue  # Exceeds clustering threshold
            
        tls_masks.append(tls_group)
        
    # 5. Map TLS clusters to flat gene indices
    gene_subsets = []
    for mask in tls_masks:
        subset = []
        for tls_id in mask:
            if tls_id in tls_to_genes:
                start, end = tls_to_genes[tls_id]
                subset.extend(range(start, end))
        if len(subset) >= 2:
            gene_subsets.append(subset)

    # 6. Write FOS to file (space-separated indices, one group per line)
    Path(output_fos_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_fos_path, "w") as f:
        # First write univariate singletons
        for i in range(num_genes):
            f.write(f"{i}\n")
        # Then write multi-variable clusters
        for subset in gene_subsets:
            subset_str = " ".join(str(idx) for idx in subset)
            f.write(f"{subset_str}\n")
            
    print(f"FOS file generated at: {output_fos_path} ({num_genes} singletons, {len(gene_subsets)} clusters)")
    return len(gene_subsets)


# ── JSON Reconstruction ──────────────────────────────────────────────────────

def _rebuild_json(sol: np.ndarray, baseline: dict, tls_to_genes: dict) -> dict:
    """Convert flat solution vector back to full SUMO TLS JSON configuration (90 s total per TLS)."""
    out = copy.deepcopy(baseline)
    for tls_id in sorted(out["tls_data"]):
        if tls_id not in tls_to_genes:
            continue
        s, e = tls_to_genes[tls_id]
        raw = sol[s:e]
        keys = sorted(out["tls_data"][tls_id])
        n = len(keys)
        total = float(sum(raw))

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


# ── Single Experiment Runner ─────────────────────────────────────────────────

def run_single_experiment(
    tree_name: str,
    dist_path: str,
    baseline_data: dict,
    wrapper,
    num_genes: int,
    tls_to_genes: dict,
    pop_size: int,
    max_evals: int,
    seed: int = 42,
) -> dict:
    """Run GOMEA optimization for a single linkage tree."""
    print(f"\n{'='*60}")
    print(f"GOMEA | Tree: {tree_name} | Pop Size: {pop_size} | Max Evals: {max_evals}")
    print(f"{'='*60}")

    # 1. Generate FOS file
    threshold = THRESHOLDS[tree_name]
    fos_file_path = f"src/outputs/gomea_fos_{tree_name}.txt"
    num_clusters = build_fos_file(dist_path, threshold, tls_to_genes, num_genes, fos_file_path)

    # 2. Setup linkage model from file
    linkage_model = gomea.linkage.Custom(file=fos_file_path)

    # 3. Setup custom fitness function
    fitness_inst = TLSFitness(num_genes, wrapper)

    # 4. Instantiate and configure GOMEA
    rvgom = gomea.RealValuedGOMEA(
        fitness=fitness_inst,
        linkage_model=linkage_model,
        lower_init_range=GENE_LOW,
        upper_init_range=GENE_HIGH,
        random_seed=seed,
        base_population_size=pop_size,
        max_number_of_populations=1,  # disables IMS multi-start, forces pop size = 100
        max_number_of_evaluations=max_evals,
        verbose=False,
    )

    # 5. Run optimization
    t0 = time.time()
    try:
        res = rvgom.run()
        elapsed = time.time() - t0
        best_sol_str = res.final_metrics_dict["best_solution"]
        solution = np.array(json.loads(best_sol_str))
        best_cost = float(res.final_metrics_dict["best_obj_val"])
        evals_done = int(res.final_metrics_dict["evaluations"])
        
        # Extract history from run object
        fitness_history = []
        if "generation" in res.generational_metrics_dict:
            for i in range(len(res.generational_metrics_dict["generation"])):
                fitness_history.append({
                    "gen": int(res.generational_metrics_dict["generation"][i]),
                    "best": float(res.generational_metrics_dict["best_obj_val"][i]),
                    "mean": float(res.generational_metrics_dict["best_obj_val"][i]),
                })
        else:
            fitness_history = fitness_inst.history
            
    except RuntimeError as e:
        elapsed = time.time() - t0
        if "evaluations" in str(e) or "generations" in str(e) or "time" in str(e):
            print(f"GOMEA terminated via expected limit exception: {e}")
            solution = fitness_inst.best_sol
            best_cost = fitness_inst.best_fit
            evals_done = fitness_inst.eval_count
            fitness_history = fitness_inst.history
        else:
            raise e

    print(f"Completed in {elapsed:.1f}s | Best Fitness: {best_cost:.2f} | Evals: {evals_done}")

    # Reconstruct final best JSON configuration
    best_json = _rebuild_json(solution, baseline_data, tls_to_genes)
    best_json["composite_cost"] = float(best_cost)

    return {
        "best_configuration": best_json,
        "best_fitness":       float(best_cost),
        "fitness_history":    fitness_history,
        "time_s":             round(elapsed, 2),
        "algorithm":          "gomea_library",
        "tree":               tree_name,
        "threshold":          threshold,
        "strategy":           "random",
        "pop_size":           pop_size,
        "generations":        evals_done // pop_size,
        "num_masks":          num_clusters,
        "seed":               seed,
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ── Experiments Coordinator ──────────────────────────────────────────────────

def run_all_experiments():
    # Load baseline traffic configurations
    with open(BASELINE_TRAFFIC_DATA) as f:
        baseline = json.load(f)

    # Initialize SUMO fitness evaluation wrapper
    wrapper, num_genes, _, _, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline,
        fitness_function=_traffic_fitness,
    )

    out_dir = Path("src/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define paths to distance files
    trees = {
        "shortest":  out_dir / "tls_distances_shortest.json",
        "euclidian": out_dir / "tls_distances_euclidian.json",
        "fastest":   out_dir / "tls_distances_fastest.json",
    }

    # Gather gene mapping metadata
    tls_to_genes = {}
    idx = 0
    for tls_id in sorted(baseline["tls_data"]):
        phases = sorted(baseline["tls_data"][tls_id])
        tls_to_genes[tls_id] = (idx, idx + len(phases))
        idx += len(phases)

    summary = {}

    for tree_name, dist_path in trees.items():
        try:
            res = run_single_experiment(
                tree_name=tree_name,
                dist_path=str(dist_path),
                baseline_data=baseline,
                wrapper=wrapper,
                num_genes=num_genes,
                tls_to_genes=tls_to_genes,
                pop_size=POPULATION_SIZE,
                max_evals=MAX_EVALS,
            )

            out_file = out_dir / f"gomea_{tree_name}_random.json"
            with open(out_file, "w") as f:
                json.dump(res, f, indent=4)
            print(f"Saved results to: {out_file}")
            
            summary[tree_name] = {"best": res["best_fitness"], "time_s": res["time_s"]}

        except Exception as e:
            print(f"ERROR running experiment [{tree_name}]: {e}")
            import traceback
            traceback.print_exc()
            summary[tree_name] = {"error": str(e)}

    # Print summary comparison table
    print(f"\n{'Linkage Tree':<15} {'Best Fitness (Min)':>20} {'Runtime (s)':>12}")
    print("─" * 51)
    for tree_name, info in summary.items():
        if "error" in info:
            print(f"{tree_name:<15} {'ERROR':>20} {'-':>12}")
        else:
            print(f"{tree_name:<15} {info['best']:>20.2f} {info['time_s']:>11.1f}s")


if __name__ == "__main__":
    run_all_experiments()
