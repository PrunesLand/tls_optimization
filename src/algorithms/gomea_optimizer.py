"""
GOMEA Optimizer using the official GOMEA Python library for traffic light optimization.

Optimizes the baseline traffic light configuration using the real-valued GOMEA library.
Runs 3 experiments using custom linkage tree FOS files loaded from precalculated distance JSONs.
Executes experiments in PARALLEL, strictly enforces MAX_EVALS, and guarantees
identical rounding parity with the Differential Evolution (SHADE) optimizer.

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
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# ── Custom Exceptions ────────────────────────────────────────────────────────

class StopOptimization(Exception):
    """Custom exception to hard-stop GOMEA the moment MAX_EVALS is reached."""
    pass

# ── Custom GOMEA Fitness Function ──────────────────────────────────────────

class TLSFitness(gomea.fitness.BBOFitnessFunctionRealValued):
    """Custom BBO fitness function wrapping the SUMO simulator wrapper."""
    def __init__(self, number_of_variables, max_evals, pop_size):
        super().__init__(number_of_variables)
        self.max_evals = max_evals
        self.pop_size = pop_size
        self.wrapper = None
        self.best_fit = float("inf")
        self.best_sol = None
        self.history = []
        self.eval_count = 0
        self._gen_costs = []

    def objective_function(self, objective_index, variables):
        # Strictly enforce MAX_EVALS limit before evaluating
        if self.eval_count >= self.max_evals:
            raise StopOptimization(f"Strict evaluation limit of {self.max_evals} reached.")

        self.eval_count += 1

        # 1. Round continuous variables to integers for simulator evaluation
        rounded_vars = np.round(variables).astype(int)

        # 2. Evaluate fitness via the SUMO wrapper
        cost = float(self.wrapper(rounded_vars))

        # 3. Track best continuous solution and cost
        if cost < self.best_fit:
            self.best_fit = cost
            self.best_sol = np.array(variables)

        # 4. Buffer per-generation stats; emit when buffer fills one pop_size window
        self._gen_costs.append(cost)
        if len(self._gen_costs) >= self.pop_size:
            gen = self.eval_count // self.pop_size
            self.history.append({
                "gen":       gen,
                "best":      float(self.best_fit),
                "gen_best":  float(min(self._gen_costs)),
                "gen_worst": float(max(self._gen_costs)),
                "mean":      float(sum(self._gen_costs) / len(self._gen_costs)),
            })
            self._gen_costs = []

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
    """Build a Family of Subsets (FOS) file from Ward linkage of the distance matrix."""
    arr, tls_ids = _load_distance_array(distance_json)
    n = len(tls_ids)
    
    condensed = squareform(arr)
    Z = linkage(condensed, method="ward")
    
    members = {i: [i] for i in range(n)}
    for i, row in enumerate(Z):
        left, right = int(row[0]), int(row[1])
        members[n + i] = members[left] + members[right]
        
    root_id = n + len(Z) - 1
    
    tls_masks = []
    for i, row in enumerate(Z):
        node_id = n + i
        merge_dist = float(row[2])
        tls_group = [tls_ids[m] for m in members[node_id]]
        
        if node_id == root_id or merge_dist > threshold:
            continue
            
        tls_masks.append(tls_group)
        
    gene_subsets = []
    for mask in tls_masks:
        subset = []
        for tls_id in mask:
            if tls_id in tls_to_genes:
                start, end = tls_to_genes[tls_id]
                subset.extend(range(start, end))
        if len(subset) >= 2:
            gene_subsets.append(subset)

    Path(output_fos_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_fos_path, "w") as f:
        for i in range(num_genes):
            f.write(f"{i}\n")
        for subset in gene_subsets:
            subset_str = " ".join(str(idx) for idx in subset)
            f.write(f"{subset_str}\n")
            
    return len(gene_subsets)


# ── JSON Reconstruction ──────────────────────────────────────────────────────

def _rebuild_json(sol: np.ndarray, baseline: dict, tls_to_genes: dict) -> dict:
    """Convert flat solution vector back to full SUMO TLS JSON configuration."""
    
    # NEW: First convert the continuous best solution to integers to match the exact 
    # numbers that the SUMO simulator scored (parity with DE rounding logic).
    sol_rounded = np.round(sol).astype(int)

    out = copy.deepcopy(baseline)
    for tls_id in sorted(out["tls_data"]):
        if tls_id not in tls_to_genes:
            continue
        s, e = tls_to_genes[tls_id]
        
        # Use the integer-rounded values for distribution
        raw = sol_rounded[s:e]
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
    num_genes: int,
    tls_to_genes: dict,
    pop_size: int,
    max_evals: int,
    seed: int = 42,
) -> dict:
    """Run GOMEA optimization for a single linkage tree in an isolated process."""
    print(f"\n[Process: {tree_name}] Starting... | Pop Size: {pop_size} | Max Evals: {max_evals}")

    # Initialize SUMO wrapper inside the parallel process to avoid port collisions
    wrapper, _, _, _, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline_data,
        fitness_function=_traffic_fitness,
    )

    threshold = THRESHOLDS[tree_name]
    fos_file_path = f"src/outputs/gomea_fos_{tree_name}.txt"
    num_clusters = build_fos_file(dist_path, threshold, tls_to_genes, num_genes, fos_file_path)

    linkage_model = gomea.linkage.Custom(file=fos_file_path)

    fitness_inst = TLSFitness(num_genes, max_evals, pop_size)
    fitness_inst.wrapper = wrapper

    rvgom = gomea.RealValuedGOMEA(
        fitness=fitness_inst,
        linkage_model=linkage_model,
        lower_init_range=GENE_LOW,
        upper_init_range=GENE_HIGH,
        random_seed=seed,
        base_population_size=pop_size,
        max_number_of_populations=1, 
        max_number_of_evaluations=max_evals,
        verbose=False,
    )

    t0 = time.time()
    try:
        res = rvgom.run()
        best_sol_str = res.final_metrics_dict["best_solution"]
        solution = np.array(json.loads(best_sol_str))
        best_cost = float(res.final_metrics_dict["best_obj_val"])
        evals_done = int(res.final_metrics_dict["evaluations"])
        fitness_history = fitness_inst.history

    except StopOptimization as e:
        # Strictly caught MAX_EVALS limit
        print(f"\n[Process: {tree_name}] Halted precisely at MAX_EVALS: {e}")
        solution = fitness_inst.best_sol
        best_cost = fitness_inst.best_fit
        evals_done = fitness_inst.eval_count
        fitness_history = fitness_inst.history

    except RuntimeError as e:
        # Standard GOMEA termination handling
        if "evaluations" in str(e) or "generations" in str(e):
            print(f"\n[Process: {tree_name}] GOMEA Native limit reached: {e}")
            solution = fitness_inst.best_sol
            best_cost = fitness_inst.best_fit
            evals_done = fitness_inst.eval_count
            fitness_history = fitness_inst.history
        else:
            raise e

    elapsed = time.time() - t0
    print(f"[Process: {tree_name}] Completed in {elapsed:.1f}s | Best Fitness: {best_cost:.2f} | Evals: {evals_done}")

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
        "strategy":           "parallel_eval",
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

    # Gather gene mapping metadata (calculate once on main thread)
    tls_to_genes = {}
    idx = 0
    for tls_id in sorted(baseline["tls_data"]):
        phases = sorted(baseline["tls_data"][tls_id])
        tls_to_genes[tls_id] = (idx, idx + len(phases))
        idx += len(phases)
    
    num_genes = idx

    out_dir = Path("src/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = {
        "shortest":  out_dir / "tls_distances_shortest.json",
        "euclidian": out_dir / "tls_distances_euclidian.json",
        "fastest":   out_dir / "tls_distances_fastest.json",
    }

    summary = {}
    
    # Run experiments in parallel using ProcessPoolExecutor
    print(f"Starting {len(trees)} experiments in PARALLEL...")
    
    with ProcessPoolExecutor(max_workers=len(trees)) as executor:
        futures = {}
        for tree_name, dist_path in trees.items():
            # Submit the task to an independent process
            future = executor.submit(
                run_single_experiment,
                tree_name=tree_name,
                dist_path=str(dist_path),
                baseline_data=baseline,
                num_genes=num_genes,
                tls_to_genes=tls_to_genes,
                pop_size=POPULATION_SIZE,
                max_evals=MAX_EVALS,
            )
            futures[future] = tree_name

        # Process results as they complete
        for future in as_completed(futures):
            tree_name = futures[future]
            try:
                res = future.result()
                
                out_file = out_dir / f"gomea_{tree_name}_parallel.json"
                with open(out_file, "w") as f:
                    json.dump(res, f, indent=4)
                    
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