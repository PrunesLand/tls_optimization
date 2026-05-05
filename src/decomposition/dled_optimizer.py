import numpy as np
import json
import time
import os
import sys
import concurrent.futures
from pathlib import Path

# Add project root to sys.path to import config and other modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import specific variables from your config
from config import BASELINE_TRAFFIC_DATA, NUM_PROCESSORS
from src.genetic_algorithm.fitness_evaluation import fitness_function

# Load Baseline Data
with open(BASELINE_TRAFFIC_DATA, 'r') as f:
    GLOBAL_BASELINE_DATA = json.load(f)

def create_tls_mapping(baseline_data):
    """Creates a static mapping of the flat vector to specific TLS IDs."""
    mapping = []
    gene_idx = 0
    for tls_id in sorted(baseline_data["tls_data"].keys()):
        num_phases = len(baseline_data["tls_data"][tls_id].keys())
        mapping.append({
            "tls_id": tls_id,
            "num_phases": num_phases,
            "start_idx": gene_idx,
            "end_idx": gene_idx + num_phases
        })
        gene_idx += num_phases
    return mapping

TLS_MAPPING = create_tls_mapping(GLOBAL_BASELINE_DATA)
NUM_GENES = TLS_MAPPING[-1]["end_idx"]

def get_normalized_durations(vector, mapping):
    """Normalizes vector using mapping and enforces the 90-second rule."""
    tls_durations = {}
    for tls in mapping:
        tls_id = tls["tls_id"]
        num_phases = tls["num_phases"]
        raw_durations = vector[tls["start_idx"] : tls["end_idx"]]
        
        total_raw = sum(raw_durations)
        if total_raw <= 0:
            durations = [90 // num_phases] * num_phases
            durations[-1] += 90 - sum(durations)
        else:
            durations = [max(1, int(round(d * 90 / total_raw))) for d in raw_durations]
            current_sum = sum(durations)
            if current_sum != 90:
                diff = 90 - current_sum
                idx = int(np.argmax(durations))
                durations[idx] += diff
                if durations[idx] < 1:
                    durations[idx] = 1
                    other_idx = (idx + 1) % num_phases
                    durations[other_idx] += (90 - sum(durations))
        tls_durations[tls_id] = durations
    return tls_durations

# ---------------------------------------------------------
# CACHING & PARALLEL EVALUATION LOGIC
# ---------------------------------------------------------

# Global cache to hold our pre-calculated SUMO simulation results
EVALUATION_CACHE = {}

def evaluate_fitness_raw(vector):
    """Runs the actual SUMO simulation for a given vector. Safe for multiprocessing."""
    tls_durations = get_normalized_durations(vector, TLS_MAPPING)
    try:
        # We negate the cost so higher is better, matching the DLED > comparisons
        cost = fitness_function(tls_durations)
        return -cost
    except Exception as e:
        print(f"Error evaluating fitness: {e}")
        return -9999999.0

def evaluate_fitness(vector):
    """Instantly returns the fitness of a specific gene vector from the pre-populated cache."""
    return EVALUATION_CACHE[tuple(vector)]

def perturb(gene_value):
    """
    Perturbs an integer gene within the 5-85 boundary.
    The shift must be large enough to survive the 90s normalization process.
    """
    if gene_value < 45:
        return 85
    else:
        return 5

def generate_unique_states(base_individual):
    """
    Generates all unique gene combinations DLED will possibly check mathematically.
    This replaces the need for SUMO to be run iteratively during the DLED loops.
    """
    states = set()
    
    # 1. Base individual (0 genes perturbed)
    states.add(tuple(base_individual))
    
    n = len(base_individual)
    for i in range(n):
        # 2. 1-gene perturbed states
        ind_i = list(base_individual)
        ind_i[i] = perturb(ind_i[i])
        states.add(tuple(ind_i))
        
        # 3. 2-gene perturbed states
        for j in range(i + 1, n):
            ind_ij = list(ind_i)
            ind_ij[j] = perturb(ind_ij[j])
            states.add(tuple(ind_ij))
            
    return list(states)

# ---------------------------------------------------------
# DLED LOGIC
# ---------------------------------------------------------

def extract_dled_linkage(gene_no, ind):
    """
    Executes Direct Linkage Empirical Discovery using the fast cache.
    Returns a Directly Dependent Genes List (DDGL) for the target gene.
    """
    # Baseline comparison [cite: 1195, 1222, 1224]
    fit_orig = evaluate_fitness(ind)
    
    ind_pert = ind.copy()
    ind_pert[gene_no] = perturb(ind_pert[gene_no])
    fit_pert = evaluate_fitness(ind_pert)
    
    dep_genes = []
    
    for other_no in range(len(ind)):
        if other_no == gene_no:
            continue
            
        ind_mod = ind.copy()
        ind_mod[other_no] = perturb(ind_mod[other_no])
        fit_orig_mod = evaluate_fitness(ind_mod)
        
        ind_pert_mod = ind_pert.copy()
        ind_pert_mod[other_no] = perturb(ind_pert_mod[other_no])
        fit_pert_mod = evaluate_fitness(ind_pert_mod)
        
        # Check if the mutation triggered an improvement[cite: 1200].
        new_val_better = fit_orig_mod > fit_orig
        pert_new_val_better = fit_pert_mod > fit_pert
        
        # If flipping 'other_no' improves one state but not the other, it is directly linked[cite: 1200].
        if new_val_better != pert_new_val_better:
            dep_genes.append(other_no)
            
    return dep_genes

def run_dled_analysis():
    print(f"Starting Parallel Direct Linkage Empirical Discovery (DLED) on {NUM_GENES} genes...")
    start_time = time.time()
    
    base_individual = list(np.random.randint(5, 86, size=NUM_GENES))
    
    # 1. Pre-calculate the exact list of unique states required
    print("Generating unique simulation states to evaluate...")
    states_to_evaluate = generate_unique_states(base_individual)
    total_states = len(states_to_evaluate)
    print(f"Total unique SUMO simulations required: {total_states}")
    
    # 2. Blast them through the SUMO simulator in parallel
    print(f"Running simulations in parallel across processors. Please wait...")
    global EVALUATION_CACHE
    EVALUATION_CACHE.clear()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSORS) as executor:
        # map() automatically handles chunking and distributes the vectors to idle cores
        results = list(executor.map(evaluate_fitness_raw, states_to_evaluate))
        
    # Zip the results into our global dictionary
    for state, fitness in zip(states_to_evaluate, results):
        EVALUATION_CACHE[state] = fitness
        
    print("Simulations complete! Mapping dependencies...")
    
    # 3. Run the DLED algorithm 
    ddgl_results = {}
    for gene_no in range(NUM_GENES):
        # Obtain the Directly Dependent Genes List (DDGL) [cite: 1239]
        dependent_genes = extract_dled_linkage(gene_no, base_individual)
        ddgl_results[f"gene_{gene_no}"] = dependent_genes
        print(f"  -> Gene {gene_no+1}/{NUM_GENES} Dependencies: {dependent_genes}")
        
    end_time = time.time()
    duration = end_time - start_time
    
    # Format the results
    output_dir = Path("src/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_json = {
        "execution_time_seconds": duration,
        "total_sumo_evaluations": len(EVALUATION_CACHE),
        "parallel_cores_used": NUM_PROCESSORS,
        "base_individual": base_individual,
        "ddgl_map": ddgl_results,
        "note": "A list [x, y] means gene_i is directly dependent on genes x and y."
    }
    
    output_file = output_dir / "dled_linkage_results.json"
    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=4)
        
    print(f"\nDLED Analysis Complete in {duration:.2f} seconds.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    run_dled_analysis()