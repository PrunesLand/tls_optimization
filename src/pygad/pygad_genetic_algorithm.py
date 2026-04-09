import pygad
import numpy as np
import json
import time
import os
import sys
from pathlib import Path

# Add project root to sys.path to import config and other modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (
    PYGAD_POPULATION_SIZE,
    PYGAD_NUM_GENERATIONS,
    PYGAD_MUTATION_PERCENT_GENES,
    PYGAD_NUM_PARENTS_MATING,
    PYGAD_KEEP_PARENTS,
    NUM_PROCESSORS,
    BASELINE_TRAFFIC_DATA,
    SUMO_ARGS
)
from src.genetic_algorithm.fitness_evaluation import fitness_function

with open(BASELINE_TRAFFIC_DATA, 'r') as f:
    GLOBAL_BASELINE_DATA = json.load(f)

def create_tls_mapping(baseline_data):
    """
    Creates a static mapping of the flat GA vector to specific TLS IDs.
    This prevents us from needing to parse JSON during the generations.
    """
    mapping = []
    gene_idx = 0
    
    for tls_id in sorted(baseline_data["tls_data"].keys()):
        phase_keys = sorted(baseline_data["tls_data"][tls_id].keys())
        num_phases = len(phase_keys)
        
        mapping.append({
            "tls_id": tls_id,
            "num_phases": num_phases,
            "start_idx": gene_idx,
            "end_idx": gene_idx + num_phases
        })
        gene_idx += num_phases
        
    return mapping

# Create the map in memory immediately
TLS_MAPPING = create_tls_mapping(GLOBAL_BASELINE_DATA)

def get_normalized_durations(vector, mapping):
    """
    Slices the vector using the mapping and enforces the 90-second rule.
    Returns a lightweight dictionary: { 'tls_id': [duration1, duration2, ...] }
    """
    tls_durations = {}
    
    for tls in mapping:
        tls_id = tls["tls_id"]
        num_phases = tls["num_phases"]
        
        # Grab only the genes for this specific traffic light
        raw_durations = vector[tls["start_idx"] : tls["end_idx"]]
        
        # Enforce the 90-second constraint mathematically
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
                    new_sum = sum(durations)
                    other_idx = (idx + 1) % num_phases
                    durations[other_idx] += (90 - new_sum)
                    
        tls_durations[tls_id] = durations
        
    return tls_durations

def pygad_fitness_func(ga_instance, solution, solution_idx):
    """
    Passes the lightweight dict to the simulator to calculate cost.
    """
    # 1. Get the {tls_id: [durations]} dictionary (Lightning fast!)
    tls_durations = get_normalized_durations(solution, TLS_MAPPING)
    
    try:
        composite_fitness = fitness_function(tls_durations)
        return -composite_fitness
    except Exception as e:
        print(f"Error evaluating fitness: {e}")
        return -9999999.0

def custom_callback(ga_instance):
    solution, fitness, idx = ga_instance.best_solution()
    print(f"Generation {ga_instance.generations_completed} completed. Best Cost: {-fitness:.2f}")

def run_genetic_algorithm():
    # Calculate total genes based on our mapping cheat-sheet
    num_genes = TLS_MAPPING[-1]["end_idx"]
    
    print(f"Number of genes (phases): {num_genes}")
    print(f"Total cycle duration per TLS: 90 seconds (enforced via normalization)")
    print(f"Optimization Goal: Minimize total_delay + (undelivered_vehicles * 10)")

    gene_space = {'low': 5, 'high': 85}

    ga_instance = pygad.GA(
        num_generations=PYGAD_NUM_GENERATIONS,
        num_parents_mating=PYGAD_NUM_PARENTS_MATING,
        fitness_func=pygad_fitness_func,
        sol_per_pop=PYGAD_POPULATION_SIZE,
        num_genes=num_genes,
        gene_type=int,
        gene_space=gene_space,
        mutation_percent_genes=PYGAD_MUTATION_PERCENT_GENES,
        keep_parents=PYGAD_KEEP_PARENTS,
        on_generation=custom_callback,
        save_best_solutions=True,
        parallel_processing=["process", NUM_PROCESSORS]
    )

    print("Starting Genetic Algorithm...")
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"GA completed in {duration:.2f} seconds")

    # Extract winning solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_cost = -float(solution_fitness)
    print(f"Best solution cost: {best_cost}")
    
    # Reconstruct the full JSON structure for the final save using the winner
    winning_durations = get_normalized_durations(solution, TLS_MAPPING)
    best_json = json.loads(json.dumps(GLOBAL_BASELINE_DATA))
    for tls_id, phases in winning_durations.items():
        phase_keys = sorted(best_json["tls_data"][tls_id].keys())
        for i, p_key in enumerate(phase_keys):
            best_json["tls_data"][tls_id][p_key]["duration"] = int(phases[i])
            
    best_json["composite_cost"] = best_cost
    
    # Save results
    output_dir = Path("src/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "best_configuration": best_json,
        "execution_time_seconds": duration,
        "best_cost": best_cost,
        "generations": PYGAD_NUM_GENERATIONS,
        "population_size": PYGAD_POPULATION_SIZE,
        "constraint": "Sum of phase durations = 90s per TLS",
        "fitness_formula": "total_delay + (undelivered_vehicles * 10)"
    }
    
    with open(output_dir / "pygad_best_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to src/outputs/pygad_best_results.json")

if __name__ == "__main__":
    run_genetic_algorithm()