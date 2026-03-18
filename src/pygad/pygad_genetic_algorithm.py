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

# Load baseline data once
with open(BASELINE_TRAFFIC_DATA, 'r') as f:
    GLOBAL_BASELINE_DATA = json.load(f)

def vector_to_json(vector, baseline_data):
    """
    Converts a chromosome vector to the JSON format used by the simulation.
    Ensures that for each TLS, the sum of phase durations is exactly 90 seconds.
    """
    new_data = json.loads(json.dumps(baseline_data)) # Deep copy
    gene_idx = 0
    
    # Sort TLS IDs to ensure consistent mapping
    for tls_id in sorted(new_data["tls_data"].keys()):
        phases = new_data["tls_data"][tls_id]
        phase_keys = sorted(phases.keys())
        num_phases = len(phase_keys)
        
        # Get the genes (raw durations) for this TLS
        raw_durations = vector[gene_idx : gene_idx + num_phases]
        gene_idx += num_phases
        
        # Normalize raw_durations so they sum to 90
        total_raw = sum(raw_durations)
        if total_raw <= 0:
            # Fallback: distribute evenly if all genes are 0 or negative
            durations = [90 // num_phases] * num_phases
            durations[-1] += 90 - sum(durations)
        else:
            # Scale to 90, ensuring each phase has at least 1 second
            durations = [max(1, int(round(d * 90 / total_raw))) for d in raw_durations]
            
            # Adjust to ensure exact sum of 90 due to rounding
            current_sum = sum(durations)
            if current_sum != 90:
                diff = 90 - current_sum
                # Adjust the largest duration to minimize relative impact
                idx = int(np.argmax(durations))
                durations[idx] += diff
                # Final safety check for 1-second minimum
                if durations[idx] < 1:
                    durations[idx] = 1
                    new_sum = sum(durations)
                    # Distribute remaining difference to another phase
                    other_idx = (idx + 1) % num_phases
                    durations[other_idx] += (90 - new_sum)

        for i, phase_key in enumerate(phase_keys):
            new_data["tls_data"][tls_id][phase_key]["duration"] = int(durations[i])
            
    return new_data

def pygad_fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness function for PyGAD.
    Optimizes the metric: total_delay + (total_vehicles * 10)
    where total_vehicles are cars that did not reach the destination.
    """
    individual_data = vector_to_json(solution, GLOBAL_BASELINE_DATA)
    
    try:
        # fitness_function from fitness_evaluation.py returns: total_delay + (total_vehicles * 10)
        composite_fitness = fitness_function(individual_data)
        # PyGAD maximizes fitness, but we want to minimize delay/cost.
        return -composite_fitness
    except Exception as e:
        print(f"Error evaluating fitness: {e}")
        return -9999999.0 # Large penalty for failed simulations

def custom_callback(ga_instance):
    solution, fitness, idx = ga_instance.best_solution()
    print(f"Generation {ga_instance.generations_completed} completed. Best Cost: {-fitness:.2f}")

def run_genetic_algorithm():
    # 1. Determine number of genes (total phases across all TLS)
    num_genes = 0
    for tls_id in GLOBAL_BASELINE_DATA["tls_data"]:
        num_genes += len(GLOBAL_BASELINE_DATA["tls_data"][tls_id])
    
    print(f"Number of genes (phases): {num_genes}")
    print(f"Total cycle duration per TLS: 90 seconds (enforced via normalization)")
    print(f"Optimization Goal: Minimize total_delay + (undelivered_vehicles * 10)")

    # 2. Define gene space
    gene_space = {'low': 1, 'high': 100}

    # 3. Initialize PyGAD
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

    # 4. Run the GA
    print("Starting Genetic Algorithm...")
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"GA completed in {duration:.2f} seconds")

    # 5. Get the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_cost = -float(solution_fitness)
    print(f"Best solution cost (lower is better): {best_cost}")
    
    # Normalize the best solution for the final output
    best_json = vector_to_json(solution, GLOBAL_BASELINE_DATA)
    best_json["composite_cost"] = best_cost
    
    # 6. Save results
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
