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
    BASELINE_TRAFFIC_DATA,
    SUMO_ARGS
)
from src.genetic_algorithm.fitness_evaluation import fitness_function

def vector_to_json(vector, baseline_data):
    new_data = json.loads(json.dumps(baseline_data)) # Deep copy
    gene_idx = 0
    
    # Sort TLS IDs and phase keys to ensure consistent mapping
    for tls_id in sorted(new_data["tls_data"].keys()):
        phases = new_data["tls_data"][tls_id]
        for phase_key in sorted(phases.keys()):
            new_data["tls_data"][tls_id][phase_key]["duration"] = int(vector[gene_idx])
            gene_idx += 1
            
    return new_data

def pygad_fitness_func(ga_instance, solution, solution_idx):
    """Fitness function for PyGAD."""
    # Load baseline data to use as a template
    with open(BASELINE_TRAFFIC_DATA, 'r') as f:
        baseline_data = json.load(f)
    
    individual_data = vector_to_json(solution, baseline_data)
    
    delay = fitness_function(individual_data)
    
    return -delay

def custom_callback(ga_instance):
    print(f"Executing custom method at generation {ga_instance.generations_completed}...")

def run_genetic_algorithm():
    # 1. Load baseline data to determine chromosome length
    with open(BASELINE_TRAFFIC_DATA, 'r') as f:
        baseline_data = json.load(f)
    
    # 2. Determine gene space (min/max durations)
    gene_space = {'low': 5, 'high': 90}
    
    # 3. Calculate number of genes
    num_genes = 0
    for tls_id in baseline_data["tls_data"]:
        num_genes += len(baseline_data["tls_data"][tls_id])
    
    print(f"Number of genes (phases): {num_genes}")

    # 4. Initialize PyGAD
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
        save_best_solutions=True
    )

    # 5. Run the GA
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"GA completed in {duration:.2f} seconds")

    # 6. Get the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best solution fitness: {solution_fitness}")
    
    best_json = vector_to_json(solution, baseline_data)
    best_json["fitness"] = -solution_fitness # Convert back to delay
    
    # 7. Save results
    output_dir = Path("src/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "best_configuration": best_json,
        "execution_time_seconds": duration,
        "best_fitness": float(solution_fitness),
        "generations": PYGAD_NUM_GENERATIONS,
        "population_size": PYGAD_POPULATION_SIZE
    }
    
    with open(output_dir / "pygad_best_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to src/outputs/pygad_best_results.json")

if __name__ == "__main__":
    run_genetic_algorithm()
