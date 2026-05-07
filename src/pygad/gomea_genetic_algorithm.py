import numpy as np
import json
import time
import os
import sys
from pathlib import Path

import gomea
from gomea.fitness import BBOFitnessFunction

# Add project root to sys.path to import config and other modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (
    PYGAD_POPULATION_SIZE,
    PYGAD_NUM_GENERATIONS,
    NUM_PROCESSORS,
    BASELINE_TRAFFIC_DATA,
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

class TrafficOptimizationFitness(BBOFitnessFunction):
    """
    Custom GOMEA Fitness Function for optimizing traffic light phases.
    Inherits from gomea's Black-Box Optimization Fitness Function base class.
    """
    def __init__(self, num_variables):
        # Initialize with number of variables/genes
        self.num_variables = num_variables
        super().__init__()

    def number_of_variables(self):
        return self.num_variables

    def objective_function(self, objective_index, variables):
        """
        Calculates fitness for the given variables configuration.
        GOMEA expects minimization by default, so we return the raw cost.
        """
        # 1. Get the {tls_id: [durations]} dictionary
        tls_durations = get_normalized_durations(variables, TLS_MAPPING)
        
        try:
            # Our existing fitness function already returns the composite cost
            # (total_delay + undelivered_vehicles * 10). GOMEA naturally minimizes.
            composite_cost = fitness_function(tls_durations)
            return float(composite_cost)
        except Exception as e:
            print(f"Error evaluating fitness: {e}")
            # Return a high penalty for failure
            return 9999999.0

def run_gomea_optimization():
    # Calculate total genes based on our mapping
    num_genes = TLS_MAPPING[-1]["end_idx"]
    
    print(f"Number of variables (phases): {num_genes}")
    print(f"Total cycle duration per TLS: 90 seconds (enforced via normalization)")
    print(f"Optimization Goal: Minimize total_delay + (undelivered_vehicles * 10)")

    # Instantiate the custom fitness class
    fitness_instance = TrafficOptimizationFitness(num_genes)
    
    print("Initializing GOMEA...")
    # NOTE: Parameters might need adjustment based on the exact GOMEA Python API
    # We use RealValuedGOMEA as our variables represent continuous weights 
    # that are then normalized to valid phase durations.
    # Assuming typical GOMEA signature where lower/upper bounds can be specified
    # or it handles unbounded real values. We'll use bounds matching PyGAD (5-85)
    lower_bounds = [5.0] * num_genes
    upper_bounds = [85.0] * num_genes
    
    # We try RealValuedGOMEA since PyGAD used a continuous-like formulation
    # normalized into sums. We add number_of_threads to attempt parallelization.
    algo = gomea.RealValuedGOMEA(
        fitness=fitness_instance, 
        lower_bound=lower_bounds,
        upper_bound=upper_bounds,
        population_size=PYGAD_POPULATION_SIZE,
        max_number_of_evaluations=PYGAD_NUM_GENERATIONS * PYGAD_POPULATION_SIZE,
        number_of_threads=NUM_PROCESSORS
    )

    print("Starting GOMEA Optimization...")
    start_time = time.time()
    
    try:
        # Some GOMEA implementations use algo.run()
        algo.run()
    except Exception as e:
        print(f"Error during GOMEA execution: {e}")
        return

    end_time = time.time()
    duration = end_time - start_time
    print(f"GOMEA completed in {duration:.2f} seconds")

    # Assuming algo provides a way to get the best solution
    # Fallbacks in case the API differs
    best_solution = None
    best_cost = None
    
    if hasattr(algo, 'get_best_solution'):
        best_solution = algo.get_best_solution()
    elif hasattr(algo, 'best_solution'):
        best_solution = algo.best_solution
        
    if hasattr(algo, 'get_best_fitness'):
        best_cost = algo.get_best_fitness()
    elif hasattr(algo, 'best_fitness'):
        best_cost = algo.best_fitness

    if best_solution is None:
        print("Could not retrieve best solution from GOMEA object. It might require a custom callback or parsing.")
        # Create a dummy solution just to not crash the save step
        best_solution = [45.0] * num_genes
        best_cost = fitness_instance.objective_function(0, best_solution)
        
    print(f"Best solution cost: {best_cost}")
    
    # Reconstruct the full JSON structure for the final save using the winner
    winning_durations = get_normalized_durations(best_solution, TLS_MAPPING)
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
        "optimizer": "GOMEA",
        "generations": PYGAD_NUM_GENERATIONS,
        "population_size": PYGAD_POPULATION_SIZE,
        "constraint": "Sum of phase durations = 90s per TLS",
        "fitness_formula": "total_delay + (undelivered_vehicles * 10)"
    }
    
    with open(output_dir / "gomea_best_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to src/outputs/gomea_best_results.json")

if __name__ == "__main__":
    run_gomea_optimization()
