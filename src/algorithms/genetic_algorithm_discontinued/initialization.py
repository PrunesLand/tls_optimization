import json
from pathlib import Path
import random
import copy

def _initialize_solution(num_phases, cycle_length=90):
    if num_phases <= 1:
        return [cycle_length]
        
    cuts = sorted(random.sample(range(1, cycle_length), num_phases - 1))
    
    durations = []
    prev_cut = 0
    
    for cut in cuts:
        durations.append(cut - prev_cut)
        prev_cut = cut
        
    durations.append(cycle_length - prev_cut)
    
    return durations

def generate_individual(input_json_path, output_json_path):
    with open(input_json_path, 'r') as file:
        data = json.load(file)
        
    individual = copy.deepcopy(data)
    
    for _, phases_dict in individual['tls_data'].items():
        
        num_phases = len(phases_dict)
        
        new_durations = _initialize_solution(num_phases)
        
        sorted_keys = sorted(phases_dict.keys(), key=lambda x: int(x.split('_')[1]))
        for index, phase_key in enumerate(sorted_keys):
            phases_dict[phase_key]['duration'] = new_durations[index]
            
    with open(output_json_path, 'w') as file: #temporarily write a new json for sanity check
        json.dump(individual, file, indent=4)
        
    return individual

def generate_population(input_json_path, output_json_path, population_size=10):
    population = []
    
    for _ in range(population_size):
        individual = generate_individual(input_json_path, output_json_path)
        population.append(individual)
    
    output_dir = Path("src/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_filepath = output_dir / "population_tls_data.json"
    
    with open(json_filepath, "w") as f:
        json.dump(population, f, indent=4)
        
    return population