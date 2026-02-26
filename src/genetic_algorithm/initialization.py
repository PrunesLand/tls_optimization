import json
import random
import copy

def generate_individual(num_phases, cycle_length=90):
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

def generate_population(input_json_path, output_json_path):
    with open(input_json_path, 'r') as file:
        tls_data = json.load(file)
        
    new_tls_data = copy.deepcopy(tls_data)
    
    for _, phases_dict in new_tls_data.items():
        
        num_phases = len(phases_dict)
        
        new_durations = generate_individual(num_phases, cycle_length=90)
        
        for index, phase_key in enumerate(sorted(phases_dict.keys())):
            phases_dict[phase_key]['duration'] = new_durations[index]
            
    with open(output_json_path, 'w') as file: #temporarily write a new json for sanity check
        json.dump(new_tls_data, file, indent=4)
        
    return new_tls_data