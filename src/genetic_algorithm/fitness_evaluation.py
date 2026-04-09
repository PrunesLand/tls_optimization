import json
from pathlib import Path
import libsumo as traci
import copy
from config import SUMO_ARGS

import json
import copy
import libsumo as traci
import sys
from pathlib import Path

# Add project root to sys.path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import SUMO_ARGS, BASELINE_TRAFFIC_DATA

with open(BASELINE_TRAFFIC_DATA, 'r') as f:
    BASELINE_DATA = json.load(f)


def fitness_function(tls_durations):
    """
    Evaluates a lightweight dictionary of {tls_id: [durations]}
    """
    traci.start(SUMO_ARGS)

    # 2. Iterate through the lightweight dictionary provided by PyGAD
    for tl_id, durations in tls_durations.items():
        phase_list = []
        
        # Look up the baseline phases for this specific traffic light
        baseline_phases = BASELINE_DATA["tls_data"][tl_id]
        phase_keys = sorted(baseline_phases.keys())

        # Combine the lightweight duration with the baseline state string
        for i, phase_key in enumerate(phase_keys):
            duration = int(durations[i])
            state = baseline_phases[phase_key]["state"] 
            
            # Create the phase object for SUMO
            phase_list.append(traci.trafficlight.Phase(duration, state))

        # Assign the logic to the simulator
        logic = traci.trafficlight.Logic(
            programID="custom",
            type=0,
            currentPhaseIndex=0,
            phases=phase_list
        )
        traci.trafficlight.setProgramLogic(tl_id, logic)
        traci.trafficlight.setProgram(tl_id, "custom")

    # 3. Run the simulation and calculate costs
    total_delay = 0.0
    total_vehicles = 0

    simulation_steps = int(traci.simulation.getEndTime())

    for step in range(simulation_steps):
        traci.simulationStep()
        vehicle_ids = traci.vehicle.getIDList()

        for vid in vehicle_ids:
            speed = traci.vehicle.getSpeed(vid)
            max_speed = traci.vehicle.getMaxSpeed(vid)
            if max_speed > 0:
                total_delay += (1.0 - speed / max_speed)

        total_vehicles = max(total_vehicles, len(vehicle_ids))

    traci.close()

    # Calculate final fitness score
    fitness = total_delay + (total_vehicles * 10)
    return fitness


def evaluate_individual(individual):
    fitness = fitness_function(individual)

    evaluated_individual = copy.deepcopy(individual)
    evaluated_individual["fitness"] = fitness

    return evaluated_individual

def evaluate_population(population):
    evaluated_population = []
    for individual in population:
        evaluated_individual = evaluate_individual(individual)
        evaluated_population.append(evaluated_individual)

    output_dir = Path("src/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_filepath = output_dir / "population_tls_data_with_fitness.json"
    
    with open(json_filepath, "w") as f:
        json.dump(evaluated_population, f, indent=4)
    
    return evaluated_population