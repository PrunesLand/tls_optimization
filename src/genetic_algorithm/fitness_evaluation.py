import json
from pathlib import Path
import libsumo as traci
import copy
from config import SUMO_ARGS

def fitness_function(data):
    traci.start(SUMO_ARGS)

    traffic_lights = data.get("tls_data", {})

    for tl_id, phases in traffic_lights.items():
        phase_list = []
        for phase_key in sorted(phases.keys()):
            phase = phases[phase_key]
            duration = phase["duration"]
            state = phase["state"]
            phase_list.append(traci.trafficlight.Phase(duration, state))

        logic = traci.trafficlight.Logic(
            programID="custom",
            type=0,
            currentPhaseIndex=0,
            phases=phase_list
        )
        traci.trafficlight.setProgramLogic(tl_id, logic)
        traci.trafficlight.setProgram(tl_id, "custom")

    total_delay = 0.0
    total_vehicles = 0

    simulation_steps = traci.simulation.getEndTime()

    for step in range(int(simulation_steps)):
        traci.simulationStep()

        vehicle_ids = traci.vehicle.getIDList()

        for vid in vehicle_ids:
            speed = traci.vehicle.getSpeed(vid)
            max_speed = traci.vehicle.getMaxSpeed(vid)
            if max_speed > 0:
                total_delay += (1.0 - speed / max_speed)

        total_vehicles = max(total_vehicles, len(vehicle_ids))

    traci.close()

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