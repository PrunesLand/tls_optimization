import libsumo as traci
import json
from config import SUMO_ARGS
from pathlib import Path

def generate_data():

    traci.start(SUMO_ARGS)
    tls_ids = traci.trafficlight.getIDList()

    output_data = {"configurations": {}}

    for tls_id in tls_ids:
        logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
        
        active_logic = logics[0]
        
        total_green_duration = 0
        total_red_duration = 0
        
        for phase in active_logic.phases:
            state = phase.state
            duration = phase.duration
            
            if 'G' in state or 'g' in state:
                total_green_duration += duration
            
            elif 'r' in state and 'y' not in state and 'Y' not in state and 'G' not in state and 'g' not in state:
                total_red_duration += duration

        output_data["configurations"][tls_id] = {
            "green_duration": total_green_duration,
            "red_duration": total_red_duration
        }

    traci.close()

    output_dir = Path("src/sumo_setup")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_filepath = output_dir / "traffic_cycle_results.json"
    
    with open(json_filepath, "w") as f:
        json.dump(output_data, f, indent=4)

    return 

if __name__ == "__main__":
    generate_data()