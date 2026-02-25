import json
from pathlib import Path

from .extraction import extract_traffic_light_data

def generate_data():

    output_data = extract_traffic_light_data()

    output_dir = Path("src/sumo_setup")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_filepath = output_dir / "traffic_cycle_results.json"
    
    with open(json_filepath, "w") as f:
        json.dump(output_data, f, indent=4)

    return 

if __name__ == "__main__":
    generate_data()