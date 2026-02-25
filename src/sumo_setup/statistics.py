import libsumo as traci
import os
import json
from pathlib import Path
from config import SUMO_ARGS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IGNORE_THRESHOLD = 1

def get_min_max_stats():
    try:
        traci.start(SUMO_ARGS)
    except Exception as e:
        return print(f"Error starting SUMO: {e}")

    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids: return print("No traffic lights found.")

    stats = {
        'lanes': [], 'signals': [], 'phases': [], 'cycles': [],
        'green': [], 'yellow': [], 'red': [], 'modifiable': []
    }
    
    tls_json_data = {}

    for tls_id in tls_ids:
        logics = traci.trafficlight.getAllProgramLogics(tls_id)
        if not logics: continue
        
        phases = [p for p in logics[0].phases if p.duration >= IGNORE_THRESHOLD]
        
        temp_green, temp_yellow, temp_red = [], [], []
        mod_count = 0
        
        for p in phases:
            state = p.state.lower()
            if 'y' in state or 'u' in state:
                temp_yellow.append(p.duration)
            elif 'g' in state:
                temp_green.append(p.duration)
                mod_count += 1
            else:
                temp_red.append(p.duration)
                mod_count += 1
        
        if mod_count == 0: # Skip TLS if no modifiable phases found
            continue

        links = traci.trafficlight.getControlledLinks(tls_id)
        
        num_signals = len(links)
        num_lanes = sum(len(group) for group in links)
        num_phases = len(phases)
        cycle_time = sum(p.duration for p in phases)

        stats['signals'].append(num_signals)
        stats['lanes'].append(num_lanes)
        stats['phases'].append(num_phases)
        stats['cycles'].append(cycle_time)
        stats['modifiable'].append(mod_count)
        stats['green'].extend(temp_green)
        stats['yellow'].extend(temp_yellow)
        stats['red'].extend(temp_red)
        
        tls_json_data[tls_id] = {
            "Lanes Controlled": num_lanes,
            "Signal Heads": num_signals,
            "Phases per Cycle": num_phases,
            "Modifiable Phases": mod_count,
            "Cycle Time (s)": cycle_time,
            "Green Durations (s)": temp_green,
            "Yellow Durations (s)": temp_yellow,
            "Red Durations (s)": temp_red
        }

    traci.close()

    try:
        output_dir = Path("src/sumo_setup")
        output_dir.mkdir(parents=True, exist_ok=True)
        json_filepath = output_dir / "TLS_statistics.json"
        with open(json_filepath, 'w') as f:
            json.dump(tls_json_data, f, indent=4)
        print(f"\n Individual traffic light data saved to: {json_filepath}")
    except Exception as e:
        print(f"\n[ERROR] Could not save JSON file: {e}")

    def report(label, data, unit=""):
        if not data: return print(f"{label:20}: None")
        print(f"{label:20}: Max {max(data)}{unit} | Min {min(data)}{unit} | Total {len(data) if 'Duration' in label else sum(data)}")

    print(f"\n{' NETWORK EXTREMA REPORT ':=^50}")
    report("Lanes Controlled", stats['lanes'])
    report("Signal Heads", stats['signals'])
    report("Phases per Cycle", stats['phases'])
    report("Modifiable Phases", stats['modifiable'])
    report("Cycle Time", stats['cycles'], "s")
    print("-" * 50)
    report("Green Duration", stats['green'], "s")
    report("Yellow Duration", stats['yellow'], "s")
    report("Red Duration", stats['red'], "s")
    print("=" * 50)

if __name__ == "__main__":
    get_min_max_stats()