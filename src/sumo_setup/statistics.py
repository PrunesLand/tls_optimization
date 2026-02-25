import libsumo as traci
import os
import json
from pathlib import Path
from config import SUMO_ARGS

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
        
        phases = [p for p in logics[0].phases if p.duration >= 1] # Filter out phases with zero duration
        if not phases: continue # Safeguard against empty phase lists
        
        temp_green, temp_yellow, temp_red = [], [], []
        mod_count = 0
        
        # Counts how many phases in the cycle have actual stop/go logic
        for p in phases:
            state = p.state.lower()
            if 'g' in state or 'r' in state:
                mod_count += 1
        
        if mod_count == 0: # Skip TLS if no modifiable phases found
            continue

        num_links = len(phases[0].state)
        for i in range(num_links):
            link_green, link_yellow, link_red = 0, 0, 0
            
            for p in phases:
                char = p.state[i].lower() 
                if char == 'g':
                    link_green += p.duration
                elif char in ['y', 'u']:
                    link_yellow += p.duration
                elif char == 'r':
                    link_red += p.duration
            
            if link_green > 0: temp_green.append(link_green)
            if link_yellow > 0: temp_yellow.append(link_yellow)
            if link_red > 0: temp_red.append(link_red)

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
        json_filepath = output_dir / "tls_statistics.json"
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