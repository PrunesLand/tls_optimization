import json
from pathlib import Path
from src.sumo_setup.extraction import extract_traffic_light_data

def get_min_max_stats():
    extracted = extract_traffic_light_data(detail=True)

    if not extracted:
        print("No data extracted. Ensure SUMO is configured correctly.")
        return

    tls_json_data = extracted.get("tls_data", {})

    if not tls_json_data:
        print("No traffic light data found.")
        return

    stats = {
        'lanes': [], 'signals': [], 'phases': [], 'cycles': [],
        'green': [], 'yellow': [], 'red': []
    }

    for _, data in tls_json_data.items():
        metadata = data.get("metadata", {})
        phases = data.get("phases", {})

        stats['lanes'].append(metadata.get("number_of_lanes", 0))
        stats['signals'].append(metadata.get("number_of_signals", 0))
        stats['phases'].append(metadata.get("total_phases", 0))
        stats['cycles'].append(metadata.get("cycle_time", 0))

        for _, p_info in phases.items():
            state = p_info['state'].lower()
            duration = p_info['duration']
            if 'g' in state:
                stats['green'].append(duration)
            elif 'y' in state or 'u' in state:
                stats['yellow'].append(duration)
            else:
                stats['red'].append(duration)

    try:
        output_dir = Path("src/sumo_setup")
        output_dir.mkdir(parents=True, exist_ok=True)
        json_filepath = output_dir / "tls_statistics.json"
        with open(json_filepath, 'w') as f:
            json.dump(extracted, f, indent=4)
        print(f"\nIndividual traffic light data saved to: {json_filepath}")
    except Exception as e:
        print(f"\n[ERROR] Could not save JSON file: {e}")

    def report(label, data, unit=""):
        if not data:
            return print(f"{label:20}: None")
        print(
            f"{label:20}: Max {max(data):>3}{unit} | "
            f"Min {min(data):>3}{unit} | Total {len(data)}"
        )

    print(f"\nTotal Intersections Analyzed: {len(tls_json_data)}")
    print(f"\n{' NETWORK EXTREMA REPORT ':=^50}")
    report("Lanes Controlled", stats['lanes'])
    report("Signal Heads", stats['signals'])
    report("Phases per Cycle", stats['phases'])
    report("Cycle Time", stats['cycles'], "s")
    print("-" * 50)
    report("Green Duration", stats['green'], "s")
    report("Yellow Duration", stats['yellow'], "s")
    report("Red Duration", stats['red'], "s")
    print("=" * 50)

if __name__ == "__main__":
    get_min_max_stats()