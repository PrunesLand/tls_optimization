import libsumo as traci
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "osm.sumocfg")
IGNORE_THRESHOLD = 1

def get_min_max_stats():
    try:
        traci.start(["sumo", "-c", CONFIG_FILE, "--no-step-log", "true", "--no-warnings", "true"])
    except Exception as e:
        return print(f"Error: {e}")

    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids: return print("No traffic lights found.")

    stats = {
        'lanes': [], 'signals': [], 'phases': [], 'cycles': [],
        'green': [], 'yellow': [], 'red': [], 'modifiable': []
    }

    for tls_id in tls_ids:
        links = traci.trafficlight.getControlledLinks(tls_id)
        stats['signals'].append(len(links))
        stats['lanes'].append(sum(len(group) for group in links))

        logics = traci.trafficlight.getAllProgramLogics(tls_id)
        if not logics: continue
        
        phases = [p for p in logics[0].phases if p.duration >= IGNORE_THRESHOLD]
        stats['phases'].append(len(phases))
        stats['cycles'].append(sum(p.duration for p in phases))
        
        mod_count = 0
        for p in phases:
            state = p.state.lower()
            if 'y' in state or 'u' in state:
                stats['yellow'].append(p.duration)
            elif 'g' in state:
                stats['green'].append(p.duration)
                mod_count += 1
            else:
                stats['red'].append(p.duration)
                mod_count += 1
        stats['modifiable'].append(mod_count)

    traci.close()

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