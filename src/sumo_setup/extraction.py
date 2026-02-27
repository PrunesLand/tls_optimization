import libsumo as traci
from config import SUMO_ARGS

def extract_traffic_light_data(detail: bool = False):
    try:
        traci.start(SUMO_ARGS)
    except Exception as e:
        print(f"Error starting SUMO: {e}")
        return {}

    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids: 
        print("No traffic lights found.")
        return {}

    tls_json_data = {}
    
    for tls_id in tls_ids:
        logics = traci.trafficlight.getAllProgramLogics(tls_id)
        if not logics: continue
        
        # Filter out phases with duration < 1s
        phases = [p for p in logics[0].phases if p.duration >= 1]
        if not phases: continue
        
        # Skip TLS if it has no modifiable logic (Green/Red)
        has_logic = any('g' in p.state.lower() or 'r' in p.state.lower() for p in phases)
        if not has_logic:
            continue

        # Create the Phase-by-Phase dictionary
        phase_map = {}
        for idx, p in enumerate(phases):
            phase_key = f"phase_{idx + 1}"
            phase_map[phase_key] = {
                "duration": int(p.duration),
                "state": p.state,
            }

        if detail:
            links = traci.trafficlight.getControlledLinks(tls_id)
            tls_json_data[tls_id] = {
                "phases": phase_map,
                "metadata": {
                    "number_of_signals": len(links),
                    "number_of_lanes": sum(len(group) for group in links),
                    "cycle_time": sum(p.duration for p in phases),
                    "total_phases": len(phases)
                }
            }
        else:
            tls_json_data[tls_id] = phase_map

    traci.close()
    return { "tls_data": tls_json_data, "fitness": 0 }