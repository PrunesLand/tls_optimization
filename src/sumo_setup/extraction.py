import libsumo as traci

from config import SUMO_ARGS

def extract_traffic_light_data():
    try:
        traci.start(SUMO_ARGS)
    except Exception as e:
        return print(f"Error starting SUMO: {e}")

    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids: return print("No traffic lights found.")

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

        tls_json_data[tls_id] = {
            "Green Durations (s)": temp_green,
            "Yellow Durations (s)": temp_yellow,
            "Red Durations (s)": temp_red
        }
    traci.close()
    return tls_json_data

        