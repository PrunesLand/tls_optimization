"""
Compute pairwise fastest-path (road) travel times between all traffic lights
in the SUMO network. Writes the results to a JSON file.
"""

import json
import sys
from pathlib import Path
import sumolib

def road_travel_time(net, node_a, node_b) -> float | None:
    """Return the fastest passenger road travel time in seconds between two nodes."""
    costs = [
        cost
        for e_from in node_a.getOutgoing()
        for e_to in node_b.getIncoming()
        for route, cost in [net.getFastestPath(e_from, e_to, vClass="passenger")]
        if route is not None
    ]
    return min(costs) if costs else None

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    net_file = project_root / "src" / "sumo_setup" / "osm.net.xml.gz"
    out_json = project_root / "src" / "outputs" / "tls_distances_fastest.json"

    print(f"Loading SUMO network from: {net_file} ...")
    net = sumolib.net.readNet(str(net_file))

    # Collect traffic light nodes
    nodes = [n for n in net.getNodes() if n.getType() == "traffic_light"]
    tls_data = [{"id": n.getID(), "lon": net.convertXY2LonLat(*n.getCoord())[0], 
                 "lat": net.convertXY2LonLat(*n.getCoord())[1]} for n in nodes]

    print(f"Found {len(nodes)} traffic-light nodes.")
    print("Computing pairwise fastest-path travel times ...")

    pairwise, matrix = [], {n.getID(): {n.getID(): 0.0} for n in nodes}

    # Compute distances and build matrix
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            na, nb = nodes[i], nodes[j]
            ida, idb = na.getID(), nb.getID()
            
            times = [t for t in (road_travel_time(net, na, nb), road_travel_time(net, nb, na)) if t is not None]
            time_sec = round(min(times), 2) if times else None
            
            # Use travel_time_s instead of distance_m for clarity
            pairwise.append({"tls_a": ida, "tls_b": idb, "travel_time_s": time_sec})
            matrix[ida][idb] = matrix[idb][ida] = time_sec
            
            if len(pairwise) % 50 == 0:
                sys.stdout.write(f"\r  Progress: {len(pairwise)} pairs done")
                sys.stdout.flush()

    pairwise.sort(key=lambda p: (p["travel_time_s"] is None, p["travel_time_s"] or 0))
    reachable = [p["travel_time_s"] for p in pairwise if p["travel_time_s"] is not None]

    # Save results
    output = {
        "description": "Pairwise fastest-path road travel times (seconds) between traffic lights.",
        "statistics": {
            "num_traffic_lights": len(nodes),
            "num_pairs": len(pairwise),
            "num_reachable": len(reachable),
            "num_unreachable": len(pairwise) - len(reachable),
            "min_travel_time_s": min(reachable) if reachable else None,
            "max_travel_time_s": max(reachable) if reachable else None,
            "mean_travel_time_s": round(sum(reachable) / len(reachable), 2) if reachable else None,
        },
        "traffic_lights": tls_data,
        "travel_time_matrix": matrix,
        "pairwise_travel_times": pairwise,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! Results saved to: {out_json}")

if __name__ == "__main__":
    main()
