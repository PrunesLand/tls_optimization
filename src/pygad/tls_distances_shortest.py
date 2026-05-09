"""
Compute pairwise shortest-path (road) distances between all traffic lights
in the SUMO network. Writes the results to a JSON file.
"""

import json
import sys
from pathlib import Path
import sumolib

def road_distance(net, node_a, node_b) -> float | None:
    """Return the shortest passenger road distance in metres between two nodes."""
    costs = [
        cost
        for e_from in node_a.getOutgoing()
        for e_to in node_b.getIncoming()
        for route, cost in [net.getShortestPath(e_from, e_to, vClass="passenger")]
        if route is not None
    ]
    return min(costs) if costs else None

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    net_file = project_root / "src" / "sumo_setup" / "osm.net.xml.gz"
    out_json = project_root / "src" / "outputs" / "tls_distances_shortest.json"

    print(f"Loading SUMO network from: {net_file} ...")
    net = sumolib.net.readNet(str(net_file))

    # Collect traffic light nodes
    nodes = [n for n in net.getNodes() if n.getType() == "traffic_light"]
    tls_data = [{"id": n.getID(), "lon": net.convertXY2LonLat(*n.getCoord())[0], 
                 "lat": net.convertXY2LonLat(*n.getCoord())[1]} for n in nodes]

    print(f"Found {len(nodes)} traffic-light nodes.")
    print("Computing pairwise shortest-path distances ...")

    pairwise, matrix = [], {n.getID(): {n.getID(): 0.0} for n in nodes}

    # Compute distances and build matrix
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            na, nb = nodes[i], nodes[j]
            ida, idb = na.getID(), nb.getID()
            
            dists = [d for d in (road_distance(net, na, nb), road_distance(net, nb, na)) if d is not None]
            dist = round(min(dists), 2) if dists else None
            
            pairwise.append({"tls_a": ida, "tls_b": idb, "distance_m": dist})
            matrix[ida][idb] = matrix[idb][ida] = dist
            
            if len(pairwise) % 50 == 0:
                sys.stdout.write(f"\r  Progress: {len(pairwise)} pairs done")
                sys.stdout.flush()

    pairwise.sort(key=lambda p: (p["distance_m"] is None, p["distance_m"] or 0))
    reachable = [p["distance_m"] for p in pairwise if p["distance_m"] is not None]

    # Save results
    output = {
        "description": "Pairwise shortest-path road distances (metres) between traffic lights.",
        "statistics": {
            "num_traffic_lights": len(nodes),
            "num_pairs": len(pairwise),
            "num_reachable": len(reachable),
            "num_unreachable": len(pairwise) - len(reachable),
            "min_distance_m": min(reachable) if reachable else None,
            "max_distance_m": max(reachable) if reachable else None,
            "mean_distance_m": round(sum(reachable) / len(reachable), 2) if reachable else None,
        },
        "traffic_lights": tls_data,
        "distance_matrix": matrix,
        "pairwise_distances": pairwise,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! Results saved to: {out_json}")

if __name__ == "__main__":
    main()
