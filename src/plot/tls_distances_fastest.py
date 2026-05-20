import json
import sys
from pathlib import Path
import numpy as np
import sumolib
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import folium

project_root_dir = Path(__file__).resolve().parent.parent.parent
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))
import config

# Colours available in Folium markers
CLUSTER_COLOURS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "darkblue", "darkgreen", "cadetblue", "pink",
    "lightred", "lightblue", "lightgreen", "gray", "black",
]

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

def plot_cluster_map(threshold=config.CLUSTER_THRESHOLD_FASTEST) -> None:
    """
    Cluster traffic lights using Ward linkage at the given distance threshold
    (matching the dendrogram color_threshold) and plot them on a Folium map.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    dist_json = project_root / "src" / "outputs" / "tls_distances_fastest.json"
    out_map = project_root / "src" / "outputs" / "tls_clusters_fastest.html"

    print(f"Loading distance data from: {dist_json} ...")
    with open(dist_json) as f:
        data = json.load(f)

    tls_list = data["traffic_lights"]
    matrix = data["travel_time_matrix"]
    ids = [t["id"] for t in tls_list]
    n = len(ids)

    # Build symmetric NxN distance matrix
    valid_vals = [v for row in matrix.values() for v in row.values() if v is not None]
    penalty = max(valid_vals) * 1.5 if valid_vals else 1e6

    dist_array = np.zeros((n, n))
    for i, id_a in enumerate(ids):
        for j, id_b in enumerate(ids):
            val = matrix[id_a].get(id_b)
            dist_array[i, j] = val if val is not None else penalty

    dist_array = (dist_array + dist_array.T) / 2
    np.fill_diagonal(dist_array, 0)

    # Cluster using Ward linkage (same as dendrograms)
    condensed = squareform(dist_array)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=threshold, criterion="distance")

    id_to_cluster = {tid: int(label) for tid, label in zip(ids, labels)}
    num_clusters = len(set(labels))
    print(f"Formed {num_clusters} clusters (threshold = {threshold})\n")

    for c in sorted(set(labels)):
        members = [tid for tid, lbl in id_to_cluster.items() if lbl == c]
        colour = CLUSTER_COLOURS[(c - 1) % len(CLUSTER_COLOURS)]
        print(f"  Cluster {c} ({colour:>10}): {len(members)} lights  ->  {members}")

    # Generate Folium map
    center_lat = sum(t["lat"] for t in tls_list) / n
    center_lon = sum(t["lon"] for t in tls_list) / n
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    for tls in tls_list:
        cluster = id_to_cluster[tls["id"]]
        colour = CLUSTER_COLOURS[(cluster - 1) % len(CLUSTER_COLOURS)]
        folium.Marker(
            location=[tls["lat"], tls["lon"]],
            popup=f"TLS: {tls['id']}<br>Cluster: {cluster}",
            icon=folium.Icon(color=colour, icon="info-sign"),
        ).add_to(m)

    out_map.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_map))
    print(f"\nCluster map saved! Open in your browser:\n-> {out_map}")

if __name__ == "__main__":
    main()
    plot_cluster_map()

