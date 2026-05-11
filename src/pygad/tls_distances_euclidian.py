import json
import math
import sys
from pathlib import Path
import numpy as np
import sumolib
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import folium

# Colours available in Folium markers
CLUSTER_COLOURS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "darkblue", "darkgreen", "cadetblue", "pink",
    "lightred", "lightblue", "lightgreen", "gray", "black",
]

def euclidean_distance(node_a, node_b) -> float:
    """Return the Euclidean distance in metres between two nodes based on their SUMO x,y coordinates."""
    xa, ya = node_a.getCoord()
    xb, yb = node_b.getCoord()
    return math.hypot(xa - xb, ya - yb)

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    net_file = project_root / "src" / "sumo_setup" / "osm.net.xml.gz"
    out_json = project_root / "src" / "outputs" / "tls_distances_euclidian.json"

    print(f"Loading SUMO network from: {net_file} ...")
    net = sumolib.net.readNet(str(net_file))

    # Collect traffic light nodes
    nodes = [n for n in net.getNodes() if n.getType() == "traffic_light"]
    tls_data = [{"id": n.getID(), "lon": net.convertXY2LonLat(*n.getCoord())[0], 
                 "lat": net.convertXY2LonLat(*n.getCoord())[1]} for n in nodes]

    print(f"Found {len(nodes)} traffic-light nodes.")
    print("Computing pairwise Euclidean distances ...")

    pairwise, matrix = [], {n.getID(): {n.getID(): 0.0} for n in nodes}

    # Compute distances and build matrix
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            na, nb = nodes[i], nodes[j]
            ida, idb = na.getID(), nb.getID()
            
            dist = round(euclidean_distance(na, nb), 2)
            
            pairwise.append({"tls_a": ida, "tls_b": idb, "distance_m": dist})
            matrix[ida][idb] = matrix[idb][ida] = dist
            
            if len(pairwise) % 50 == 0:
                sys.stdout.write(f"\r  Progress: {len(pairwise)} pairs done")
                sys.stdout.flush()

    pairwise.sort(key=lambda p: (p["distance_m"] is None, p["distance_m"] or 0))
    reachable = [p["distance_m"] for p in pairwise if p["distance_m"] is not None]

    # Save results
    output = {
        "description": "Pairwise Euclidean distances (metres) between traffic lights.",
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

def plot_cluster_map(threshold=3500) -> None:
    """
    Cluster traffic lights using Ward linkage at the given distance threshold
    (matching the dendrogram color_threshold) and plot them on a Folium map.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    dist_json = project_root / "src" / "outputs" / "tls_distances_euclidian.json"
    out_map = project_root / "src" / "outputs" / "tls_clusters_euclidian.html"

    print(f"Loading distance data from: {dist_json} ...")
    with open(dist_json) as f:
        data = json.load(f)

    tls_list = data["traffic_lights"]
    matrix = data["distance_matrix"]
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

