"""
Cluster traffic lights by road distance and plot them on a Folium map.

Reads the pairwise distance data from tls_distances.json, performs
agglomerative clustering using the mean distance as the cut-off threshold,
and generates an interactive HTML map where each cluster is shown in a
distinct colour.
"""

import json
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# Colours available in Folium markers (enough for typical cluster counts)
CLUSTER_COLOURS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "darkblue", "darkgreen", "cadetblue", "pink",
    "lightred", "lightblue", "lightgreen", "gray", "black",
]


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    dist_json = project_root / "src" / "outputs" / "tls_distances.json"
    out_map = project_root / "src" / "outputs" / "tls_clusters_map.html"

    # --- load distance data ---
    print(f"Loading distance data from: {dist_json} ...")
    with open(dist_json) as f:
        data = json.load(f)

    tls_list = data["traffic_lights"]
    matrix = data["distance_matrix"]
    mean_dist = data["statistics"]["mean_distance_m"]
    ids = [t["id"] for t in tls_list]
    n = len(ids)

    print(f"Loaded {n} traffic lights. Mean road distance: {mean_dist:.2f} m")

    # --- build a symmetric NxN numpy distance matrix ---
    dist_array = np.zeros((n, n))
    for i, id_a in enumerate(ids):
        for j, id_b in enumerate(ids):
            val = matrix[id_a].get(id_b)
            dist_array[i, j] = val if val is not None else 1e9  # large fallback

    # Make perfectly symmetric (average of both directions)
    dist_array = (dist_array + dist_array.T) / 2
    np.fill_diagonal(dist_array, 0)

    # --- agglomerative clustering with mean distance as threshold ---
    condensed = squareform(dist_array)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=mean_dist, criterion="distance")

    # --- map TLS id -> cluster label ---
    id_to_cluster = {tid: int(label) for tid, label in zip(ids, labels)}
    num_clusters = len(set(labels))
    print(f"Formed {num_clusters} clusters (threshold = mean distance {mean_dist:.2f} m)\n")

    for c in sorted(set(labels)):
        members = [tid for tid, lbl in id_to_cluster.items() if lbl == c]
        print(f"  Cluster {c} ({CLUSTER_COLOURS[(c - 1) % len(CLUSTER_COLOURS)]:>10}): "
              f"{len(members)} lights  ->  {members}")

    # --- generate Folium map ---
    if not HAS_FOLIUM:
        print("\nInstall folium to generate the map:  pip install folium")
        return

    print("\nGenerating interactive cluster map ...")

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
    print(f"Map saved! Open in your browser:\n-> {out_map}")


if __name__ == "__main__":
    main()
