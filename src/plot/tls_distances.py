import json
import math
import random
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


# ---------- Distance functions ----------

def euclidean_distance(net, node_a, node_b) -> float:
    xa, ya = node_a.getCoord()
    xb, yb = node_b.getCoord()
    return math.hypot(xa - xb, ya - yb)


def _directed_cost(net, src, dst, pathfn) -> float | None:
    costs = [
        cost
        for e_from in src.getOutgoing()
        for e_to in dst.getIncoming()
        for route, cost in [pathfn(e_from, e_to, vClass="passenger")]
        if route is not None
    ]
    return min(costs) if costs else None


def shortest_road_distance(net, node_a, node_b) -> float | None:
    costs = [
        c for c in (
            _directed_cost(net, node_a, node_b, net.getShortestPath),
            _directed_cost(net, node_b, node_a, net.getShortestPath),
        ) if c is not None
    ]
    return min(costs) if costs else None


def fastest_road_travel_time(net, node_a, node_b) -> float | None:
    costs = [
        c for c in (
            _directed_cost(net, node_a, node_b, net.getFastestPath),
            _directed_cost(net, node_b, node_a, net.getFastestPath),
        ) if c is not None
    ]
    return min(costs) if costs else None


def random_distance(net, node_a, node_b) -> float:
    return random.random()


# ---------- Variant configuration ----------
# Each variant preserves the JSON schema of its original file so downstream
# consumers (src/novel/node_finder, dendrograms, optimizers) keep working unchanged.

VARIANTS = {
    "euclidian": {
        "compute": euclidean_distance,
        "round_to": 2,
        "out_json": "tls_distances_euclidian.json",
        "out_map":  "tls_clusters_euclidian.html",
        "description": "Pairwise Euclidean distances (metres) between traffic lights.",
        "value_key":   "distance_m",
        "matrix_key":  "distance_matrix",
        "pair_key":    "pairwise_distances",
        "stat_prefix": "distance_m",
        "threshold":   config.CLUSTER_THRESHOLD_EUCLIDIAN,
    },
    "shortest": {
        "compute": shortest_road_distance,
        "round_to": 2,
        "out_json": "tls_distances_shortest.json",
        "out_map":  "tls_clusters_shortest.html",
        "description": "Pairwise shortest-path road distances (metres) between traffic lights.",
        "value_key":   "distance_m",
        "matrix_key":  "distance_matrix",
        "pair_key":    "pairwise_distances",
        "stat_prefix": "distance_m",
        "threshold":   config.CLUSTER_THRESHOLD_SHORTEST,
    },
    "fastest": {
        "compute": fastest_road_travel_time,
        "round_to": 2,
        "out_json": "tls_distances_fastest.json",
        "out_map":  "tls_clusters_fastest.html",
        "description": "Pairwise fastest-path road travel times (seconds) between traffic lights.",
        "value_key":   "travel_time_s",
        "matrix_key":  "travel_time_matrix",
        "pair_key":    "pairwise_travel_times",
        "stat_prefix": "travel_time_s",
        "threshold":   config.CLUSTER_THRESHOLD_FASTEST,
    },
    "random": {
        "compute": random_distance,
        "round_to": 6,
        "out_json": "tls_distances_random.json",
        "out_map":  "tls_clusters_random.html",
        "description": "Pairwise random distances in [0, 1) between traffic lights (not derived from SUMO).",
        "value_key":   "distance_m",
        "matrix_key":  "distance_matrix",
        "pair_key":    "pairwise_distances",
        "stat_prefix": "distance_m",
        "threshold":   0.5,
    },
}


# ---------- Build / save ----------

def _empty_state(ids):
    return {
        "matrix": {a: {a: 0.0} for a in ids},
        "pairwise": [],
    }


def compute_all(net, nodes):
    """Compute all variant matrices in a single pass over node pairs."""
    ids = [n.getID() for n in nodes]
    state = {name: _empty_state(ids) for name in VARIANTS}
    total_pairs = len(nodes) * (len(nodes) - 1) // 2
    done = 0

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            na, nb = nodes[i], nodes[j]
            ida, idb = na.getID(), nb.getID()

            for name, cfg in VARIANTS.items():
                val = cfg["compute"](net, na, nb)
                if val is not None:
                    val = round(val, cfg["round_to"])
                state[name]["matrix"][ida][idb] = val
                state[name]["matrix"][idb][ida] = val
                state[name]["pairwise"].append(
                    {"tls_a": ida, "tls_b": idb, cfg["value_key"]: val}
                )

            done += 1
            if done % 50 == 0:
                sys.stdout.write(f"\r  Progress: {done}/{total_pairs} pairs done")
                sys.stdout.flush()

    print()
    return state


def save_variant(name, cfg, tls_data, var_state, out_dir):
    pairwise = var_state["pairwise"]
    matrix = var_state["matrix"]
    vkey = cfg["value_key"]
    stat = cfg["stat_prefix"]

    pairwise.sort(key=lambda p: (p[vkey] is None, p[vkey] or 0))
    reachable = [p[vkey] for p in pairwise if p[vkey] is not None]

    output = {
        "description": cfg["description"],
        "statistics": {
            "num_traffic_lights": len(tls_data),
            "num_pairs": len(pairwise),
            "num_reachable": len(reachable),
            "num_unreachable": len(pairwise) - len(reachable),
            f"min_{stat}": min(reachable) if reachable else None,
            f"max_{stat}": max(reachable) if reachable else None,
            f"mean_{stat}": round(sum(reachable) / len(reachable), 2) if reachable else None,
        },
        "traffic_lights": tls_data,
        cfg["matrix_key"]: matrix,
        cfg["pair_key"]: pairwise,
    }

    out_path = out_dir / cfg["out_json"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  [{name}] saved -> {out_path}")


def build_distance_matrices() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    net_file = project_root / "src" / "sumo_setup" / "osm.net.xml.gz"
    out_dir = project_root / "src" / "outputs"

    # Seed for the random variant so runs are reproducible.
    random.seed(config.SEED_BASE)

    print(f"Loading SUMO network from: {net_file} ...")
    net = sumolib.net.readNet(str(net_file))

    # Restrict to the same filtered TLS set as the generated baseline
    # (generation.generate_data drops phase counts outside OPTIMIZE_PHASE_COUNTS).
    # Using the baseline JSON as the single source of truth keeps this plot in
    # sync with every other downstream consumer.
    with open(config.BASELINE_TRAFFIC_DATA) as f:
        baseline_ids = set(json.load(f)["tls_data"].keys())

    nodes = [
        n for n in net.getNodes()
        if n.getType() == "traffic_light" and n.getID() in baseline_ids
    ]
    print(f"Baseline lists {len(baseline_ids)} TLS(s); "
          f"{len(nodes)} matched in the network.")
    tls_data = [
        {
            "id": n.getID(),
            "lon": net.convertXY2LonLat(*n.getCoord())[0],
            "lat": net.convertXY2LonLat(*n.getCoord())[1],
        }
        for n in nodes
    ]
    print(f"Found {len(nodes)} traffic-light nodes.")
    print(f"Computing pairwise values for variants: {', '.join(VARIANTS)} ...")

    state = compute_all(net, nodes)

    print("Saving variant JSON files ...")
    for name, cfg in VARIANTS.items():
        save_variant(name, cfg, tls_data, state[name], out_dir)


# ---------- Clustering / map ----------

def plot_cluster_map(variant: str, threshold: float | None = None) -> None:
    """Cluster TLS using Ward linkage at the given threshold and plot on a Folium map."""
    cfg = VARIANTS[variant]
    if threshold is None:
        threshold = cfg["threshold"]

    project_root = Path(__file__).resolve().parent.parent.parent
    dist_json = project_root / "src" / "outputs" / cfg["out_json"]
    out_map = project_root / "src" / "outputs" / cfg["out_map"]

    print(f"\n[{variant}] Loading distance data from: {dist_json} ...")
    with open(dist_json) as f:
        data = json.load(f)

    tls_list = data["traffic_lights"]
    matrix = data[cfg["matrix_key"]]
    ids = [t["id"] for t in tls_list]
    n = len(ids)

    valid_vals = [v for row in matrix.values() for v in row.values() if v is not None]
    penalty = max(valid_vals) * 1.5 if valid_vals else 1e6

    dist_array = np.zeros((n, n))
    for i, id_a in enumerate(ids):
        for j, id_b in enumerate(ids):
            val = matrix[id_a].get(id_b)
            dist_array[i, j] = val if val is not None else penalty

    dist_array = (dist_array + dist_array.T) / 2
    np.fill_diagonal(dist_array, 0)

    condensed = squareform(dist_array)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=threshold, criterion="distance")

    id_to_cluster = {tid: int(label) for tid, label in zip(ids, labels)}
    num_clusters = len(set(labels))
    print(f"[{variant}] Formed {num_clusters} clusters (threshold = {threshold})")

    for c in sorted(set(labels)):
        members = [tid for tid, lbl in id_to_cluster.items() if lbl == c]
        colour = CLUSTER_COLOURS[(c - 1) % len(CLUSTER_COLOURS)]
        print(f"  Cluster {c} ({colour:>10}): {len(members)} lights  ->  {members}")

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
    print(f"[{variant}] Cluster map saved -> {out_map}")


def plot_all_cluster_maps() -> None:
    for variant in VARIANTS:
        plot_cluster_map(variant)


if __name__ == "__main__":
    build_distance_matrices()
    plot_all_cluster_maps()
