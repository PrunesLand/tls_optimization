import json
import math
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def analyze_linkage(distance_json):
    print(f"\n--- Analyzing: {Path(distance_json).name} ---")
    with open(distance_json) as f:
        data = json.load(f)

    key = 'distance_matrix' if 'distance_matrix' in data else 'travel_time_matrix'
    matrix = data[key]
    ids = [t['id'] for t in data['traffic_lights']]
    n = len(ids)

    vals = [v for row in matrix.values() for v in row.values() if v is not None]
    penalty = max(vals) * 1.5 if vals else 1e6

    arr = np.zeros((n, n))
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            arr[i, j] = matrix[a].get(b) if matrix[a].get(b) is not None else penalty
    arr = (arr + arr.T) / 2
    np.fill_diagonal(arr, 0)

    Z = linkage(squareform(arr), method='ward')
    
    # "check what is the maximum number of clusters that can be formed with all the IDs belonging to a cluster"
    # A cluster usually means size >= 2, because a cluster of size 1 is just an isolated ID.
    max_non_singleton_clusters = 0
    best_threshold = 0
    best_distribution = []
    valid_num_clusters = []

    # Z[:, 2] contains the distances at which merges happen.
    # We can use these as thresholds to see the state of clusters.
    thresholds_to_test = [0.0] + list(Z[:, 2]) + [Z[-1, 2] * 1.1]
    
    # Also find max clusters overall
    
    for t in sorted(list(set(thresholds_to_test))):
        labels = fcluster(Z, t=t, criterion='distance')
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        num_clusters = len(unique_labels)
        min_cluster_size = np.min(counts)
        
        if min_cluster_size >= 2:
            valid_num_clusters.append((num_clusters, t))
            if num_clusters > max_non_singleton_clusters:
                max_non_singleton_clusters = num_clusters
                best_threshold = t
                best_distribution = counts
                
    valid_counts = [item[0] for item in valid_num_clusters]
    median_valid_clusters = math.ceil(np.median(valid_counts)) if valid_counts else 0
    median_cluster_size = math.ceil(np.median(best_distribution)) if len(best_distribution) > 0 else 0
    
    if valid_num_clusters:
        closest_count = min(valid_counts, key=lambda x: abs(x - median_valid_clusters))
        median_thresholds = [item[1] for item in valid_num_clusters if item[0] == closest_count]
        if len(median_thresholds) > 1:
            thresholds_str = f"between ~{min(median_thresholds):.2f} and ~{max(median_thresholds):.2f}"
        else:
            thresholds_str = f"~{median_thresholds[0]:.2f}"
        actual_count_str = f" (actual clusters: {closest_count})" if closest_count != median_valid_clusters else ""
    else:
        thresholds_str = "N/A"
        actual_count_str = ""
                
    print(f"Total Traffic Lights (IDs): {n}")
    print(f"Maximum possible clusters (all singletons): {n}")
    print(f"Maximum clusters formed where ALL IDs belong to a cluster of size >= 2:")
    print(f"  -> {max_non_singleton_clusters} clusters (at distance threshold ~{best_threshold:.2f})")
    print(f"  -> Median cluster size at this threshold: {median_cluster_size}")
    print(f"  -> Median number of clusters across all valid non-singleton thresholds: {median_valid_clusters}")
    print(f"     -> Thresholds yielding this median{actual_count_str}: {thresholds_str}")

if __name__ == '__main__':
    import sys
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(root))
    from src.novel.distance_trees import distance_tree_paths
    files = list(distance_tree_paths(root / "src/outputs").values())
    for f in files:
        if f.exists():
            analyze_linkage(f)
        else:
            print(f"\nFile not found: {f}")
