from config import CLUSTER_THRESHOLD_FASTEST
from config import CLUSTER_THRESHOLD_SHORTEST
from config import CLUSTER_THRESHOLD_EUCLIDIAN
import json
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

def load_matrix(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    matrix_key = 'distance_matrix' if 'distance_matrix' in data else 'travel_time_matrix'
    matrix_dict = data[matrix_key]
    tls_list = data['traffic_lights']
    ids = [t['id'] for t in tls_list]
    n = len(ids)
    
    # Calculate a reasonable penalty for unreachable nodes instead of 1e6
    valid_vals = [v for row in matrix_dict.values() for v in row.values() if v is not None]
    penalty = max(valid_vals) * 1.5 if valid_vals else 1e6

    dist_array = np.zeros((n, n))
    for i, id_a in enumerate(ids):
        for j, id_b in enumerate(ids):
            val = matrix_dict[id_a].get(id_b)
            if val is None:
                val = penalty
            dist_array[i, j] = val
            
    # Ensure symmetry to avoid squareform errors due to floating point inaccuracies
    dist_array = (dist_array + dist_array.T) / 2
    np.fill_diagonal(dist_array, 0)
    
    condensed = squareform(dist_array)
    return ids, condensed

def plot_fastest_dendrogram(filepath, out_dir):
    print(f"Processing {filepath.name}...")
    ids, condensed = load_matrix(filepath)
    Z = linkage(condensed, method='ward')
    
    # --- Full Dendrogram ---
    plt.figure(figsize=(14, 8))
    dendrogram(Z, labels=ids, leaf_rotation=90, leaf_font_size=8)
    plt.title("Fastest Path (Travel Time) - Full")
    plt.xlabel("Traffic Light ID")
    plt.ylabel("Distance")
    plt.ylim(0, 350)
    plt.yticks(range(0, 351, 350))
    plt.tight_layout()
    out_png_full = out_dir / "linkage_dendrogram_fastest.png"
    plt.savefig(out_png_full, dpi=300)
    plt.close()

    # --- Threshold Dendrogram ---
    plt.figure(figsize=(14, 8))
    # Configure threshold and colors for fastest
    dendrogram(Z, labels=ids, leaf_rotation=75, leaf_font_size=8, color_threshold=CLUSTER_THRESHOLD_FASTEST, above_threshold_color='none')
    plt.title("Fastest Path (Travel Time) - Threshold")
    plt.xlabel("Traffic Light ID")
    plt.ylabel("Distance")
    # Configure y-axis scale, limits, and ticks for fastest
    max_y = CLUSTER_THRESHOLD_FASTEST * 1.1
    plt.ylim(0, max_y)
    ticks = np.linspace(0, max_y, 6)
    plt.yticks(ticks, [f"{t:.1f}" for t in ticks])
    plt.tight_layout()
    out_png_thresh = out_dir / "linkage_dendrogram_fastest_threshold.png"
    plt.savefig(out_png_thresh, dpi=300)
    plt.close()
    print(f"  Saved dendrograms for {filepath.stem}")

def plot_shortest_dendrogram(filepath, out_dir):
    print(f"Processing {filepath.name}...")
    ids, condensed = load_matrix(filepath)
    Z = linkage(condensed, method='ward')
    
    # --- Full Dendrogram ---
    plt.figure(figsize=(14, 8))
    dendrogram(Z, labels=ids, leaf_rotation=90, leaf_font_size=8)
    plt.title("Shortest Path (Metres) - Full")
    plt.xlabel("Traffic Light ID")
    plt.ylabel("Distance")
    plt.ylim(0, 4000)
    plt.yticks(range(0, 4001, 500))
    plt.tight_layout()
    out_png_full = out_dir / "linkage_dendrogram_shortest.png"
    plt.savefig(out_png_full, dpi=300)
    plt.close()

    # --- Threshold Dendrogram ---
    plt.figure(figsize=(14, 8))
    # Configure threshold and colors for shortest
    dendrogram(Z, labels=ids, leaf_rotation=75, leaf_font_size=8, color_threshold=CLUSTER_THRESHOLD_SHORTEST, above_threshold_color='none')
    plt.title("Shortest Path (Metres) - Threshold")
    plt.xlabel("Traffic Light ID")
    plt.ylabel("Distance")
    # Configure y-axis scale, limits, and ticks for shortest
    max_y = CLUSTER_THRESHOLD_SHORTEST * 1.2
    plt.ylim(0, max_y)
    ticks = np.linspace(0, max_y, 8)
    plt.yticks(ticks, [f"{t:.1f}" for t in ticks])
    plt.tight_layout()
    out_png_thresh = out_dir / "linkage_dendrogram_shortest_threshold.png"
    plt.savefig(out_png_thresh, dpi=300)
    plt.close()
    print(f"  Saved dendrograms for {filepath.stem}")

def plot_euclidean_dendrogram(filepath, out_dir):
    print(f"Processing {filepath.name}...")
    ids, condensed = load_matrix(filepath)
    Z = linkage(condensed, method='ward')
    
    # --- Full Dendrogram ---
    plt.figure(figsize=(14, 8))
    dendrogram(Z, labels=ids, leaf_rotation=90, leaf_font_size=8)
    plt.title("Euclidean Distance (Metres) - Full")
    plt.xlabel("Traffic Light ID")
    plt.ylabel("Distance")
    plt.ylim(0, 3500)
    plt.yticks(range(0, 3501, 500))
    plt.tight_layout()
    out_png_full = out_dir / "linkage_dendrogram_euclidian.png"
    plt.savefig(out_png_full, dpi=300)
    plt.close()

    # --- Threshold Dendrogram ---
    plt.figure(figsize=(14, 8))
    # Configure threshold and colors for euclidean
    dendrogram(Z, labels=ids, leaf_rotation=75, leaf_font_size=8, color_threshold=CLUSTER_THRESHOLD_EUCLIDIAN, above_threshold_color='none')
    plt.title("Euclidean Distance (Metres) - Threshold")
    plt.xlabel("Traffic Light ID")
    plt.ylabel("Distance")
    # Configure y-axis scale, limits, and ticks for euclidean
    max_y = CLUSTER_THRESHOLD_EUCLIDIAN * 1.2
    plt.ylim(0, max_y)
    ticks = np.linspace(0, max_y, 8)
    plt.yticks(ticks, [f"{t:.1f}" for t in ticks])
    plt.tight_layout()
    out_png_thresh = out_dir / "linkage_dendrogram_euclidian_threshold.png"
    plt.savefig(out_png_thresh, dpi=300)
    plt.close()
    print(f"  Saved dendrograms for {filepath.stem}")

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = project_root / "src" / "outputs"
    
    fastest_file = out_dir / "tls_distances_fastest.json"
    shortest_file = out_dir / "tls_distances_shortest.json"
    euclidean_file = out_dir / "tls_distances_euclidian.json"

    plot_fastest_dendrogram(fastest_file, out_dir)
    plot_shortest_dendrogram(shortest_file, out_dir)
    plot_euclidean_dendrogram(euclidean_file, out_dir)

if __name__ == "__main__":
    main()
