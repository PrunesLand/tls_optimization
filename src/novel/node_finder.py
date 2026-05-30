"""
Linkage-tree node picker for TLS clustering.

Builds a Ward-linkage hierarchical-clustering tree from one of the
precomputed pairwise distance / travel-time JSONs produced by
`src/plot/tls_distances_{shortest,euclidian,fastest}.py` and exposes:

    LinkageTree.find_node_closest_to_size(target_size, rng=None)

which returns the TLS-ID members of a node whose size is closest to
*target_size*.  Leaves are treated as nodes of size 1, so size-1
lookups always succeed exactly.  Ties on absolute size-difference are
broken uniformly at random.

Usage:  python -m src.novel.node_finder
"""

import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform


# The plot scripts emit two different matrix keys depending on the metric.
_MATRIX_KEYS = ("distance_matrix", "travel_time_matrix")


class LinkageTree:
    """Ward-linkage tree over the TLS pairwise distance matrix.

    Each scipy ClusterNode has an integer id: leaves are 0..n-1 and
    internal nodes are n..2n-2.  Sizes and leaf membership are
    precomputed for every node, so closest-size lookup is one NumPy
    reduction.
    """

    def __init__(self, ids, linkage_matrix):
        self.ids = list(ids)
        self.Z = linkage_matrix
        self._root, self._nodes = to_tree(linkage_matrix, rd=True)

        self.node_sizes = {}
        self.node_members = {}
        for node in self._nodes:
            nid = node.get_id()
            leaf_indices = node.pre_order()  # default: leaf .id values
            self.node_members[nid] = [self.ids[i] for i in leaf_indices]
            self.node_sizes[nid] = len(leaf_indices)

    @classmethod
    def from_distance_json(cls, json_path):
        """Build from one of the `tls_distances_*.json` files."""
        with open(json_path) as f:
            data = json.load(f)

        tls_list = data["traffic_lights"]
        ids = [t["id"] for t in tls_list]
        n = len(ids)

        matrix_key = next((k for k in _MATRIX_KEYS if k in data), None)
        if matrix_key is None:
            raise KeyError(
                f"{json_path}: expected one of {_MATRIX_KEYS} in JSON payload"
            )
        matrix = data[matrix_key]

        # Unreachable pairs get a penalty (same recipe the plot scripts use).
        valid_vals = [v for row in matrix.values() for v in row.values()
                      if v is not None]
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
        return cls(ids, Z)

    def find_node_closest_to_size(self, target_size, rng=None):
        """Return the TLS-ID members of a node whose size is closest to
        *target_size*.  *target_size* may be a float; the node with the
        smallest |size - target_size| wins.  Ties broken uniformly at random.
        """
        if rng is None:
            rng = np.random.default_rng()

        node_ids = np.fromiter(self.node_sizes.keys(),   dtype=int)
        sizes    = np.fromiter(self.node_sizes.values(), dtype=int)

        diffs = np.abs(sizes - target_size)
        candidates = node_ids[diffs == diffs.min()]
        chosen = int(rng.choice(candidates))
        return self.node_members[chosen]


# ── Demo ───────────────────────────────────────────────────────────────

def _demo():
    project_root = Path(__file__).resolve().parent.parent.parent
    outputs = project_root / "src" / "outputs"

    trees = {
        "shortest":  outputs / "tls_distances_shortest.json",
        "euclidian": outputs / "tls_distances_euclidian.json",
        "fastest":   outputs / "tls_distances_fastest.json",
    }

    rng = np.random.default_rng(42)
    targets = [1, 2, 3, 5, 8, 20, 100]

    for name, path in trees.items():
        if not path.exists():
            print(f"[skip] {name}: {path} not found (run step 13 first)")
            continue

        tree = LinkageTree.from_distance_json(path)
        sizes_present = sorted(set(tree.node_sizes.values()))
        preview = sizes_present[:8]
        more = "..." if len(sizes_present) > 8 else ""
        print(f"\n=== {name} ({len(tree.ids)} TLS, sizes present: "
              f"{preview}{more}) ===")

        for t in targets:
            members = tree.find_node_closest_to_size(t, rng=rng)
            print(f"  target={t:>4} → got size={len(members):>3}  members={members}")


if __name__ == "__main__":
    _demo()
