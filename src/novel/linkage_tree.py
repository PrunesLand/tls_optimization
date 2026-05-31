"""
Ward linkage-tree construction for TLS clustering.

Single home for the linkage-tree machinery shared across the novel operators
(optimal mixing and pair-cluster / tree-walk mutation):

    build_all_tree_masks — distance JSON → (mixing_masks, pair_clusters,
                           tree_structure).  Anything that needs the Ward tree
                           — the optimal-mixing masks, the pair clusters, or
                           the full hierarchy for tree-walk mutation — gets it
                           from here.
"""
import json

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


# ── Distance matrix helpers ──────────────────────────────────────────────────

def _load_distance_array(distance_json: str):
    """Return (symmetric_np_array, ordered_tls_id_list)."""
    with open(distance_json) as f:
        data = json.load(f)

    key    = "distance_matrix" if "distance_matrix" in data else "travel_time_matrix"
    matrix = data[key]
    tls_ids = [t["id"] for t in data["traffic_lights"]]
    n       = len(tls_ids)

    vals    = [v for row in matrix.values() for v in row.values() if v is not None]
    penalty = max(vals) * 1.5 if vals else 1e6

    arr = np.zeros((n, n))
    for i, a in enumerate(tls_ids):
        for j, b in enumerate(tls_ids):
            v = matrix[a].get(b)
            arr[i, j] = v if v is not None else penalty
    arr = (arr + arr.T) / 2
    np.fill_diagonal(arr, 0)
    return arr, tls_ids


# ── Linkage tree → masks ─────────────────────────────────────────────────────

def build_all_tree_masks(
    distance_json: str,
    threshold: float | None = None,
) -> tuple[list[list[str]], list[tuple[str, str]], dict]:
    """
    Build a Ward linkage tree and return three collections.

    mixing_masks
        Every internal node whose merge distance ≤ *threshold*, excluding the
        root node and singletons.  Both parent clusters AND their children are
        collected, giving the full sub-threshold subtree as mixing candidates.
        Only built when *threshold* is given; when it is ``None`` this list is
        empty, for callers that need only ``pair_clusters`` / ``tree_structure``
        (e.g. the SHADE cluster-crossover, which never mixes).

    pair_clusters
        Every internal node that groups exactly two TLS IDs, collected from
        the *entire* tree with no threshold restriction.  These feed the
        pair-cluster mutation regardless of where they sit in the hierarchy.

    tree_structure
        Full hierarchical representation of the Ward linkage tree.  Each
        internal node is stored as::

            node_id: {
                "members":  [tls_id, ...],   # all leaf TLS IDs under this node
                "left":     int,              # left child node id
                "right":    int,              # right child node id
                "size":     int,              # number of leaves
            }

        Leaf nodes (id < n) have a single-entry ``members`` list and no
        children.  This structure enables the tree-walk mutation to
        descend the hierarchy.

    Parameters
    ----------
    distance_json : path to a distance/travel-time JSON file
    threshold     : Ward merge-distance cut-off (same value used for dendrograms);
                    omit (``None``) to skip building mixing_masks entirely

    Returns
    -------
    mixing_masks   : list[list[str]]         — TLS-ID groups for mixing
    pair_clusters  : list[tuple[str, str]]   — 2-TLS pairs for mutation
    tree_structure : dict                    — full Ward tree (for tree-walk mutation)
    """
    arr, tls_ids = _load_distance_array(distance_json)
    n = len(tls_ids)
    Z = linkage(squareform(arr), method="ward")

    # Build leaf-index membership for every node bottom-up
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    for i, row in enumerate(Z):
        left, right    = int(row[0]), int(row[1])
        members[n + i] = members[left] + members[right]

    root_id = n + len(Z) - 1          # last merge = root
    mixing_masks:  list[list[str]]        = []
    pair_clusters: list[tuple[str, str]]  = []

    # ── Build full tree structure for tree-walk mutation ──────────────────
    tree_structure: dict = {}

    # Register every leaf node
    for i in range(n):
        tree_structure[i] = {
            "members": [tls_ids[i]],
            "left":    None,
            "right":   None,
            "size":    1,
        }

    for i, row in enumerate(Z):
        node_id    = n + i
        left, right = int(row[0]), int(row[1])
        merge_dist = float(row[2])
        tls_group  = [tls_ids[m] for m in members[node_id]]

        # Register this internal node in the tree structure
        tree_structure[node_id] = {
            "members": tls_group,
            "left":    left,
            "right":   right,
            "size":    len(tls_group),
        }

        # ── Pair clusters: collect from whole tree (no threshold gate) ──
        if len(tls_group) == 2:
            pair_clusters.append((tls_group[0], tls_group[1]))

        # ── Mixing masks: sub-threshold, not root, not singleton ────────
        # Skipped entirely when no threshold is supplied.
        if threshold is None:
            continue
        if node_id == root_id:
            continue                  # root always excluded
        if merge_dist > threshold:
            continue                  # above cut — skip
        if len(tls_group) < 2:
            continue                  # singleton — skip (single gene forbidden)

        mixing_masks.append(tls_group)

    # Store root_id and tls_id → leaf_node_id mapping for lookups
    tls_to_node: dict[str, int] = {tls_ids[i]: i for i in range(n)}
    tree_structure["__root__"]      = root_id
    tree_structure["__tls_to_node__"] = tls_to_node

    return mixing_masks, pair_clusters, tree_structure
