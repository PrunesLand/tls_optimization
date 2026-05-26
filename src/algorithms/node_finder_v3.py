"""
v3 linkage-tree node picker for TLS clustering.

Subclasses the base `LinkageTree` from `src.algorithms.node_finder` and
adds `find_node_decomposition`, which assembles a TLS-ID set by walking
the Ward-linkage tree rather than returning a single nearest-size node.

Algorithm:
  1. Start at the smallest cluster whose size >= target_size
     ("closest and highest"). Ties broken uniformly at random.
  2. At each internal node: if a child's size matches the remainder
     exactly, pick it and terminate. Otherwise pick one of the two
     children 50/50, append it to the selection, subtract its size from
     the remainder, and recurse into the sibling subtree.
  3. Stop when remainder reaches 0, when a leaf is hit, or when a pick
     overshoots (remainder < 0).

Usage:  python -m src.algorithms.node_finder_v3
"""

from pathlib import Path

import numpy as np

from src.algorithms.node_finder import LinkageTree as _BaseLinkageTree


class LinkageTree(_BaseLinkageTree):
    """LinkageTree with the v3 cluster-decomposition walk."""

    def find_node_decomposition(self, target_size, rng=None):
        """Return a flat list of TLS-IDs assembled by walking the tree
        to decompose *target_size* into one or more clusters.
        """
        if rng is None:
            rng = np.random.default_rng()

        remainder = float(target_size)
        if remainder <= 0:
            return []

        selected_nids  = []
        selected_sizes = []

        node_ids = np.fromiter(self.node_sizes.keys(),   dtype=int)
        sizes    = np.fromiter(self.node_sizes.values(), dtype=int)

        # Step 1: starting cluster = smallest node whose size >= target.
        ge_mask = sizes >= remainder
        if ge_mask.any():
            ge_ids   = node_ids[ge_mask]
            ge_sizes = sizes[ge_mask]
            min_size = ge_sizes.min()
            ties     = ge_ids[ge_sizes == min_size]
        else:
            max_size = sizes.max()
            ties     = node_ids[sizes == max_size]
        current_nid = int(rng.choice(ties))

        while remainder > 0:
            current_size = self.node_sizes[current_nid]

            # Whole current subtree matches — take it and stop.
            if current_size == remainder:
                selected_nids.append(current_nid)
                selected_sizes.append(current_size)
                remainder = 0
                break

            node = self._nodes[current_nid]
            left, right = node.get_left(), node.get_right()
            if left is None or right is None:
                # Leaf — cannot decompose further.
                break

            l_id, r_id = left.get_id(), right.get_id()
            l_size = self.node_sizes[l_id]
            r_size = self.node_sizes[r_id]

            # Prefer an exact-match child if one exists (terminates cleanly).
            exact = None
            if l_size == remainder:
                exact = (l_id, l_size)
            elif r_size == remainder:
                exact = (r_id, r_size)
            if exact is not None:
                selected_nids.append(exact[0])
                selected_sizes.append(exact[1])
                remainder = 0
                break

            # Otherwise 50/50 between the two children.
            if rng.random() < 0.5:
                pick_id, pick_size, sibling_id = l_id, l_size, r_id
            else:
                pick_id, pick_size, sibling_id = r_id, r_size, l_id

            selected_nids.append(pick_id)
            selected_sizes.append(pick_size)
            remainder -= pick_size

            if remainder <= 0:
                break

            current_nid = sibling_id

        members = []
        for nid in selected_nids:
            members.extend(self.node_members[nid])

        # Diagnostic: only when we leave more than a fractional unit on
        # the table (i.e. real dead-end or overshoot, not float dust).
        if abs(remainder) >= 1:
            print(
                f"[node_finder_v3] target={target_size:.3f} "
                f"selected_sizes={selected_sizes} remainder={remainder:.3f}"
            )

        return members


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
    targets = [1, 2, 3, 5, 8, 18, 18.5, 36]

    for name, path in trees.items():
        if not path.exists():
            print(f"[skip] {name}: {path} not found (run step 13 first)")
            continue

        tree = LinkageTree.from_distance_json(path)
        print(f"\n=== {name} ({len(tree.ids)} TLS) ===")
        for t in targets:
            members = tree.find_node_decomposition(t, rng=rng)
            print(f"  target={t:>5}  got={len(members):>3}  members={members}")


if __name__ == "__main__":
    _demo()
