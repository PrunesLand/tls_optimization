"""
v4 linkage-tree node picker for TLS clustering.

Subclasses the base `LinkageTree` from `src.algorithms.node_finder` and
adds `find_node_decomposition`, a *best-fit* tree walk that assembles a
TLS-ID set whose total size matches the requested target as closely as
possible.

Difference from v3
------------------
v3 flips a fair coin at every fork: it keeps one child 50/50 and recurses
into the sibling.  At a lopsided fork (e.g. children of size 1 and 35) the
coin can keep the giant child and overshoot the target massively (in the
`fastest` tree this reached +11 TLS over target).

v4 chooses by *fit* instead of by coin.  With `remainder` left to fill at
each node:

  1. Start at the smallest cluster whose size >= target (ties random).
  2. If the whole subtree, or one immediate child, equals the remainder
     -> take it and stop (exact).
  3. Else keep the LARGEST child whose size < remainder, subtract it, and
     recurse into the sibling to fill the rest.  (With probability
     `explore_prob`, keep the *smaller* fitting child instead, for
     composition diversity — the final total size is unaffected.)
  4. If NEITHER child fits (both > remainder), descend into the SMALLER
     child *without keeping anything*: it alone can still cover the
     remainder, so we never overshoot.

Because `current_size >= remainder` is preserved at every step, the walk
always terminates on an exact match (remainder -> 0).  It can never
overshoot, and it only under-fills if it dead-ends at a leaf — which the
invariant rules out for remainder > 1.

Usage:  python -m src.algorithms.node_finder_v4
"""

from pathlib import Path

import numpy as np

from src.algorithms.node_finder import LinkageTree as _BaseLinkageTree


class LinkageTree(_BaseLinkageTree):
    """LinkageTree with the v4 best-fit cluster-decomposition walk."""

    def find_node_decomposition(self, target_size, rng=None, explore_prob=0.0):
        """Return a flat list of TLS-IDs whose total size best-fits
        *target_size*, assembled by walking the Ward tree.

        Unlike the v3 walk, the total size cannot exceed *target_size*;
        the only shortfall path is a leaf dead-end, which the size
        invariant prevents for target_size > 1.
        """
        if rng is None:
            rng = np.random.default_rng()

        remainder = int(round(target_size))
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
                # Leaf — cannot decompose further (invariant makes this
                # unreachable while remainder > 1).
                break

            l_id, r_id = left.get_id(), right.get_id()
            l_size = self.node_sizes[l_id]
            r_size = self.node_sizes[r_id]

            # Exact-match child terminates cleanly (remainder -> 0).
            if l_size == remainder:
                selected_nids.append(l_id); selected_sizes.append(l_size)
                remainder = 0; break
            if r_size == remainder:
                selected_nids.append(r_id); selected_sizes.append(r_size)
                remainder = 0; break

            # Children that FIT inside the remainder (strictly smaller).
            # Each entry: (pick_id, pick_size, sibling_id).
            fits = []
            if l_size < remainder:
                fits.append((l_id, l_size, r_id))
            if r_size < remainder:
                fits.append((r_id, r_size, l_id))

            if fits:
                # Best-fit: keep the largest fitting child, then recurse
                # into its sibling for the rest.  Occasionally keep the
                # smaller fitting child instead (composition diversity);
                # this never changes the final total size.
                fits.sort(key=lambda t: t[1])              # ascending by size
                if len(fits) == 2 and rng.random() < explore_prob:
                    pick_id, pick_size, sibling_id = fits[0]   # smaller
                else:
                    pick_id, pick_size, sibling_id = fits[-1]  # largest fit
                selected_nids.append(pick_id)
                selected_sizes.append(pick_size)
                remainder -= pick_size
                current_nid = sibling_id
            else:
                # Neither child fits (both > remainder): descend into the
                # smaller child without keeping anything.  It alone still
                # covers the remainder, so we avoid overshooting.
                current_nid = l_id if l_size <= r_size else r_id

        members = []
        for nid in selected_nids:
            members.extend(self.node_members[nid])

        # Diagnostic: best-fit cannot overshoot, so any non-zero remainder
        # is a genuine leaf-dead-end under-fill worth surfacing.
        if abs(remainder) >= 1:
            print(
                f"[node_finder_v4] target={target_size:.3f} "
                f"selected_sizes={selected_sizes} remainder={remainder}"
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
    targets = [1, 2, 3, 5, 8, 18, 36]

    for name, path in trees.items():
        if not path.exists():
            print(f"[skip] {name}: {path} not found (run step 13 first)")
            continue

        tree = LinkageTree.from_distance_json(path)
        print(f"\n=== {name} ({len(tree.ids)} TLS) ===")
        for t in targets:
            members = tree.find_node_decomposition(t, rng=rng)
            # got should equal target exactly (best-fit never overshoots).
            print(f"  target={t:>5}  got={len(members):>3}  members={members}")


if __name__ == "__main__":
    _demo()
