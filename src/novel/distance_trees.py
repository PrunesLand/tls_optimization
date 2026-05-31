"""Distance-tree path resolution for linkage-based optimisers.

Maps each linkage-tree distance strategy (``config.TREE_STRATEGIES``) to its
generated ``tls_distances_<name>.json`` file. Kept out of ``config`` so the
latter holds plain settings only.
"""
from pathlib import Path

from config import TREE_STRATEGIES


def distance_tree_paths(out_dir):
    """Map each tree strategy to its distance JSON inside *out_dir*.

    Returns an insertion-ordered ``{name: Path}`` dict. *out_dir* may be a
    str or Path. Filenames follow the ``tls_distances_<name>.json`` convention.
    """
    out_dir = Path(out_dir)
    return {name: out_dir / f"tls_distances_{name}.json" for name in TREE_STRATEGIES}
