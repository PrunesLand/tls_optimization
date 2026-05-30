"""
Novel pair-wise (pair-cluster) mutation for traffic-light optimisation.

Isolated here so any optimiser can share the exact same operators:

    build_phase_split   — per-TLS green / red / yellow gene-index breakdown
                          consumed by the operators below.
    mutate_pair_cluster — grows the second TLS's green time within a pair.
    mutate_tree_walk    — structure-aware mutation that walks the Ward tree
                          and dispatches to the pair operator or per-TLS
                          re-sampling depending on cluster size.

Self-contained: depends only on numpy, the config constants, and ``phase_type``;
consumes a ``phase_split`` (from ``build_phase_split`` here) together with a
Ward ``tree_structure`` from ``src.novel.linkage_tree.build_all_tree_masks``.
"""
import sys
from pathlib import Path

import numpy as np

# Project root on sys.path so ``config`` / ``src...`` imports resolve when this
# module is loaded from anywhere.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import GENE_LOW, CYCLE_LENGTH, PHASE_BOUNDS
from src.sumo_setup.fitness_evaluation import phase_type


# ── Per-TLS phase-type split (metadata the operators consume) ────────────────

def build_phase_split(
    baseline_data: dict,
    tls_to_genes: dict[str, tuple[int, int]],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Per-TLS gene-index breakdown by phase type.

    Returns, for each TLS, a dict of gene indices::

        "green"   : green phases — grown by the pair operator
        "red"     : red phases   — pinned at their minimum by the pair operator
        "yellow"  : yellow phases — frozen, never written
        "mutable" : green + red  — everything operators may write

    Yellow phases are frozen, so they are excluded from ``mutable``.
    """
    split: dict[str, dict[str, np.ndarray]] = {}

    for tls_id, phases in baseline_data["tls_data"].items():
        if tls_id not in tls_to_genes:
            continue
        s, _ = tls_to_genes[tls_id]
        green: list[int] = []
        red:   list[int] = []
        yellow: list[int] = []
        for i, pk in enumerate(sorted(phases)):
            ptype = phase_type(phases[pk]["state"])
            idx   = s + i
            (green if ptype == "green" else red if ptype == "red" else yellow).append(idx)
        split[tls_id] = {
            "green":   np.array(green,        dtype=int),
            "red":     np.array(red,          dtype=int),
            "yellow":  np.array(yellow,       dtype=int),
            "mutable": np.array(green + red,  dtype=int),
        }

    return split


def mutate_pair_cluster(
    sol: np.ndarray,
    pair_clusters: list[tuple[str, str]],
    tls_to_genes: dict[str, tuple[int, int]],
    phase_split: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    ub: np.ndarray,
) -> np.ndarray:
    """
    Novel pair-cluster mutation — grows the second TLS's **green** time.

    Yellow phases are frozen and red phases are pinned at their minimum
    (``PHASE_BOUNDS["red"][0]``); only green phases are sampled.  The second
    TLS is given a green budget strictly larger than the first's, capped at
    the most green the cycle allows once the frozen yellows and a minimum red
    are reserved::

        max_green2 = CYCLE_LENGTH − Σ(yellow durations) − Σ(red minimums)

    Algorithm
    ---------
    1. Randomly select a 2-TLS cluster from *pair_clusters*.
    2. Randomly assign which TLS plays the "first" (smaller green) role and
       which plays the "second" (larger green) role.
    3. ``sum1`` = the first TLS's current **green** sum (read from ``sol``);
       the first TLS is otherwise left intact.
    4. Sample a green budget ``target2 ~ U(sum1 + 1, max_green2)``, distribute
       it across the second TLS's green phases (scaled, clipped to each
       green's dynamic ceiling ``ub``, with a deficit-redistribution loop),
       and pin its red phases at their minimum.  If ``sum1 + 1`` already
       reaches ``max_green2`` the second TLS is capped at its maximum green.
    5. ``normalize_to_cycle`` later restores the exact 90 s cycle.  Because
       pinned red (its minimum) is smaller than every green, the cycle
       remainder is absorbed into red, so the green budget set here survives
       normalisation unchanged.

    Parameters
    ----------
    sol           : current solution vector — returned as a new copy
    pair_clusters : list of (tls_id_a, tls_id_b) from the full Ward tree
    tls_to_genes  : maps tls_id → (start_idx, end_idx) in the gene vector
    phase_split   : maps tls_id → {"green"/"red"/"yellow"/"mutable": indices}
    rng           : numpy random Generator
    ub            : per-gene dynamic upper bounds (green/red ceilings)

    Returns
    -------
    new_sol : np.ndarray — mutated copy (original is not modified)
    """
    if not pair_clusters:
        return sol.copy()

    # Keep only pairs where both TLS IDs appear in the gene map
    valid = [(a, b) for a, b in pair_clusters
             if a in tls_to_genes and b in tls_to_genes]
    if not valid:
        return sol.copy()

    # ── Step 1: pick a random 2-TLS cluster ─────────────────────────────
    tls_first, tls_second = valid[int(rng.integers(0, len(valid)))]

    # ── Step 2: randomly flip the first/second role ──────────────────────
    if rng.random() < 0.5:
        tls_first, tls_second = tls_second, tls_first

    meta1 = phase_split.get(tls_first)
    meta2 = phase_split.get(tls_second)

    # The second TLS must have a green phase to grow.
    if meta2 is None or meta2["green"].size == 0:
        return sol.copy()

    green2, red2, yellow2 = meta2["green"], meta2["red"], meta2["yellow"]
    ng       = int(green2.size)
    ub_green = ub[green2]            # per-green dynamic ceilings

    # ── Step 3: first TLS's current green sum (the floor to beat) ───────
    green1 = meta1["green"] if meta1 is not None else np.empty(0, dtype=int)
    sum1   = float(sol[green1].sum()) if green1.size else 0.0

    # ── Step 4: green budget for the second TLS ─────────────────────────
    #   max_green2 = cycle − frozen yellows − reserved minimum red.
    red_min     = float(PHASE_BOUNDS["red"][0])
    yellow_sum2 = float(sol[yellow2].sum()) if yellow2.size else 0.0
    max_sum2    = float(CYCLE_LENGTH) - yellow_sum2 - red_min * red2.size
    min_sum2    = sum1 + 1.0          # strictly greener than the first TLS

    if min_sum2 < max_sum2:
        # Pick a green budget above sum1 and at most max_sum2
        target2 = float(rng.uniform(min_sum2, max_sum2))

        # Draw random proportions and scale to the green budget
        raw    = rng.uniform(GENE_LOW, ub_green, ng)
        greens = raw / raw.sum() * target2
        greens = np.clip(greens, GENE_LOW, ub_green)

        # ── Hard enforcement: GENE_LOW clipping can shrink the sum below the
        #   floor.  Raise greens with headroom until the floor is met or every
        #   green sits at its ceiling.
        deficit = min_sum2 - float(greens.sum())
        while deficit > 0:
            headroom = ub_green - greens
            total_headroom = float(headroom.sum())

            if total_headroom <= 0:
                break

            boost   = np.minimum(headroom, headroom / total_headroom * deficit)
            greens += boost
            greens  = np.clip(greens, GENE_LOW, ub_green)
            deficit = min_sum2 - float(greens.sum())

    else:
        # Edge case: the first TLS already saturates the green budget — cap
        # the second at its maximum green, split evenly across green phases.
        greens = np.full(ng, max_sum2 / ng)

    new_sol = sol.copy()
    new_sol[green2] = greens             # grow green
    if red2.size:
        new_sol[red2] = red_min          # pin red at its minimum
    return new_sol


# ── Tree-walk mutation ───────────────────────────────────────────────────────

def _find_containing_node(
    tls_id: str,
    tree_structure: dict,
) -> int | None:
    """
    Find the smallest internal node (cluster) that contains *tls_id*.

    Walks upward from the leaf to find the first internal node (size ≥ 2)
    that contains this TLS.  Returns the node_id, or None if the TLS is
    only found as a leaf (i.e. it was never merged into any cluster, which
    shouldn't happen in a complete Ward tree but is handled defensively).
    """
    tls_to_node = tree_structure.get("__tls_to_node__", {})
    if tls_id not in tls_to_node:
        return None

    root_id = tree_structure["__root__"]
    leaf_id = tls_to_node[tls_id]

    # Walk every internal node and find the smallest one containing this leaf.
    # Internal nodes have size ≥ 2.
    best_node = None
    best_size = float("inf")

    for node_id, info in tree_structure.items():
        if isinstance(node_id, str):           # skip metadata keys
            continue
        if info["size"] < 2:                   # skip leaves
            continue
        if tls_id in info["members"] and info["size"] < best_size:
            best_node = node_id
            best_size = info["size"]

    return best_node


def _collect_tree_walk_genes(
    node_id: int,
    tree_structure: dict,
    tls_to_genes: dict[str, tuple[int, int]],
    rng: np.random.Generator,
) -> list[str]:
    """
    Recursively walk a sub-tree rooted at *node_id* and collect TLS IDs
    that should be mutated, following the tree-walk mutation rules.

    Rules applied at each internal node:
    - If the node is a pair cluster (size == 2): skip it entirely (return []).
    - If the node has size > 2: examine its two children.
      - A child that is a leaf (single TLS): mutate it.
      - A child that is a pair cluster (size == 2): skip it (paired mutation
        territory).
      - A child that is a larger cluster (size > 2): recurse into it.

    Returns
    -------
    tls_ids_to_mutate : list[str]  — TLS IDs whose genes should be randomised
    """
    node = tree_structure.get(node_id)
    if node is None:
        return []

    # Leaf node — this TLS is an individual gene, collect it for mutation
    if node["size"] == 1:
        tls_id = node["members"][0]
        if tls_id in tls_to_genes:
            return [tls_id]
        return []

    # Pair cluster (size == 2) — skip entirely
    if node["size"] == 2:
        return []

    # Cluster with size > 2 — descend into children
    to_mutate: list[str] = []
    for child_id in (node["left"], node["right"]):
        if child_id is None:
            continue
        child = tree_structure.get(child_id)
        if child is None:
            continue

        if child["size"] == 1:
            # Individual gene — mutate it
            tls_id = child["members"][0]
            if tls_id in tls_to_genes:
                to_mutate.append(tls_id)
        elif child["size"] == 2:
            # Pair sub-cluster — skip (leave for paired mutation)
            continue
        else:
            # Larger sub-cluster — recurse
            to_mutate.extend(
                _collect_tree_walk_genes(child_id, tree_structure,
                                        tls_to_genes, rng)
            )

    return to_mutate


def mutate_tree_walk(
    sol: np.ndarray,
    tree_structure: dict,
    tls_to_genes: dict[str, tuple[int, int]],
    pair_clusters: list[tuple[str, str]],
    phase_split: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    ub: np.ndarray,
) -> np.ndarray:
    """
    Tree-walk mutation — structure-aware single-gene mutation.

    Algorithm
    ---------
    1. Randomly select a TLS gene from the gene map.
    2. Find the smallest cluster in the Ward tree that contains this TLS.
    3. Apply the tree-walk rules:
       a. If that cluster is a pair (size 2): apply **paired mutation** on
          that specific pair via ``mutate_pair_cluster``.
       b. If the TLS is not part of any cluster (standalone leaf): mutate
          just that TLS.
       c. If the cluster has size > 2: enter it and recursively collect
          individual TLS genes to mutate.  The initially selected TLS is
          always included if it is a direct leaf child.  Pair sub-clusters
          encountered during traversal are skipped.
    4. Each collected TLS has its phase-duration genes re-sampled uniformly
       from [GENE_LOW, ub] (per-gene dynamic ceiling).

    Example
    -------
    Cluster structure:  (2, [5, [3, 4]])
    - Enter root cluster (size 3): children are leaf 2 and sub-cluster [5,[3,4]].
    - Leaf 2 → mutate.
    - Sub-cluster [5,[3,4]] (size 3): children are leaf 5 and pair [3,4].
      - Leaf 5 → mutate.
      - Pair [3,4] → skip.
    - Result: genes for TLS 2 and TLS 5 are re-sampled.

    If gene 3 had been selected initially instead, its smallest cluster is
    the pair [3,4], so paired mutation is applied to (3,4).

    Parameters
    ----------
    sol            : current solution vector (not modified)
    tree_structure : full Ward tree dict from build_all_tree_masks
    tls_to_genes   : maps tls_id → (start_idx, end_idx) in the gene vector
    pair_clusters  : list of (tls_a, tls_b) pairs for paired mutation
    rng            : numpy random Generator

    Returns
    -------
    new_sol : np.ndarray — mutated copy
    """
    all_tls = [t for t in tls_to_genes if t in
               tree_structure.get("__tls_to_node__", {})]
    if not all_tls:
        return sol.copy()

    # ── Step 1: randomly select a TLS gene ───────────────────────────────
    selected_tls = all_tls[int(rng.integers(0, len(all_tls)))]

    # ── Step 2: find the smallest containing cluster ─────────────────────
    containing_node = _find_containing_node(selected_tls, tree_structure)

    new_sol = sol.copy()

    if containing_node is None:
        # TLS is not part of any cluster — mutate it directly
        meta    = phase_split.get(selected_tls)
        mutable = meta["mutable"] if meta is not None else np.empty(0, dtype=int)
        if mutable.size:
            new_sol[mutable] = rng.uniform(GENE_LOW, ub[mutable], mutable.size)
        return new_sol

    containing_info = tree_structure[containing_node]

    # ── Step 3a: smallest cluster is a pair → apply paired mutation ──────
    if containing_info["size"] == 2:
        pair = (containing_info["members"][0], containing_info["members"][1])
        return mutate_pair_cluster(
            sol, [pair], tls_to_genes,
            phase_split, rng, ub,
        )

    # ── Step 3c: cluster size > 2 → tree-walk to collect genes ───────────
    tls_to_mutate = _collect_tree_walk_genes(
        containing_node, tree_structure, tls_to_genes, rng
    )

    # ── Step 4: mutate collected genes (green/all-red only) ──────────────
    for tls_id in tls_to_mutate:
        meta    = phase_split.get(tls_id)
        mutable = meta["mutable"] if meta is not None else np.empty(0, dtype=int)
        if mutable.size:
            new_sol[mutable] = rng.uniform(GENE_LOW, ub[mutable], mutable.size)

    return new_sol
