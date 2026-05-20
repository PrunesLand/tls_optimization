"""
Custom Optimier — Linkage Tree Optimal Mixing for traffic light optimization.

1. Optimal-mixing masks now include EVERY internal Ward node whose merge
   distance ≤ threshold (not just the top-level fcluster partition).
   The root node is always excluded.  Example: if the tree contains

       root  (1,2,3,4)   ← always excluded
       node  (1,2,3)     ← included when merge_dist ≤ threshold
       node  (1,2)       ← included
       node  (3,4)       ← included

   All three sub-root nodes become candidate masks, so parent and
   children compete equally during optimal mixing.  Single-gene masks
   are rejected (clusters only).

2. Novel pair-cluster mutation selects a random 2-TLS cluster from the
   FULL Ward tree (no threshold filter), randomly assigns "first" and
   "second" roles, samples the first TLS's raw gene values freely, then
   samples the second TLS's raw gene values so their sum strictly exceeds
   the first's — enforcing the assumption that the second TLS has more
   available duration in gene-space.  _rebuild_json normalises both TLS
   to exactly 90 s, preserving the new phase ratios.

3. Tree-walk mutation traverses the Ward linkage tree starting from a
   randomly selected TLS gene:
   - If the gene is in a pair cluster (size 2), apply paired mutation.
   - If the gene is standalone (leaf, not inside any cluster), mutate it.
   - If the gene is in a cluster of size > 2, enter the cluster and
     recursively mutate individual genes while skipping pair sub-clusters.
   Example: cluster (2, [5, [3,4]]) → mutate 2, enter sub-cluster,
   mutate 5, enter [3,4] which is a pair → skip both.
   But if gene 3 was selected, its smallest cluster is [3,4] (pair) →
   apply paired mutation on (3,4).

9 experiments: 3 linkage trees × 3 population strategies.

Usage:  python -m src.pygad.custom_optimizer
"""
from config import (
    CLUSTER_THRESHOLD_FASTEST,
    CLUSTER_THRESHOLD_SHORTEST,
    CLUSTER_THRESHOLD_EUCLIDIAN,
    MAX_EVALS, BASELINE_TRAFFIC_DATA, NUM_PROCESSORS,
    LT_GOMEA_POPULATION_SIZE, LT_GOMEA_NUM_GENERATIONS,
    LT_GOMEA_BASELINE_NOISE_STD, LT_GOMEA_USE_MUTATION,
    GENE_LOW, GENE_HIGH, MUTATION_RATE,
)
import json, copy, time, os, sys
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.sumo_setup.fitness_evaluation import (
    fitness_function as _traffic_fitness,
    build_traffic_fitness_wrapper,
)

THRESHOLDS = {
    "shortest":  CLUSTER_THRESHOLD_SHORTEST,
    "euclidian": CLUSTER_THRESHOLD_EUCLIDIAN,
    "fastest":   CLUSTER_THRESHOLD_FASTEST,
}  # fraction of individuals mutated per generation


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
    threshold: float,
) -> tuple[list[list[str]], list[tuple[str, str]], dict]:
    """
    Build a Ward linkage tree and return three collections.

    mixing_masks
        Every internal node whose merge distance ≤ *threshold*, excluding the
        root node and singletons.  Both parent clusters AND their children are
        collected, giving the full sub-threshold subtree as mixing candidates.

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
    threshold     : Ward merge-distance cut-off (same value used for dendrograms)

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


# ── Gene mapping ─────────────────────────────────────────────────────────────

def build_gene_map(baseline_data: dict):
    """
    Returns
    -------
    tls_to_genes : {tls_id: (start, end)} in the flat gene vector
    num_genes    : total gene count
    baseline_vec : np.ndarray of baseline phase durations
    """
    tls_to_genes: dict[str, tuple[int, int]] = {}
    idx      = 0
    baseline: list[float] = []

    for tls_id in sorted(baseline_data["tls_data"]):
        phases = sorted(baseline_data["tls_data"][tls_id])
        tls_to_genes[tls_id] = (idx, idx + len(phases))
        for pk in phases:
            baseline.append(float(baseline_data["tls_data"][tls_id][pk]["duration"]))
        idx += len(phases)

    return tls_to_genes, idx, np.array(baseline)


def mask_to_gene_indices(
    tls_mask: list[str],
    tls_to_genes: dict[str, tuple[int, int]],
) -> list[int]:
    """Flatten a list of TLS IDs into a flat list of gene indices."""
    out: list[int] = []
    for tls_id in tls_mask:
        if tls_id in tls_to_genes:
            s, e = tls_to_genes[tls_id]
            out.extend(range(s, e))
    return out


# ── Population init ──────────────────────────────────────────────────────────

def init_population(
    strategy: str,
    n: int,
    num_genes: int,
    baseline_vec: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create initial population via 'random', 'baseline', or 'mixed' strategy."""
    if strategy == "random":
        return rng.uniform(GENE_LOW, GENE_HIGH, (n, num_genes))

    elif strategy == "baseline":
        pop  = np.tile(baseline_vec, (n, 1))
        pop += rng.normal(0, noise_std, pop.shape) * pop
        return np.clip(pop, GENE_LOW, GENE_HIGH)

    elif strategy == "mixed":
        half = n // 2
        rand = rng.uniform(GENE_LOW, GENE_HIGH, (half, num_genes))
        base = np.tile(baseline_vec, (n - half, 1))
        base += rng.normal(0, noise_std, base.shape) * base
        return np.vstack([rand, np.clip(base, GENE_LOW, GENE_HIGH)])

    raise ValueError(f"Unknown strategy: {strategy}")


# ── Fitness helpers (parallel) ───────────────────────────────────────────────

def _eval(args):
    wrapper, sol, i = args
    return i, float(wrapper(sol))


def eval_pop(wrapper, pop: np.ndarray, n_workers: int) -> np.ndarray:
    """Evaluate entire population in parallel. Returns fitness array."""
    fit = np.full(len(pop), np.inf)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_eval, (wrapper, pop[i], i)): i for i in range(len(pop))}
        for f in as_completed(futs):
            i, v = f.result()
            fit[i] = v
    return fit


def _mix(args):
    """
    Optimal mixing for one individual.

    Copies gene values from *donor* into *src* at the positions given by
    *mask* and accepts the child only if fitness improves.
    """
    wrapper, src, src_fit, donor, mask = args
    child = src.copy()
    for gi in mask:
        child[gi] = donor[gi]
    child_fit = float(wrapper(child))
    if child_fit < src_fit:
        return child, child_fit, True
    return src, src_fit, False


# ── Pair-cluster mutation ────────────────────────────────────────────────────

def mutate_pair_cluster(
    sol: np.ndarray,
    pair_clusters: list[tuple[str, str]],
    tls_to_genes: dict[str, tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Novel pair-cluster mutation.

    Algorithm
    ---------
    1. Randomly select a 2-TLS cluster from *pair_clusters*.
    2. Randomly assign which TLS plays the "first" (smaller raw-sum) role
       and which plays the "second" (larger raw-sum) role.
    3. Sample raw phase-duration genes for the **first** TLS uniformly from
       [GENE_LOW, GENE_HIGH].
    4. Compute ``sum1`` = sum of those genes.  Sample a target total for the
       **second** TLS uniformly from ``(sum1, GENE_HIGH × n2)``, then draw
       phase proportions at random and scale them to that target.  After
       clipping to ``[GENE_LOW, GENE_HIGH]``, a hard enforcement loop
       redistributes any deficit proportionally across genes with headroom,
       guaranteeing ``sum(genes_second) > sum(genes_first)`` in all cases
       where the physical ceiling ``GENE_HIGH × n2`` permits it.
    5. Both TLS are still normalised to exactly 90 s by ``_rebuild_json``,
       so the 90-second budget is always satisfied.  The ordering constraint
       shapes the raw gene-space, which in turn biases the phase-ratio
       distribution that lands in the solution after normalisation.

    Parameters
    ----------
    sol           : current solution vector — returned as a new copy
    pair_clusters : list of (tls_id_a, tls_id_b) from the full Ward tree
    tls_to_genes  : maps tls_id → (start_idx, end_idx) in the gene vector
    rng           : numpy random Generator

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

    s1, e1 = tls_to_genes[tls_first]
    s2, e2 = tls_to_genes[tls_second]
    n1 = e1 - s1          # number of phase genes in first TLS
    n2 = e2 - s2          # number of phase genes in second TLS

    # ── Step 3: sample first TLS freely ──────────────────────────────────
    genes1 = rng.uniform(GENE_LOW, GENE_HIGH, n1)
    sum1   = float(genes1.sum())

    # ── Step 4: sample second TLS with sum strictly > sum1 ───────────────
    #   Physical ceiling: every gene pinned at GENE_HIGH.
    max_sum2 = GENE_HIGH * n2
    min_sum2 = sum1 + 1e-6        # strictly greater than first TLS

    if min_sum2 < max_sum2:
        # Pick a target total for the second TLS, above sum1
        target2 = float(rng.uniform(min_sum2, max_sum2))

        # Draw random proportions and scale to target2
        raw2   = rng.uniform(GENE_LOW, GENE_HIGH, n2)
        genes2 = raw2 / raw2.sum() * target2
        genes2 = np.clip(genes2, GENE_LOW, GENE_HIGH)

        # ── Hard enforcement: clipping can shrink the sum below sum1.
        #   Iteratively raise the smallest gene(s) until the constraint
        #   is satisfied or every gene is already at GENE_HIGH.
        deficit = sum1 + 1e-6 - float(genes2.sum())
        while deficit > 0:
            # Find genes that still have headroom
            headroom = GENE_HIGH - genes2
            total_headroom = float(headroom.sum())

            if total_headroom <= 0:
                # Every gene is at GENE_HIGH; constraint physically impossible
                # for this first-TLS draw — fall through to the else branch
                break

            # Distribute the deficit proportionally across available headroom
            boost   = np.minimum(headroom, headroom / total_headroom * deficit)
            genes2 += boost
            genes2  = np.clip(genes2, GENE_LOW, GENE_HIGH)
            deficit = sum1 + 1e-6 - float(genes2.sum())

    else:
        # Edge case: sum1 is already at the physical ceiling for n2 genes.
        # Cannot satisfy the ordering constraint — pin all second genes at
        # GENE_HIGH to get as close as possible.
        genes2 = np.full(n2, GENE_HIGH)

    new_sol        = sol.copy()
    new_sol[s1:e1] = genes1
    new_sol[s2:e2] = genes2
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
    rng: np.random.Generator,
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
       from [GENE_LOW, GENE_HIGH].

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
        s, e = tls_to_genes[selected_tls]
        new_sol[s:e] = rng.uniform(GENE_LOW, GENE_HIGH, e - s)
        return new_sol

    containing_info = tree_structure[containing_node]

    # ── Step 3a: smallest cluster is a pair → apply paired mutation ──────
    if containing_info["size"] == 2:
        pair = (containing_info["members"][0], containing_info["members"][1])
        return mutate_pair_cluster(sol, [pair], tls_to_genes, rng)

    # ── Step 3c: cluster size > 2 → tree-walk to collect genes ───────────
    tls_to_mutate = _collect_tree_walk_genes(
        containing_node, tree_structure, tls_to_genes, rng
    )

    # ── Step 4: mutate collected genes ───────────────────────────────────
    for tls_id in tls_to_mutate:
        if tls_id in tls_to_genes:
            s, e = tls_to_genes[tls_id]
            new_sol[s:e] = rng.uniform(GENE_LOW, GENE_HIGH, e - s)

    return new_sol


# ── Core GOMEA loop ──────────────────────────────────────────────────────────

def run_custom_optimizer(
    tree_name: str,
    dist_path: str,
    strategy: str,
    baseline_data: dict,
    wrapper,
    num_genes: int,
    baseline_vec: np.ndarray,
    tls_to_genes: dict[str, tuple[int, int]],
    pop_size: int,
    num_gen: int,
    noise_std: float,
    n_workers: int,
    seed: int = 42,
) -> dict:
    """Run one custom optimization experiment. Returns a results dict."""
    rng       = np.random.default_rng(seed)
    threshold = THRESHOLDS[tree_name]

    print(f"\n{'='*60}")
    print(f"Tree: {tree_name} (t={threshold}) | Strategy: {strategy} | Pop: {pop_size}")
    print(f"{'='*60}")

    # 1. Build masks ─────────────────────────────────────────────────────────
    #    mixing_masks : all sub-threshold Ward clusters (parent + children),
    #                   root excluded, singletons excluded.
    #    pair_clusters: all 2-TLS nodes in the full tree (for mutation).
    tls_masks, pair_clusters, tree_structure = build_all_tree_masks(
        dist_path, threshold
    )

    # Convert TLS-ID masks → gene-index masks; require ≥ 2 gene indices
    gene_masks = [mask_to_gene_indices(m, tls_to_genes) for m in tls_masks]
    gene_masks = [m for m in gene_masks if len(m) >= 2]

    # Filter pair_clusters to those where both TLS exist in the gene map
    valid_pairs = [(a, b) for a, b in pair_clusters
                   if a in tls_to_genes and b in tls_to_genes]

    cluster_sizes = sorted(set(len(m) for m in gene_masks))
    print(f"Mixing masks : {len(gene_masks)} clusters "
          f"(gene-group sizes: {cluster_sizes})")
    print(f"Pair-mutation: {len(valid_pairs)} 2-TLS pairs available")
    n_walk_clusters = sum(
        1 for k, v in tree_structure.items()
        if not isinstance(k, str) and v["size"] > 2
    )
    print(f"Tree-walk   : {n_walk_clusters} clusters with size > 2")

    if not gene_masks:
        raise RuntimeError(
            "No valid mixing masks found — check threshold / distance file."
        )

    # 2. Init & evaluate population ─────────────────────────────────────────
    pop = init_population(strategy, pop_size, num_genes, baseline_vec, noise_std, rng)
    fit = eval_pop(wrapper, pop, n_workers)

    best_i              = int(np.argmin(fit))
    best_sol, best_fit  = pop[best_i].copy(), float(fit[best_i])
    print(f"Gen 0 | Best: {best_fit:.2f} | Mean: {np.mean(fit):.2f}")

    history = [{"gen": 0, "best": float(best_fit), "mean": float(np.mean(fit))}]

    # 3. Generational loop ───────────────────────────────────────────────────
    t0 = time.time() 
    num_evals = pop_size
    gen = 1
    while num_evals < MAX_EVALS: 

        # ── Optimal mixing ───────────────────────────────────────────────────
        # Each individual is paired with a random donor; a random mask from the
        # full sub-threshold cluster set (parents AND children) is applied.
        mix_tasks = []
        for i in range(pop_size):
            donor_i = i
            while donor_i == i:
                donor_i = int(rng.integers(0, pop_size))
            mask = gene_masks[int(rng.integers(0, len(gene_masks)))]
            mix_tasks.append(
                (wrapper, pop[i].copy(), fit[i], pop[donor_i].copy(), mask)
            )

        mix_improved = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_mix, t): idx for idx, t in enumerate(mix_tasks)}
            for f in as_completed(futs):
                idx = futs[f]
                new_sol, new_fit, improved = f.result()
                pop[idx] = new_sol
                fit[idx] = new_fit
                num_evals += 1
                if improved:
                    mix_improved += 1

        # ── Mutation ──────────────────────────────────────────────────────────
        # A random subset (≈ MUTATION_RATE) of individuals are mutated.
        # Each mutant randomly selects a TLS gene; the mutation method is
        # determined by that gene's position in the Ward tree:
        #   - smallest cluster is a pair (size 2) → paired mutation
        #   - smallest cluster has size > 2       → tree-walk mutation
        #   - standalone gene (no cluster)        → direct re-sampling
        # Each mutant is evaluated; the mutation is accepted only if fitness
        # improves, preserving the greedy quality of LT-GOMEA.
        mut_improved = 0
        mutant_idxs = []

        if LT_GOMEA_USE_MUTATION: 
            mutant_idxs = [i for i in range(pop_size) if rng.random() < MUTATION_RATE]

            if mutant_idxs:
                mutants = [
                    mutate_tree_walk(
                        pop[i], tree_structure, tls_to_genes,
                        valid_pairs, rng
                    )
                    for i in mutant_idxs
                ]

                mut_fit_map: dict[int, float] = {}
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futs = {
                        pool.submit(_eval, (wrapper, mutants[j], j)): j
                        for j in range(len(mutant_idxs))
                    }
                    for f in as_completed(futs):
                        j, v = f.result()
                        mut_fit_map[j] = v

                for j, i in enumerate(mutant_idxs):
                    new_fit = mut_fit_map[j]
                    num_evals += 1
                    if new_fit < fit[i]:          # accept only improvements
                        pop[i] = mutants[j]
                        fit[i] = new_fit
                        mut_improved += 1

        # ── Track global best ────────────────────────────────────────────────
        gi = int(np.argmin(fit))
        if fit[gi] < best_fit:
            best_fit = float(fit[gi])
            best_sol = pop[gi].copy()

        history.append({
            "gen":          gen,
            "best":         float(best_fit),
            "mean":         float(np.mean(fit)),
            "mix_improved": mix_improved,
            "mut_improved": mut_improved,
            "mutants":      len(mutant_idxs),
        })
        print(
            f"Gen {gen:2d} | Best: {best_fit:.2f} | Mean: {np.mean(fit):.2f} "
            f"| Mix+{mix_improved} | Mut+{mut_improved}/{len(mutant_idxs)}"
        )
        gen += 1

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s | Final best: {best_fit:.2f}")

    best_json = _rebuild_json(best_sol, baseline_data, tls_to_genes)
    best_json["composite_cost"] = float(best_fit)

    return {
        "best_configuration":  best_json,
        "best_fitness":        float(best_fit),
        "fitness_history":     history,
        "time_s":              round(elapsed, 2),
        "tree":                tree_name,
        "threshold":           threshold,
        "strategy":            strategy,
        "pop_size":            pop_size,
        "generations":         num_gen,
        "num_mixing_masks":    len(gene_masks),
        "num_pair_clusters":   len(valid_pairs),
        "seed":                seed,
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _rebuild_json(sol: np.ndarray, baseline: dict, tls_to_genes: dict) -> dict:
    """Convert flat gene vector back to the full TLS JSON format (90 s per TLS)."""
    out = copy.deepcopy(baseline)
    for tls_id in sorted(out["tls_data"]):
        if tls_id not in tls_to_genes:
            continue
        s, e  = tls_to_genes[tls_id]
        raw   = sol[s:e]
        keys  = sorted(out["tls_data"][tls_id])
        n     = len(keys)
        total = float(sum(raw))

        if total <= 0:
            dur       = [90 // n] * n
            dur[-1]  += 90 - sum(dur)
        else:
            dur  = [max(1, int(round(d * 90 / total))) for d in raw]
            diff = 90 - sum(dur)
            if diff:
                dur[int(np.argmax(dur))] += diff

        for i, pk in enumerate(keys):
            out["tls_data"][tls_id][pk]["duration"] = int(dur[i])
    return out


# ── Experiment runner ────────────────────────────────────────────────────────

def run_all_experiments():
    with open(BASELINE_TRAFFIC_DATA) as f:
        baseline = json.load(f)

    wrapper, num_genes, _, _, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline, fitness_function=_traffic_fitness
    )
    tls_to_genes, _, baseline_vec = build_gene_map(baseline)
    n_workers = NUM_PROCESSORS or os.cpu_count() or 1

    root    = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "src" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = {
        "shortest":  out_dir / "tls_distances_shortest.json",
        "euclidian": out_dir / "tls_distances_euclidian.json",
        "fastest":   out_dir / "tls_distances_fastest.json",
    }
    strategies = ["random", "baseline", "mixed"]
    summary: dict[str, dict] = {}

    for tree_name, path in trees.items():
        for strat in strategies:
            label = f"{tree_name}_{strat}"
            try:
                res = run_custom_optimizer(
                    tree_name, str(path), strat, baseline,
                    wrapper, num_genes, baseline_vec, tls_to_genes,
                    LT_GOMEA_POPULATION_SIZE, LT_GOMEA_NUM_GENERATIONS,
                    LT_GOMEA_BASELINE_NOISE_STD, n_workers,
                )
                mutation_suffix = "_mutation" if LT_GOMEA_USE_MUTATION else ""
                out_file = out_dir / f"custom_optimizer_{label}{mutation_suffix}.json"
                with open(out_file, "w") as f:
                    json.dump(res, f, indent=4)
                print(f"Saved → {out_file}")
                summary[label] = {"best": res["best_fitness"], "time_s": res["time_s"]}

            except Exception as e:
                print(f"ERROR [{label}]: {e}")
                import traceback; traceback.print_exc()
                summary[label] = {"error": str(e)}

    # ── Results table ────────────────────────────────────────────────────────
    print(f"\n{'Tree':<15} {'Strategy':<10} {'Best':>12} {'Time':>8}")
    print("─" * 47)
    for label, info in summary.items():
        t, s = label.rsplit("_", 1)
        if "error" in info:
            print(f"{t:<15} {s:<10} {'ERROR':>12}")
        else:
            print(f"{t:<15} {s:<10} {info['best']:>12.2f} {info['time_s']:>7.1f}s")


if __name__ == "__main__":
    run_all_experiments()