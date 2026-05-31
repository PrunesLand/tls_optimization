# Novel Bin-Crossing Crossover (v3)

This document explains the **walk-decomposition cluster crossover** implemented in
[`differential_evolution_cluster_v3.py`](../algorithms/differential_evolution_cluster_v3.py),
backed by the tree-walk in
[`node_finder_v3.py`](../novel/node_finder_v3.py).

It is a drop-in replacement for SHADE's standard per-gene binomial crossover.
Instead of deciding gene-by-gene whether to copy from the mutant or the parent,
v3 selects whole **clusters of traffic-light signals (TLS)** — coherent groups of
nearby intersections — and copies all their genes together.

> For how this operator fits into the full SHADE run, see
> [`pipeline_differential_evolution_cluster_v3.md`](pipeline_differential_evolution_cluster_v3.md).

---

## 1. Background: what crossover normally does in SHADE

Standard Differential Evolution crossover (`DE_binary_crossover`) builds a *trial*
vector from two parents:

- the **mutant** (donor) vector produced by DE's mutation step, and
- the **current** (parent) vector.

For each gene `j` it flips a biased coin with probability `CR_i` (the per-individual,
self-adapted crossover rate). Heads → take the mutant's gene; tails → keep the
parent's gene. The result is a trial vector with roughly `CR_i × num_genes` genes
inherited from the mutant, scattered randomly across the genome.

**Problem for our domain:** genes are phase durations grouped by intersection. A
random per-gene mask shreds an intersection's phases across mutant and parent and
ignores the road-network structure entirely. v3 replaces that scattered mask with
a structured, *spatially coherent* selection.

---

## 2. The genome layout

Each TLS owns a **contiguous block of genes**, one gene per signal phase. The map
is built once in `build_gene_map`
([`differential_evolution_cluster_v3.py:159`](../algorithms/differential_evolution_cluster_v3.py#L159)):

```
tls_to_genes[tls_id] = (start_index, end_index)   # half-open slice [s, e)
```

So the flat genotype looks like:

```
genes:  [ TLS_A phase0 | TLS_A phase1 | TLS_B phase0 | TLS_B phase1 | TLS_B phase2 | ... ]
         └──── tls_to_genes["A"]=(0,2) ────┘└──────── tls_to_genes["B"]=(2,5) ────────┘
```

The crossover decision is made **per TLS**, then expanded to that TLS's gene slice.
A TLS is therefore inherited *entirely* from the mutant or *entirely* from the
parent — never split mid-intersection.

---

## 3. The linkage tree (where "coherent clusters" come from)

Before optimisation, a **Ward-linkage hierarchical clustering tree** is built over
the TLS pairwise distance matrix (one tree per distance metric: shortest-path,
euclidian, fastest-path). This happens in
`LinkageTree.from_distance_json` (in [`node_finder.py`](../novel/node_finder.py), inherited
by the v3 subclass).

Key properties of the tree:

- **Leaves** are individual TLS (size 1). There are `n` of them.
- **Internal nodes** are merged clusters; the **root** contains all `n` TLS.
- Every node has a precomputed `size` (number of leaves under it) and `members`
  (the actual TLS-IDs under it).
- Two TLS that are close in the chosen distance metric merge **low** in the tree;
  distant ones merge **high**. So any subtree is a *spatially coherent group* of
  intersections — e.g. a corridor or a neighbourhood.

This is the structure the crossover walks to assemble its selection.

---

## 4. The crossover, step by step

The hook is `_cluster_binary_crossover`
([`differential_evolution_cluster_v3.py:57`](../algorithms/differential_evolution_cluster_v3.py#L57)).
It is installed globally at import time:

```python
_shade_module.DE_binary_crossover = _cluster_binary_crossover
```

so SHADE calls it in place of its own crossover, with the identical signature
`(mutation_vector, current_vect, CR_vect)`. SHADE's mutation, CR/F adaptation, and
success-history archive are otherwise unmodified.

For each individual *i* in the population:

### Step 1 — CR becomes a target cluster size

```python
targets = np.rint(cr_np * _num_tls).astype(int)     # line 71
```

`CR_i ∈ [0, 1]` is scaled by `_num_tls` (the number of intersections, 36 here) and
**rounded to the nearest integer** to get a whole-number target size in
`[0, num_tls]`.

- `CR_i = 0`    → target = 0   → inherit nothing from the mutant.
- `CR_i = 1`    → target = 36  → inherit the whole network from the mutant.
- `CR_i = 0.5`  → target = 18  → inherit about half the network.

`np.rint` uses banker's rounding (ties go to the nearest even integer). Because the
target is an integer and every cluster size in the tree is also an integer, the
walk's **exact-match termination can actually fire** — see §6.

### Step 2 — decompose the target into clusters (the tree walk)

```python
members = _linkage_tree.find_node_decomposition(int(targets[i]), rng=_xover_rng)
```

`find_node_decomposition` ([`node_finder_v3.py`](../novel/node_finder_v3.py)) returns a flat
list of TLS-IDs assembled by walking the Ward tree. The algorithm:

1. **Pick the starting node** — the *smallest* cluster whose `size >= target`
   ("closest and highest"). Ties broken uniformly at random. (If nothing is large
   enough, fall back to the largest cluster, i.e. the root.)

2. **Walk down**, tracking a `remainder` (initialised to `target`):
   - If the **current subtree's** size exactly equals the remainder → take the
     whole subtree and stop.
   - Inspect the node's **two children**. If **either child's size exactly equals
     the remainder** → take that child and stop (clean termination, remainder = 0).
   - Otherwise → **flip a fair coin (50/50)** to pick one of the two children.
     Append that child's cluster to the selection, subtract its size from
     `remainder`, and continue the walk from the **sibling** subtree (the child we
     did *not* pick).

3. **Stop** when any of these happens:
   - remainder reaches exactly 0 (clean), or
   - we reach a **leaf** (cannot descend further), or
   - a pick **overshoots** so remainder goes negative.

The returned `members` is the **union of TLS-IDs across every selected cluster**.
Because each pick comes from a disjoint sibling subtree, the selected clusters never
share TLS — the union is automatically a set.

#### Worked example

Target `= 18`, with a tree whose relevant sizes are:

```
19 ─┬─ 5  ─┬─ 2
    │      └─ 3
    └─ 14 ─┬─ 6
           └─ 8 ─┬─ 4 ─┬─ 1
                 │      └─ 3
                 └─ 4
```

| Step | Current node | Children | Action | Picked | Remainder |
|------|--------------|----------|--------|--------|-----------|
| 0 | — | — | start = smallest cluster ≥ 18 | (node 19) | 18 |
| 1 | 19 | {5, 14} | 50/50 → pick 5, recurse into 14 | **5** | 18 − 5 = 13 |
| 2 | 14 | {6, 8} | 50/50 → pick 6, recurse into 8 | **6** | 13 − 6 = 7 |
| 3 | 8 | {4, 4} | 50/50 → pick 4, recurse into other 4 | **4** | 7 − 4 = 3 |
| 4 | 4 | {1, 3} | child 3 == remainder → take it, stop | **3** | 3 − 3 = 0 ✅ |

Selected clusters: sizes **5 + 6 + 4 + 3 = 18**. `members` is the union of those four
clusters' TLS-IDs. Notice the algorithm assembled the target out of several coherent
groups rather than forcing one single cluster of size 18 (which may not even exist
in the tree).

### Step 3 — translate TLS-IDs to gene indices

```python
gene_idxs = []
kept = 0
for tls_id in members:
    if tls_id in _tls_to_genes:        # line 88
        s, e = _tls_to_genes[tls_id]
        gene_idxs.extend(range(s, e))
        kept += 1
```

Each selected TLS expands to its contiguous gene slice. The
`if tls_id in _tls_to_genes` guard is defensive: the linkage tree's TLS-IDs come
from the distance JSON, and a TLS there may have no phases in the baseline data
(so no gene slice). `kept` records how many TLS blocks were actually applied.

### Step 4 — splice mutant genes into the parent

```python
trial = current_vect.clone()           # line 69 — start as a full copy of the parent
...
idx = torch.tensor(gene_idxs, dtype=torch.long, device=trial.device)
trial[i].index_copy_(0, idx, mutation_vector[i].index_select(0, idx))   # line 96
```

The trial vector starts as the parent. Only the gene positions belonging to the
selected clusters are overwritten with the **mutant's** values. Every other TLS
keeps the parent's genes.

Result: the trial inherits **whole intersections** from the mutant — a coherent
spatial subset — and the rest from the parent.

### Step 5 — logging

```python
_last_cr_vect      = CR_vect.detach().clone()    # line 65
_last_target_sizes = targets.tolist()            # line 98
_last_actual_sizes = actual.tolist()             # line 99 (actual = kept counts)
```

These module-level captures let the generation loop print and store, per
individual, the `target → actual` cluster sizes
([`differential_evolution_cluster_v3.py:236`](../algorithms/differential_evolution_cluster_v3.py#L236)).
`actual` (the `kept` count) can differ from `target` because of the gene-map guard
and the walk's overshoot/under-fill.

---

## 5. End-to-end data flow

```
CR_i  ──×num_tls, round──▶  target (int)
                         │
                         ▼
        find_node_decomposition(target)        ◀── Ward linkage tree
                         │   (greedy 50/50 walk)
                         ▼
        members = [tls_id, tls_id, ...]         (union of selected clusters)
                         │   (tls_to_genes)
                         ▼
        gene_idxs = [g, g, g, ...]              (contiguous per-TLS slices)
                         │
                         ▼
        trial = parent, then trial[gene_idxs] = mutant[gene_idxs]
```

---

## 6. Design notes & caveats

- **Integer targets can terminate cleanly.** `target = round(CR × num_tls)` is a
  whole number, and every cluster size in the tree is also an integer, so the
  "exact match" termination genuinely fires whenever the walk lands a node/child
  whose size equals the remainder (as in the worked example, which ends at
  remainder 0). Walks that do not find an exact match still stop by hitting a leaf
  or overshooting, leaving a non-zero remainder. `find_node_decomposition` prints a
  diagnostic only when `|remainder| ≥ 1`, so clean (remainder 0) walks stay quiet
  while genuine under-fill / overshoot is surfaced.

- **High stochastic variety.** A fair coin is flipped at *every* level of descent,
  so two calls with the *same* target can produce very different TLS sets. This adds
  exploration relative to v1 (which returns a single deterministic-by-tree nearest
  cluster), but also increases trial-to-trial variance.

- **CR = 0 corner case.** target ≈ 0 → the walk returns an empty list → the trial is
  a pure copy of the parent (no mutant inheritance), which matches DE semantics for
  CR = 0.

- **No within-TLS mixing.** Because selection is per-TLS and expands to whole gene
  slices, an intersection's phases are never split between mutant and parent.

- **Disjoint clusters.** Each pick comes from a sibling subtree not yet visited, so
  selected clusters never overlap; no TLS is inherited twice.


