# Pipeline — Custom Optimizer (LT-GOMEA)

This document walks through
[`custom_optimizer.py`](../algorithms/custom_optimizer.py) end-to-end, from
baseline traffic data to a saved, SUMO-ready timing plan.

The custom optimizer is a **Linkage-Tree GOMEA**: it improves individuals with
greedy *optimal mixing* over Ward clusters, plus a structure-aware *mutation*
operator. Every change is accepted only if fitness improves, so the population
never gets worse. For how the final vector becomes a valid plan see
[`traffic_light_normalization.md`](traffic_light_normalization.md).

> Sibling pipeline:
> [`pipeline_differential_evolution_cluster_v3.md`](pipeline_differential_evolution_cluster_v3.md).

---

## 1. The gene vector (shared groundwork)

The **gene vector** is the common currency: a flat array of phase durations
where each traffic light (TLS) owns one contiguous block, one gene per phase.
`build_gene_map` ([line 77](../algorithms/custom_optimizer.py#L77)) records the
`tls_id → (start, end)` slice for every TLS, so the operators can address whole
intersections at once.

```
genes:  [ TLS_A p0 | TLS_A p1 | TLS_B p0 | TLS_B p1 | TLS_B p2 | ... ]
        └── A=(0,2) ──┘└──────── B=(2,5) ─────────┘
```

The run sweeps a grid of **3 Ward linkage trees** (`shortest` / `euclidian` /
`fastest`) × **3 population strategies** (`random` / `baseline` / `mixed`) =
**9 experiments**.

---

## 2. The flow

```
run_all_experiments()                          # 3 trees × 3 strategies
  └─ run_custom_optimizer(tree, strategy)       # one GOMEA run
       1. build masks from the Ward tree
       2. init + evaluate population
       3. loop until MAX_EVALS:
            a. optimal mixing  (greedy, donor-based)
            b. mutation        (tree-walk, greedy)
       4. rebuild best vector → JSON, save
```

---

## 3. Step by step

### Step 1 — Build the operators' inputs

[`build_all_tree_masks`](../novel/linkage_tree.py#L47)
([called at line 183](../algorithms/custom_optimizer.py#L183)) turns the
distance JSON into three things:

- **mixing masks** — every sub-threshold Ward cluster (parents *and* children;
  root and singletons excluded) → used by optimal mixing;
- **pair clusters** — every 2-TLS node in the full tree → used by mutation;
- **tree structure** — the full hierarchy → walked by mutation.

Masks are converted from TLS IDs to gene indices (requiring ≥ 2 genes), and a
`phase_split` (green / red / yellow indices per TLS) is built for the mutation
operator. The cluster threshold per tree comes from `CLUSTER_THRESHOLD_*` in
[`config.py`](../../config.py).

### Step 2 — Init and evaluate

[`init_population`](../algorithms/custom_optimizer.py#L101) builds the starting
population with the run's strategy (`random`, `baseline`-perturbed, or `mixed`);
`eval_pop` scores it in parallel.

### Step 3 — The generational loop

The loop ([line 233](../algorithms/custom_optimizer.py#L233)) runs two operators
per generation:

**3a. Optimal mixing** ([line 235](../algorithms/custom_optimizer.py#L235)) —
each individual is paired with a random donor and a random mixing mask; `mix`
([`optimal_mixing.py:30`](../novel/optimal_mixing.py#L30)) copies the donor's
genes at the mask positions and keeps the child **only if fitness improves**.

**3b. Mutation** ([line 260](../algorithms/custom_optimizer.py#L260)) — when
`NOVEL_MUTATION` is enabled, a `MUTATION_RATE` fraction of individuals are
mutated by `mutate_tree_walk`
([`pairwise_mutation.py:296`](../novel/pairwise_mutation.py#L296)). The method
is chosen by where the selected TLS sits in the Ward tree:

- smallest containing cluster is a **pair (size 2)** → paired green-growth
  mutation;
- cluster **size > 2** → tree-walk that re-samples individual TLS genes,
  skipping pair sub-clusters;
- **standalone** TLS → direct re-sampling.

Again, mutations are **accepted only on improvement**. Each generation records
best/mean/worst fitness and how many mixes and mutations improved.

### Step 4 — Rebuild and save

[`_rebuild_json`](../algorithms/custom_optimizer.py#L347) applies the **same**
`normalize_to_cycle` the fitness wrapper uses, so the saved JSON matches what
was actually evaluated. Output goes to
`src/outputs/custom_optimizer_<tree>_<strategy>[_mutation].json`.

---

## 4. End-to-end data flow

```
baseline data ─▶ build_gene_map ─▶ initial population
        │
        └─ Ward tree ─▶ build_all_tree_masks ─▶ mixing masks / pair clusters / tree
                                                        │
                                                        ▼
                                  generation: optimal mixing + mutation
                                       (greedy — accept only if better)
                                                        │
                                              best vector ─▶ normalize ─▶ JSON
```
