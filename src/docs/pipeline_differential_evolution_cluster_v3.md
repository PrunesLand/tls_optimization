# Pipeline — SHADE with Cluster Crossover (v3)

This document walks through
[`differential_evolution_cluster_v3.py`](../algorithms/differential_evolution_cluster_v3.py)
end-to-end, from baseline traffic data to a saved, SUMO-ready timing plan.

`SHADE` is a self-adapting Differential Evolution from the EvoX library. The v3
twist replaces SHADE's per-gene crossover with a **cluster crossover** that
inherits whole intersections from the donor instead of scattered genes. For the
crossover internals see [`novel_bin_crossing_v3.md`](novel_bin_crossing_v3.md);
for how the final vector becomes a valid plan see
[`traffic_light_normalization.md`](traffic_light_normalization.md).

> Sibling pipeline: [`pipeline_custom_optimizer.md`](pipeline_custom_optimizer.md).

---

## 1. The gene vector (shared groundwork)

The **gene vector** is the common currency: a flat array of phase durations
where each traffic light (TLS) owns one contiguous block, one gene per phase.
`build_gene_map` ([line 159](../algorithms/differential_evolution_cluster_v3.py#L159))
records the `tls_id → (start, end)` slice for every TLS, so the crossover can
address whole intersections at once.

```
genes:  [ TLS_A p0 | TLS_A p1 | TLS_B p0 | TLS_B p1 | TLS_B p2 | ... ]
        └── A=(0,2) ──┘└──────── B=(2,5) ─────────┘
```

The run loops over three **Ward linkage trees**, one per distance metric
(`shortest` / `euclidian` / `fastest`); each gives a different clustering for the
crossover. That makes 3 experiments per invocation.

---

## 2. The flow

```
run_all_experiments()                          # 3 trees
  └─ run_single_de(tree)                        # one SHADE run
       1. install per-run state for the crossover hook
       2. build initial population + EvoX workflow
       3. loop: workflow.step()  until MAX_EVALS
       4. rebuild best vector → JSON, save
```

---

## 3. Step by step

### Step 1 — Install the crossover hook (at import)

[`differential_evolution_cluster_v3.py:103`](../algorithms/differential_evolution_cluster_v3.py#L103)
monkey-patches EvoX's `DE_binary_crossover` with `_cluster_binary_crossover`
([line 57](../algorithms/differential_evolution_cluster_v3.py#L57)):

```python
_shade_module.DE_binary_crossover = _cluster_binary_crossover
```

SHADE's mutation, CR/F adaptation, and success-history archive are untouched —
only the crossover changes.

### Step 2 — Set per-run state

[`run_single_de`](../algorithms/differential_evolution_cluster_v3.py#L174) loads
the run's Ward tree (`LinkageTree.from_distance_json`) and stashes it, the gene
map, the TLS count, and the RNG in module globals so the hook can reach them.

### Step 3 — Build the population and workflow

A `TLSProblem` evaluates a whole population in parallel through the SUMO fitness
wrapper. A custom initial population is injected directly into the algorithm;
the init-step evaluations are **not** counted against `MAX_EVALS` — the budget
covers only the generational loop.

### Step 4 — The search loop

Each `workflow.step()`
([line 224](../algorithms/differential_evolution_cluster_v3.py#L224)) runs one
SHADE generation. Inside the crossover, every individual's self-adapted `CR_i`
becomes a **target cluster size**:

```
target_i = round(CR_i × num_tls)
```

The Ward tree is then walked to assemble roughly that many TLS, and those TLS's
genes are copied from the DE donor (mutant) into the parent. Per generation the
loop logs best/mean/worst fitness and the `target → actual` cluster sizes.

### Step 5 — Rebuild and save

The best vector is converted back to the full TLS JSON (durations re-scaled to
the 90 s cycle) and written to
`src/outputs/differential_evolution_cluster_v3_<tree>.json`.

---

## 4. End-to-end data flow

```
baseline data ─▶ build_gene_map ─▶ initial population
                                        │
                                        ▼
                           SHADE generation (workflow.step)
                                        │
              CR_i ──round(×num_tls)──▶ target cluster size
                                        │  (Ward-tree walk)
                                        ▼
                    donor genes spliced into parent (whole TLS)
                                        │  (DE selection)
                                        ▼
                          best vector ─▶ normalize ─▶ JSON
```
