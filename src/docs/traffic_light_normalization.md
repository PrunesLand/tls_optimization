# Traffic Light Normalisation & Pair-wise Mutation

This document covers two related concerns:

1. How a TLS's raw gene values (which can take almost any number in their per-phase bounds) are turned into a valid 90-second cycle — the **normalisation** pass.
2. How the **pair-wise mutation** operator chooses which gene positions to rewrite and how it samples their new values.

The two work together: mutation produces raw values; normalisation enforces the 90 s cycle invariant. The same normalisation runs both inside the fitness wrapper (so SUMO sees a 90 s cycle) and at save time (so the saved JSON matches what was actually simulated).

> Normalisation is stage "rebuild JSON" of both optimizers — see
> [`pipeline_custom_optimizer.md`](pipeline_custom_optimizer.md) and
> [`pipeline_differential_evolution_cluster_v3.md`](pipeline_differential_evolution_cluster_v3.md).

---

## 0. Setup that both parts depend on

### Phase shapes in the baseline

After the phase-count filter in [generation.py](../sumo_setup/generation.py) (only `OPTIMIZE_PHASE_COUNTS = {3, 4}` survive), every TLS has one of two shapes:

| Shape | Count | Example baseline durations |
|---|---:|---|
| `G, y, r` (3-phase) | 24 | `[79, 6, 5]` |
| `G, y, G, y` (4-phase) | 11 | `[39, 6, 39, 6]` |

### Per-phase bounds

Floors come from the per-type constants in [config.py](../../config.py)
(`GREEN_FLOOR`, `RED_FLOOR`, `YELLOW_FLOOR`/`YELLOW_CEIL`). Only yellow has a
static ceiling; green/red ceilings are dynamic (below).

| Phase type | Min | Max |
|---|---:|---:|
| `green` | 24 | dynamic |
| `yellow` | 3 | 6 |
| `red` | 5 | dynamic |

### Dynamic per-TLS upper bound

Green and red have **no static ceiling**: in a fixed `CYCLE_LENGTH = 90` s
cycle a green (or red) phase can never be so long that there is no room left
for the (frozen) yellows plus a legal minimum for every other non-yellow phase.
So each TLS gets its own per-phase ceiling, computed once at baseline-load time
by `phase_upper_bounds` in
[fitness_evaluation.py](../sumo_setup/fitness_evaluation.py):

```
ceiling(phase_j) = CYCLE_LENGTH − Σ(yellow durations)
                                − Σ(min of every OTHER non-yellow phase)
```

where each non-yellow min comes from `PHASE_FLOOR[ptype]` (green=24,
red=5). Yellow phases keep their static `[YELLOW_FLOOR, YELLOW_CEIL]` =
`[3, 6]` bound.

| Shape | Yellow | Green ceiling | Red ceiling |
|---|---:|---:|---:|
| `G, y, r` (3-phase) | 6 | `90−6−5 = 79` | `90−6−24 = 60` |
| `G, y, r` (3-phase) | 3 | `90−3−5 = 82` | `90−3−24 = 63` |
| `G, y, G, y` (4-phase) | 6,6 | `90−12−24 = 54` | — |

This per-gene ceiling array (`ub`) is the single source of truth: it is
returned by `build_traffic_fitness_wrapper`, fed into `normalize_to_cycle`'s
clamp, and used as the per-gene sampling/mutation upper bound by every
algorithm. Scalar-`GREEN_FLOOR` samplers cap their lower bound at
`min(GREEN_FLOOR, ub)` so a yellow gene (ceiling 6) stays a valid
`uniform`/`gene_space` range.

### Phase-type classification

`phase_type(state)` in [fitness_evaluation.py](../sumo_setup/fitness_evaluation.py) classifies each phase by the **most frequent character** in its SUMO state string:

- contains predominantly `g`/`G` → `green`
- contains predominantly `y`/`Y` → `yellow`
- contains predominantly `r`/`R` → `red`

This classification is the single source of truth used by both the normaliser and the mutation operator.

### The gene vector

Each TLS owns a contiguous slice of the flat gene vector — one gene per phase, in the order returned by `sorted(phases)`. So for a 3-phase TLS, three consecutive positions; for a 4-phase TLS, four.

---

## Part 1 — Normalisation (`normalize_to_cycle`)

Lives at [fitness_evaluation.py:162-199](../sumo_setup/fitness_evaluation.py#L162-L199). Called from **two places**:

- `TrafficFitnessWrapper.__call__` — every fitness evaluation, before SUMO sees the durations.
- `_rebuild_json` in [custom_optimizer.py](../algorithms/custom_optimizer.py) — at save time, when writing the best solution to disk.

Both call sites use the same function with the same per-TLS phase types, so **the saved JSON is byte-for-byte the configuration that produced the reported fitness**.

### Algorithm

```
1. Clamp each phase's raw value to its floor and ceiling. Green/red ceilings
   are the per-TLS dynamic ceiling (above); yellow's ceiling is YELLOW_CEIL (6).
2. Compute remainder = CYCLE_LENGTH - sum(clamped).
3. If remainder != 0:
     a. Among green/red phases, pick the one with the SMALLEST current value.
        Add the full remainder to it.
     b. If that pushes it below its per-type minimum, clip it at its min and
        push the leftover into the LARGEST green/red phase (fallback).
4. Yellow phases never absorb the remainder.
```

Two notes:

- Yellow phases participate in step 1 (their values get clamped to `[3, 6]`) but never in step 3 — they cannot absorb the cycle remainder.
- The smallest-then-largest rule keeps adjustments "balanced": small phases grow when there's slack, big phases shrink when there's overflow.

### Worked example A — 3-phase, simple positive remainder

Raw genes: `[40, 6, 30]` for a `G, y, r` TLS.

| Step | Action | State |
|---|---|---|
| Clamp | All in bounds | `[40, 6, 30]` |
| Sum / remainder | `40+6+30=76`, remainder = `+14` | — |
| Smallest adjustable | green=40, red=30 → red (30) | — |
| Absorb | red = 30 + 14 = 44 (within `[5, 63]`) | `[40, 6, 44]` |

Final: `[40, 6, 44]`, sum 90. ✓

### Worked example B — 4-phase, negative remainder

Raw genes: `[50, 6, 35, 6]` for a `G, y, G, y` TLS with yellow baselines `[6, 6]`,
so the per-green dynamic ceiling is `90 − 12 − 24 = 54` (cycle minus yellows minus
the other green's minimum).

| Step | Action | State |
|---|---|---|
| Clamp | All in bounds (greens ≤ 54, yellows in `[3, 6]`) | `[50, 6, 35, 6]` |
| Sum / remainder | `50+6+35+6=97`, remainder = `−7` | — |
| Smallest adjustable | green1=50, green2=35 → green2 (35) | — |
| Absorb | green2 = 35 − 7 = 28 (within `[24, 54]`) | `[50, 6, 28, 6]` |

Final: `[50, 6, 28, 6]`, sum 90. ✓

### Worked example C — dynamic green ceiling absorbs the overflow

Raw genes: `[82, 6, 50]` for a `G, y, r` TLS with yellow=6 (green ceiling 79,
red ceiling 60).

| Step | Action | State |
|---|---|---|
| Clamp | green 82 → **79** (dynamic ceiling), red 50 ≤ 60 | `[79, 6, 50]` |
| Sum / remainder | `79+6+50=135`, remainder = `−45` | — |
| Smallest adjustable | green=79, red=50 → red (50) | — |
| Absorb | red = 50 − 45 = 5 (within `[5, 60]`) | `[79, 6, 5]` |

Final: `[79, 6, 5]`, sum 90. ✓

Before the dynamic ceiling, green clamped to 82 and the absorption pushed red
below its minimum, forcing a second "fallback to largest" pass. With the
ceiling, green is clamped to 79 up front, so the single smallest-adjustable
pass lands the cycle exactly. The fallback path still exists for the rare
overflow that the ceiling alone cannot resolve, but it fires far less often.

---

## Part 2 — Pair-wise mutation

The pair-wise mutation operator lives in
[pairwise_mutation.py](../novel/pairwise_mutation.py) and is shared by the custom
optimizer. The function that touches durations is
[`mutate_pair_cluster`](../novel/pairwise_mutation.py#L70); it is dispatched by
[`mutate_tree_walk`](../novel/pairwise_mutation.py#L296), which decides *which*
TLS or pair to mutate by walking the Ward tree (see
[`pipeline_custom_optimizer.md`](pipeline_custom_optimizer.md)).

The operator picks a 2-TLS cluster, keeps one TLS intact, and **grows the
other's green time** while pinning its red and freezing its yellow.

### Phase roles per TLS

For each TLS, [`build_phase_split`](../novel/pairwise_mutation.py#L31) classifies
its gene positions:

| Role | Phases | What the operator does |
|---|---|---|
| **green** | green phases | grown (re-sampled to a chosen budget) |
| **red** | red phases | pinned at their minimum (`RED_FLOOR = 5`) |
| **yellow** | yellow phases | frozen — never written |

For our two shapes:

| Shape | green idx | red idx | yellow idx |
|---|---|---|---|
| `G, y, r` | 0 | 2 | 1 |
| `G, y, G, y` | 0, 2 | — | 1, 3 |

### Algorithm

```
1. Pick a random 2-TLS pair (A, B) from the Ward tree's pair_clusters.
2. With 50/50 probability, flip the roles so either A or B can be "first".
3. tls_first  → kept intact; record sum1 = its current GREEN sum.
   tls_second → its green is regrown (below); its red pinned; yellow frozen.
4. Green budget for the second TLS:
       min_sum2 = sum1 + 1                                  # strictly greener than first
       max_sum2 = CYCLE_LENGTH − Σ(second's yellows) − red_min × (second's red count)
5. If min_sum2 < max_sum2 (feasible):
     a. Sample target2 ~ U(min_sum2, max_sum2).
     b. Draw a raw proportion per green phase from U(GREEN_FLOOR, ub_green),
        where ub_green is the second TLS's per-green dynamic ceilings.
     c. Scale them so they sum to target2, then clip to [GREEN_FLOOR, ub_green].
     d. If clipping shrinks the sum below min_sum2, redistribute the deficit
        proportionally across greens with headroom (repeat until satisfied
        or every green sits at its ceiling).
   Else (sum1 + 1 already ≥ max_sum2):
     a. Fallback: split max_sum2 evenly across the green phases (cap at max green).
6. Write the greens into tls_second's green positions; pin its reds at red_min;
   leave yellows untouched.
```

Because the pinned red (5 s) ends up the smallest phase, `normalize_to_cycle`
(Part 1) later absorbs the cycle remainder into **red**, so the green budget set
here survives normalisation unchanged.

### Worked example D — 3-phase TLS as `second`

TLS_A is `first`, with green sum `sum1 = 40`.
TLS_B (3-phase `G, y, r`) is `second`, currently `[60, 6, 24]`.

| Step | Action | Value |
|---|---|---|
| Budget | `min_sum2 = 41`, `max_sum2 = 90 − 6 − 5×1 = 79` → feasible | — |
| Sample | `target2 ~ U(41, 79)` → say `70` | green = 70 |
| Pin red | red → `red_min = 5` | — |
| Freeze yellow | yellow stays `6` | — |
| Result | raw slice | `[70, 6, 5]`, sum = 81 |

Normalisation (Part 1) then sees `[70, 6, 5]`, remainder `90 − 81 = +9`, smallest
adjustable phase is red (5) → red `= 5 + 9 = 14`.
Final stored durations: **green 70 s, yellow 6 s, red 14 s** (sum 90). The green
budget of 70 survived; the slack landed in red.

### Worked example E — fallback when the first TLS already saturates { This is not possible as maximum for 3 phase is 79. this would be automatically adjusted.}

TLS_A is `first` with a large green sum, `sum1 = 80`. 
TLS_B (3-phase) is `second`, yellow `6`, one red.

`min_sum2 = 81`, `max_sum2 = 90 − 6 − 5 = 79`. Here `min_sum2 ≥ max_sum2`, so the
feasible branch is skipped: the second TLS's green is split evenly across its
green phases at `max_sum2 = 79` (one green → 79). Red pinned at 5, yellow 6 →
`[79, 6, 5]`, already exactly 90.

### Edge cases

| Situation | Behaviour |
|---|---|
| `pair_clusters` is empty / no valid pair | Return `sol` unchanged (no-op). |
| Either TLS in the pair is not in the gene map | Skip that pair via the `valid` filter. |
| `tls_second` has no green phase to grow | Return `sol` unchanged. |
| `sum1 + 1 ≥ max_sum2` (first TLS saturates the green budget) | Fallback: split `max_sum2` evenly across the second's green phases. |
| Clipping shrinks the green sum below `min_sum2` | Deficit loop redistributes proportionally across greens with headroom until satisfied or every green is at its ceiling (`ub_green`). |

---

## How the two parts fit together

```
              ┌────────────────────────┐
              │   pair-wise mutation   │
              │  (raw gene values, not │
              │   constrained to 90s)  │
              └───────────┬────────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │   normalize_to_cycle   │
             │   (clamp + 90s rule)   │
             └───────────┬────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       SUMO simulation         _rebuild_json
       (fitness eval)          (saved JSON)
```

Same input → same function → same output. Whatever durations SUMO simulated for the best individual are exactly the durations that end up in the saved JSON.
