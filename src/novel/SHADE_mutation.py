"""
SHADE step-pairwise-mutation operator.

A near-exact variant of ``src.novel.pairwise_mutation.mutate_pair_cluster``.
The ONLY behavioural difference is how the second TLS's green budget
(``target2``) is chosen:

    original : target2 ~ U(sum1 + 1, max_sum2)        (random)
    here     : target2 = min(sum1 + step_size, max_sum2)   (fixed step)

Everything else is preserved verbatim so the operator keeps the same
invariant as the original — **the second TLS always ends up strictly
greener than the first** — via the ``min_sum2 = sum1 + 1`` floor plus the
deficit-redistribution loop.  The per-TLS green/red/yellow ``phase_split`` is
reused from ``src.novel.pairwise_mutation`` (not re-implemented here).

Usage: imported by ``src.algorithms.differential_evolution_cluster_v3`` when
``SHADE_PAIRWISE_MUTATION`` is enabled.
"""
import sys
from pathlib import Path

import numpy as np

# Project root on sys.path so ``config`` / ``src...`` imports resolve when this
# module is loaded from anywhere.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import GREEN_FLOOR, CYCLE_LENGTH, RED_FLOOR

# Re-export the shared phase-split builder so callers can import everything
# pairwise from one place if they wish; the operator below consumes its output.
from src.novel.pairwise_mutation import build_phase_split  # noqa: F401


def mutate_pair_cluster_step(
    sol: np.ndarray,
    pair_clusters: list[tuple[str, str]],
    tls_to_genes: dict[str, tuple[int, int]],
    phase_split: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    ub: np.ndarray,
    step_size: float,
) -> np.ndarray:
    """
    Step variant of the novel pair-cluster mutation — grows the second TLS's
    **green** time to ``first_green_sum + step_size`` (capped at the cycle's
    physical green ceiling).

    Identical to ``mutate_pair_cluster`` except that the second TLS's green
    budget is a fixed step above the first's green sum rather than a random
    draw.  Yellow phases stay frozen, red phases are pinned at ``RED_FLOOR``,
    and ``normalize_to_cycle`` later restores the exact 90 s cycle.

    Parameters
    ----------
    sol           : current solution vector — returned as a new copy
    pair_clusters : list of (tls_id_a, tls_id_b) from the full Ward tree
    tls_to_genes  : maps tls_id -> (start_idx, end_idx) in the gene vector
    phase_split   : maps tls_id -> {"green"/"red"/"yellow"/"mutable": indices}
    rng           : numpy random Generator
    ub            : per-gene dynamic upper bounds (green/red ceilings)
    step_size     : fixed green increment added to the first TLS's green sum

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
    #   max_sum2 = cycle − frozen yellows − reserved minimum red.
    red_min     = float(RED_FLOOR)
    yellow_sum2 = float(sol[yellow2].sum()) if yellow2.size else 0.0
    max_sum2    = float(CYCLE_LENGTH) - yellow_sum2 - red_min * red2.size
    min_sum2    = sum1 + 1.0          # strictly greener than the first TLS

    if min_sum2 < max_sum2:
        # Fixed step above the first TLS's green sum, capped at max_sum2.
        target2 = min(sum1 + float(step_size), max_sum2)

        # Draw random proportions and scale to the green budget
        raw    = rng.uniform(GREEN_FLOOR, ub_green, ng)
        greens = raw / raw.sum() * target2
        greens = np.clip(greens, GREEN_FLOOR, ub_green)

        # ── Hard enforcement: GREEN_FLOOR clipping can shrink the sum below the
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
            greens  = np.clip(greens, GREEN_FLOOR, ub_green)
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
