"""
IRRG - Incremental Recursive Ranking Grouping for Traffic Light Optimization
=============================================================================

WHAT THIS FILE DOES
--------------------
This file implements a black-box variable interaction detection algorithm called
Incremental Recursive Ranking Grouping (IRRG). Its job is to figure out which
traffic light phase durations interact with each other — meaning that changing
one of them changes the optimal value of another — and which ones are independent.

Once we know the grouping structure, a downstream cooperative co-evolution (CC)
optimizer can tackle each group separately. Optimizing 10 variables at a time is
vastly easier than optimizing 500 simultaneously, even with the same total FFE
budget.

HOW MONOTONICITY CHECKING WORKS
---------------------------------
Two variables x_p and x_q are considered interacting if we can find a situation
where perturbing x_q changes the relative ordering of fitness values when x_p
takes two different values. Concretely: if f(..., a1, ..., b1, ...) <= f(..., a2, ..., b1, ...)
but f(..., a1, ..., b2, ...) > f(..., a2, ..., b2, ...), then x_p and x_q interact.

IRRG extends this to groups of variables and uses rankings of ns fitness
evaluations instead of single pairs, making it far more sensitive to weak
interactions.

THE TRAFFIC LIGHT PROBLEM
--------------------------
Each decision variable is the duration (in seconds) of one traffic light phase
at one intersection in the SUMO network. The fitness function submits a full
configuration vector to SUMO and receives back the mean vehicle delay across
the network. We minimise this delay.

Variable bounds: each phase duration is constrained between lb[i] and ub[i],
typically 5 to 60 seconds. These are derived from the baseline traffic data
via build_traffic_fitness_wrapper() from DG2_grouping.py.

WHAT THE BASELINE DATA PROVIDES
----------------------------------
  Via build_traffic_fitness_wrapper():
  - n          : int, total number of decision variables (phase durations)
  - lb         : np.ndarray of shape (n,), lower bounds per variable (phase-type aware)
  - ub         : np.ndarray of shape (n,), upper bounds per variable (phase-type aware)
  - labels     : list of str, gene names (e.g. "186797066_phase_1")
  - wrapper    : TrafficFitnessWrapper, picklable fitness callable for parallel eval

  theta_init is set to the identity matrix (no warm start from DG2).

WHAT THE OUTPUT MEANS
-----------------------
  - Seps   : list of lists of variable indices. Each inner list is a group of
             variables that appear separable. They are chunked into groups of
             size `s` for efficient co-evolution with CMA-ES.
  - NonSeps: list of lists of variable indices. Each inner list is a group of
             variables that interact with each other and must be optimised together.

TUNABLE PARAMETERS
-------------------
  ns          = 10    Number of samples per ranking. Higher = more sensitive but
                      more expensive. 10 works well for most LSGO problems.
  sti         = 15    Stale iteration threshold. IRRG stops after this many
                      consecutive RRG calls that find no new interactions.
  s           = 100   Preferred group size for separable variables.
  shade_fes   = 5000  FFEs given to SHADE global search to find x_hq.
  mts_fes     = 15000 FFEs given to MTS-LS1 local search to refine x_hq.
"""


import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Add project root to sys.path to import config and other modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import BASELINE_TRAFFIC_DATA, NUM_PROCESSORS
from src.genetic_algorithm.fitness_evaluation import fitness_function as _traffic_fitness
from src.pygad.DG2_grouping import build_traffic_fitness_wrapper

# ---------------------------------------------------------------------------
# CONSTANTS AND CONFIG
# ---------------------------------------------------------------------------

# IEEE 754 double-precision machine epsilon. Used for automatic threshold
# estimation so we never mistake floating-point rounding noise for a real
# fitness difference.
MU_M: float = float(np.finfo(np.float64).eps)


@dataclass
class IRRGConfig:
    """
    All tunable parameters for IRRG in one place.

    Attributes
    ----------
    ns : int
        Number of samples used to build each ranking. A ranking is an ordered
        list of ns fitness evaluations that sweep the domain of a group of
        variables. Higher ns catches weaker interactions but costs more FFEs.
    sti : int
        Stale iteration threshold. IRRG keeps calling RRG until this many
        consecutive calls produce zero new interactions.
    s : int
        Preferred size of separable variable groups. Separable variables are
        chunked into groups of this size before being handed to CMA-ES, which
        performs best on groups up to ~100 variables.
    shade_fes : int
        FFE budget for the SHADE global optimizer used to find a high-quality
        starting solution x_hq before interaction detection begins.
    mts_fes : int
        FFE budget for the MTS-LS1 local search used to refine x_hq after SHADE.
    seed : int
        Random seed for full reproducibility.
    log_level : str
        Python logging level string: "DEBUG", "INFO", "WARNING", etc.
    """
    ns: int = 10
    sti: int = 15
    s: int = 100
    shade_fes: int = 5000
    mts_fes: int = 15000
    seed: int = 42
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# FITNESS FUNCTION
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MODULE-LEVEL WORKER — must be plain function for ProcessPoolExecutor pickling
# ---------------------------------------------------------------------------

def _eval_single(args):
    """
    Evaluate a single vector with the traffic fitness wrapper.

    Must be a module-level function so ProcessPoolExecutor can pickle it.
    Each worker process gets its own SUMO instance.

    Parameters
    ----------
    args : (TrafficFitnessWrapper, np.ndarray)
        The picklable fitness wrapper and the candidate vector.

    Returns
    -------
    float
        Fitness value (total delay).
    """
    wrapper, vector = args
    return float(wrapper(vector))


class IRRGFitnessWrapper:
    """
    Wrapper around TrafficFitnessWrapper with FFE tracking and batch evaluation.

    IRRG's core algorithm reads ``f.fes`` to log how many fitness function
    evaluations have been consumed.  The underlying TrafficFitnessWrapper
    (from DG2_grouping.py) does not track this, so we add a counter here.

    For parallel evaluation, ``batch_evaluate`` dispatches all vectors to a
    ProcessPoolExecutor in one shot and increments the counter by the batch
    size on return.

    Parameters
    ----------
    traffic_wrapper : TrafficFitnessWrapper
        Picklable fitness callable built by ``build_traffic_fitness_wrapper``.
    lb : np.ndarray
        Lower bounds per gene.
    ub : np.ndarray
        Upper bounds per gene.
    """

    def __init__(self, traffic_wrapper, lb: np.ndarray, ub: np.ndarray) -> None:
        self.traffic_wrapper = traffic_wrapper
        self.lb = lb.copy()
        self.ub = ub.copy()
        self.n = len(lb)
        self.fes: int = 0

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate a single vector (sequential). Increments FFE counter."""
        x = np.clip(x, self.lb, self.ub)
        self.fes += 1
        return float(self.traffic_wrapper(x))

    def batch_evaluate(self, vectors: List[np.ndarray], pool: ProcessPoolExecutor) -> List[float]:
        """
        Evaluate multiple vectors in parallel using pool.map.

        Each worker process gets its own SUMO instance.  The FFE counter
        is incremented by the batch size in the main process.

        Parameters
        ----------
        vectors : list of np.ndarray
            Candidate solutions to evaluate.
        pool : ProcessPoolExecutor
            Pre-created process pool.

        Returns
        -------
        list of float
            Fitness values in the same order as the input vectors.
        """
        tasks = [(self.traffic_wrapper, np.clip(v, self.lb, self.ub)) for v in vectors]
        results = list(pool.map(_eval_single, tasks))
        self.fes += len(results)
        return results

    def reset_counter(self) -> None:
        """Reset the FFE counter to zero."""
        self.fes = 0

    def __repr__(self) -> str:
        return f"IRRGFitnessWrapper(n={self.n}, fes={self.fes})"


# ---------------------------------------------------------------------------
# MATH UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def f_gamma(k: float) -> float:
    """
    Compute the automatic epsilon scaling factor for floating-point comparisons.

    This factor is derived from IEEE 754 arithmetic. When we compare two fitness
    values f1 and f2, we can't trust that f1 != f2 is a real difference — it
    might just be floating-point rounding noise accumulated over many operations.
    This function gives us a scale-relative threshold: differences smaller than
    f_gamma(k) * (|f1| + |f2|) are treated as zero.

    Parameters
    ----------
    k : float
        Scaling argument, typically sqrt(n) + 1 or sqrt(n) + 2 where n is the
        problem dimension.

    Returns
    -------
    float
        The epsilon scaling factor.
    """
    return (k * MU_M) / (1.0 - k * MU_M)


def sgn_epsilon(x: float, epsilon: float) -> int:
    """
    Epsilon-aware sign function for robust floating-point comparisons.

    Instead of checking x < 0, x == 0, x > 0, we use an epsilon band around
    zero. Values within [-epsilon, +epsilon] are treated as equal (return 0).
    This prevents floating-point noise from being misread as a fitness ordering
    change.

    Parameters
    ----------
    x : float
        The value to sign-test.
    epsilon : float
        The tolerance band. Values with |x| <= epsilon are treated as zero.

    Returns
    -------
    int
        -1 if x < -epsilon, 0 if |x| <= epsilon, +1 if x > epsilon.
    """
    if x < -epsilon:
        return -1
    elif x > epsilon:
        return 1
    else:
        return 0


def compute_epsilon(y1: float, y2: float, n: int) -> float:
    """
    Compute the interaction-check epsilon for a pair of fitness values.

    The epsilon is scale-relative: it grows with the magnitude of the fitness
    values being compared. This is important because SUMO delays can range
    from near-zero to thousands of seconds depending on network size.

    Parameters
    ----------
    y1 : float
        First fitness value.
    y2 : float
        Second fitness value.
    n : int
        Problem dimension (used to set the scaling factor k = sqrt(n) + 1).

    Returns
    -------
    float
        The epsilon threshold for this particular comparison.
    """
    k = math.sqrt(n) + 1.0
    return f_gamma(k) * (abs(y1) + abs(y2))


def enforce_transitivity(theta: np.ndarray) -> np.ndarray:
    """
    Enforce transitivity in the binary interaction matrix.

    If variable i interacts with j, and j interacts with k, then i must also
    interact with k. Without this, groups can be fragmented: i-j and j-k might
    be detected in different RRG iterations, but never merged into one group
    {i, j, k}.

    We enforce transitivity by repeatedly OR-multiplying the matrix with itself
    (treating 1 as True and 0 as False) until no more changes occur. This is
    equivalent to computing the transitive closure of the interaction graph.

    Parameters
    ----------
    theta : np.ndarray of shape (n, n), dtype bool or int
        Binary interaction matrix. theta[i][j] == 1 means i and j interact.

    Returns
    -------
    np.ndarray
        Updated matrix with transitivity enforced. Modifies and returns the
        input array in-place as well.
    """
    n = theta.shape[0]
    # Work with boolean for speed
    t = theta.astype(bool)
    while True:
        # OR-multiply: t_new[i,k] = any j such that t[i,j] and t[j,k]
        t_new = t | (t @ t).astype(bool)
        if np.array_equal(t_new, t):
            break
        t = t_new
    theta[:] = t.astype(theta.dtype)
    return theta


# ---------------------------------------------------------------------------
# RANKING FUNCTIONS
# ---------------------------------------------------------------------------

def build_sample_matrix(
    lb: np.ndarray,
    ub: np.ndarray,
    ns: int,
    n: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Build the ns x n sample matrix X1_bar used throughout RRG.

    For each variable j, we generate ns values that are evenly spaced between
    lb[j] and ub[j], then shuffle them independently. This gives us diverse,
    unbiased coverage of every variable's feasible domain.

    The shuffling is per-column so that the ns rows of X1_bar are not correlated
    across variables — otherwise, we might systematically evaluate only "good"
    or only "bad" combinations, which would bias the rankings.

    Parameters
    ----------
    lb : np.ndarray of shape (n,)
        Lower bounds.
    ub : np.ndarray of shape (n,)
        Upper bounds.
    ns : int
        Number of samples (rows).
    n : int
        Problem dimension (columns).
    rng : np.random.Generator
        Random generator for reproducible shuffling.

    Returns
    -------
    np.ndarray of shape (ns, n)
        Sample matrix. Row i gives the values assigned to all variables in
        one evaluation.
    """
    X1_bar = np.zeros((ns, n), dtype=np.float64)
    for j in range(n):
        # ns evenly-spaced values across [lb[j], ub[j]]
        values = np.linspace(lb[j], ub[j], ns)
        # Shuffle this column independently
        rng.shuffle(values)
        X1_bar[:, j] = values
    return X1_bar


def create_first_ranking(
    X1_indices: List[int],
    x_hq: np.ndarray,
    X1_bar: np.ndarray,
    f: IRRGFitnessWrapper,
    ns: int,
    pool: Optional[ProcessPoolExecutor] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate ns fitness values by sweeping the domain of X1 variables and
    return a ranking of those evaluations.

    We start from the high-quality solution x_hq and replace the values of
    variables in X1_indices with each row of X1_bar in turn. Everything outside
    X1_indices stays at x_hq. This creates a baseline: how does fitness vary
    as we move X1 variables around, holding everything else fixed at a good point?

    The resulting ranking r1 tells us the order of those ns configurations from
    best (lowest delay) to worst. Later, create_second_ranking_and_check will
    perturb the X2 group and see if that ordering flips — a flip means interaction.

    Parameters
    ----------
    X1_indices : List[int]
        Indices of variables in group X1.
    x_hq : np.ndarray of shape (n,)
        High-quality solution used as the base vector.
    X1_bar : np.ndarray of shape (ns, n)
        Sample matrix (all variables, all samples).
    f : IRRGFitnessWrapper
        Fitness oracle with FFE tracking and batch evaluation.
    ns : int
        Number of samples.
    pool : ProcessPoolExecutor, optional
        If provided, evaluations are dispatched in parallel via pool.map.

    Returns
    -------
    y1_bar : np.ndarray of shape (ns,)
        Fitness values for each of the ns sample configurations.
    r1 : np.ndarray of shape (ns,)
        Indices into y1_bar sorted ascending (best = lowest delay first).
    """
    # Build all ns candidate vectors up front
    vectors = []
    for i in range(ns):
        x_tmp = x_hq.copy()
        x_tmp[X1_indices] = X1_bar[i, X1_indices]
        vectors.append(x_tmp)

    # Evaluate in parallel if pool is available, otherwise sequentially
    if pool is not None:
        y1_bar = np.array(f.batch_evaluate(vectors, pool), dtype=np.float64)
    else:
        y1_bar = np.array([f(v) for v in vectors], dtype=np.float64)

    # r1[0] is the index of the best (lowest) fitness, r1[-1] is the worst
    r1 = np.argsort(y1_bar)
    return y1_bar, r1


def create_second_ranking_and_check(
    X1_indices: List[int],
    X2_indices: List[int],
    x_hq: np.ndarray,
    X1_bar: np.ndarray,
    x2_bar: np.ndarray,
    y1_bar: np.ndarray,
    r1: np.ndarray,
    f: IRRGFitnessWrapper,
    ns: int,
    n: int,
    pool: Optional[ProcessPoolExecutor] = None
) -> bool:
    """
    Build a second ranking after perturbing X2 variables and check if the
    ordering flipped relative to the first ranking.

    This is the core interaction detection step. We already know how ns
    configurations of X1 rank against each other when X2 is held at x_hq
    (that's r1 / y1_bar). Now we set X2 to a different point (x2_bar) and
    re-evaluate the same ns X1 configurations. If the relative ordering of any
    consecutive pair changes sign — specifically, if a pair that was better
    under x_hq becomes worse under x2_bar — then X1 and X2 are interacting.

    All ns evaluations are dispatched in parallel when a pool is provided.
    The flip check is then performed sequentially on the collected results.

    Parameters
    ----------
    X1_indices : List[int]
        Indices of variables in group X1.
    X2_indices : List[int]
        Indices of variables in group X2 (the group being tested for interaction).
    x_hq : np.ndarray of shape (n,)
        High-quality base solution.
    X1_bar : np.ndarray of shape (ns, n)
        Sample matrix.
    x2_bar : np.ndarray of shape (n,)
        Alternative solution; its X2 values replace x_hq's X2 values.
    y1_bar : np.ndarray of shape (ns,)
        Fitness values from the first ranking.
    r1 : np.ndarray of shape (ns,)
        Sorting indices from the first ranking (ascending order).
    f : IRRGFitnessWrapper
        Fitness oracle with FFE tracking and batch evaluation.
    ns : int
        Number of samples.
    n : int
        Problem dimension (for epsilon computation).
    pool : ProcessPoolExecutor, optional
        If provided, evaluations are dispatched in parallel via pool.map.

    Returns
    -------
    bool
        True if an ordering flip was detected (interaction exists), False otherwise.
    """
    # Build all ns vectors for the second ranking (X2 set to alternative)
    x_base = x_hq.copy()
    x_base[X2_indices] = x2_bar[X2_indices]

    vectors = []
    for i in range(ns):
        x_tmp = x_base.copy()
        x_tmp[X1_indices] = X1_bar[r1[i], X1_indices]
        vectors.append(x_tmp)

    # Evaluate all ns vectors in parallel if pool is available
    if pool is not None:
        y2_values = f.batch_evaluate(vectors, pool)
    else:
        y2_values = [f(v) for v in vectors]

    # Check for ordering flips sequentially on collected results
    for i in range(1, ns):
        # --- Check if the first-ranking gap between r1[i-1] and r1[i] is real ---
        eps1 = compute_epsilon(y1_bar[r1[i]], y1_bar[r1[i - 1]], n)
        if sgn_epsilon(y1_bar[r1[i]] - y1_bar[r1[i - 1]], eps1) == 0:
            # The two first-ranking values are indistinguishable — skip this pair
            continue

        # --- Check if the ordering flipped ---
        eps2 = compute_epsilon(y2_values[i], y2_values[i - 1], n)
        if sgn_epsilon(y2_values[i] - y2_values[i - 1], eps2) < 0:
            # y2_values[i] < y2_values[i-1] means r1[i] is now BETTER than
            # r1[i-1] under x2_bar, but r1[i] was WORSE under x_hq.
            # The ordering flipped — interaction!
            return True

    return False


# ---------------------------------------------------------------------------
# INTERACTION DETECTION FUNCTIONS
# ---------------------------------------------------------------------------

def consider_variables(
    V: List[int],
    G: List[List[int]],
    x_hq: np.ndarray,
    X1_bar: np.ndarray,
    x2_bar: np.ndarray,
    f: IRRGFitnessWrapper,
    ns: int,
    n: int,
    rng: np.random.Generator,
    pool: Optional[ProcessPoolExecutor] = None
) -> bool:
    """
    Decide whether the currently-separable variables V are worth including in
    this round of interaction search.

    Running full interaction checks for V against everything is expensive. This
    function does a cheap preliminary check: if there's any evidence that V
    variables might interact with something, return True so RRG adds them to G.
    If they're definitely isolated, return False to skip them this round.

    Four conditions trigger True (any one is sufficient):

    Condition A — No groups found yet:
        G is empty, so we have nothing to compare against. We must include V.

    Condition B — V has exactly one variable:
        Single variables are cheap to check; always include them.

    Condition C — V variables interact with each other:
        Randomly split V into two halves V1 and V2. Check if V1 interacts with
        V2 or vice versa. If yes, V contains variables that are non-separable
        among themselves and must be included.

    Condition D — V interacts with an existing group:
        For each group g in G, check if V interacts with g or g interacts with V.
        If any interaction is found, V variables are linked to an existing group
        and must be included.

    Parameters
    ----------
    V : List[int]
        Indices of variables not yet assigned to any group.
    G : List[List[int]]
        Current list of variable groups (each group is a list of indices).
    x_hq : np.ndarray
        High-quality solution.
    X1_bar : np.ndarray of shape (ns, n)
        Sample matrix.
    x2_bar : np.ndarray
        Alternative solution for second ranking.
    f : IRRGFitnessWrapper
        Fitness oracle.
    ns : int
        Number of samples.
    n : int
        Problem dimension.
    rng : np.random.Generator
        For reproducible shuffling.
    pool : ProcessPoolExecutor, optional
        If provided, evaluations are dispatched in parallel.

    Returns
    -------
    bool
        True if V should be included in the current interaction search round.
    """
    # Condition A: no groups exist yet
    if len(G) == 0:
        return True

    # Guard: nothing in V
    if len(V) == 0:
        return False

    # Condition B: single variable always included
    if len(V) == 1:
        return True

    # Condition C: check V against itself (split into two halves)
    V_shuffled = list(V)
    rng.shuffle(V_shuffled)
    mid = len(V_shuffled) // 2
    V1 = V_shuffled[:mid]
    V2 = V_shuffled[mid:]

    y1_bar, r1 = create_first_ranking(V1, x_hq, X1_bar, f, ns, pool)
    if create_second_ranking_and_check(V1, V2, x_hq, X1_bar, x2_bar, y1_bar, r1, f, ns, n, pool):
        return True

    y1_bar, r1 = create_first_ranking(V2, x_hq, X1_bar, f, ns, pool)
    if create_second_ranking_and_check(V2, V1, x_hq, X1_bar, x2_bar, y1_bar, r1, f, ns, n, pool):
        return True

    # Condition D: check V against each existing group
    for g in G:
        y1_bar, r1 = create_first_ranking(V, x_hq, X1_bar, f, ns, pool)
        if create_second_ranking_and_check(V, g, x_hq, X1_bar, x2_bar, y1_bar, r1, f, ns, n, pool):
            return True

        y1_bar, r1 = create_first_ranking(g, x_hq, X1_bar, f, ns, pool)
        if create_second_ranking_and_check(g, V, x_hq, X1_bar, x2_bar, y1_bar, r1, f, ns, n, pool):
            return True

    return False


def interact(
    G1: List[List[int]],
    G2: List[List[int]],
    x_hq: np.ndarray,
    X1_bar: np.ndarray,
    x2_bar: np.ndarray,
    y1_bar: np.ndarray,
    r1: np.ndarray,
    f: IRRGFitnessWrapper,
    ns: int,
    n: int,
    pool: Optional[ProcessPoolExecutor] = None
) -> List[List[int]]:
    """
    Find which groups in G2 interact with the combined group G1 and return
    an extended version of G1 containing those groups.

    We first flatten G1 and G2 into single variable lists and check if they
    interact at all. If not, we return G1 unchanged. If they do, we recursively
    bisect G2 to find exactly which sub-groups are responsible. This binary
    search structure gives O(log |G2|) interaction checks instead of O(|G2|).

    Example:
        G1 = [[0,1,2]], G2 = [[3,4], [5,6], [7,8], [9,10]]
        We first check {0,1,2} vs {3,4,5,6,7,8,9,10} — interaction found.
        Then check {0,1,2} vs {3,4,5,6} — interaction found.
        Then check {0,1,2} vs {3,4} — interaction found. Add [3,4] to G1.
        Then check {0,1,2,3,4} vs {5,6} — no interaction.
        Then check {0,1,2} vs {7,8,9,10} — no interaction.
        Result: G1 = [[0,1,2], [3,4]]

    Parameters
    ----------
    G1 : List[List[int]]
        Current "seed" group (list of groups, typically one group at start).
    G2 : List[List[int]]
        Candidate groups to test for interaction with G1.
    x_hq : np.ndarray
        High-quality base solution.
    X1_bar : np.ndarray of shape (ns, n)
        Sample matrix.
    x2_bar : np.ndarray
        Alternative solution.
    y1_bar : np.ndarray of shape (ns,)
        First ranking values (pre-computed for flattened G1).
    r1 : np.ndarray of shape (ns,)
        First ranking indices (pre-computed for flattened G1).
    f : IRRGFitnessWrapper
        Fitness oracle.
    ns : int
        Number of samples.
    n : int
        Problem dimension.
    pool : ProcessPoolExecutor, optional
        If provided, evaluations are dispatched in parallel.

    Returns
    -------
    List[List[int]]
        Extended G1 containing all groups from G2 that interact with original G1.
    """
    # Start with G1 as-is; we'll add to it
    G1_star = list(G1)

    # Flatten both sides into flat variable lists for the interaction check
    X1_flat = [idx for group in G1 for idx in group]
    X2_flat = [idx for group in G2 for idx in group]

    # Check if the two flat groups interact
    if not create_second_ranking_and_check(
        X1_flat, X2_flat, x_hq, X1_bar, x2_bar, y1_bar, r1, f, ns, n, pool
    ):
        # No interaction between G1 and G2 at all
        return G1_star

    # Interaction found — if G2 has just one group, add it directly
    if len(G2) == 1:
        G1_star.append(G2[0])
        return G1_star

    # Bisect G2 and recurse on each half
    mid = len(G2) // 2
    G2_left = G2[:mid]
    G2_right = G2[mid:]

    G1_star = interact(G1_star, G2_left, x_hq, X1_bar, x2_bar, y1_bar, r1, f, ns, n, pool)
    G1_star = interact(G1_star, G2_right, x_hq, X1_bar, x2_bar, y1_bar, r1, f, ns, n, pool)

    return G1_star


# ---------------------------------------------------------------------------
# RRG — RECURSIVE RANKING GROUPING
# ---------------------------------------------------------------------------

def rrg(
    x_hq: np.ndarray,
    x2_bar: np.ndarray,
    theta: np.ndarray,
    f: IRRGFitnessWrapper,
    n: int,
    lb: np.ndarray,
    ub: np.ndarray,
    ns: int,
    rng: np.random.Generator,
    logger: logging.Logger,
    pool: Optional[ProcessPoolExecutor] = None
) -> List[List[int]]:
    """
    One full pass of Recursive Ranking Grouping.

    RRG tries to discover all variable interactions in a single sweep using
    recursive binary search. IRRG calls this multiple times, each time with a
    different random x2_bar, so that weak interactions missed in one pass have
    a chance to be caught in the next.

    Parameters
    ----------
    x_hq : np.ndarray of shape (n,)
        High-quality solution (near a local optimum). Interaction detection is
        most sensitive near optima because small perturbations matter more there.
    x2_bar : np.ndarray of shape (n,)
        Random alternative solution. Its values replace x_hq for the X2 group
        when building the second ranking.
    theta : np.ndarray of shape (n, n)
        Current interaction matrix. Used to pre-populate groups G from previously
        discovered interactions.
    f : IRRGFitnessWrapper
        Fitness oracle.
    n : int
        Problem dimension.
    lb : np.ndarray of shape (n,)
        Lower bounds.
    ub : np.ndarray of shape (n,)
        Upper bounds.
    ns : int
        Number of samples per ranking.
    rng : np.random.Generator
        Random generator.
    logger : logging.Logger
        For progress logging.

    Returns
    -------
    List[List[int]]
        NonSeps: list of discovered interacting variable groups.
    """
    NonSeps: List[List[int]] = []

    # ------------------------------------------------------------------
    # Step 1 — Initialise G and V from the current interaction matrix
    # ------------------------------------------------------------------
    # G holds groups of variables already known to interact (from theta).
    # V holds variables not yet assigned to any group.
    G: List[List[int]] = []
    assigned: set = set()

    for v in range(n):
        if v in assigned:
            continue
        # Find all variables that theta says interact with v
        interacting = [j for j in range(n) if theta[v, j] == 1 and j != v]
        if interacting:
            group = [v] + interacting
            group = list(set(group))  # deduplicate
            group_shuffled = list(group)
            rng.shuffle(group_shuffled)
            G.append(group_shuffled)
            assigned.update(group_shuffled)

    V: List[int] = [v for v in range(n) if v not in assigned]

    logger.debug(f"RRG init: {len(G)} existing groups, {len(V)} unassigned variables")

    # ------------------------------------------------------------------
    # Step 2 — Build the sample matrix for this RRG pass
    # ------------------------------------------------------------------
    X1_bar = build_sample_matrix(lb, ub, ns, n, rng)

    # ------------------------------------------------------------------
    # Step 3 — Decide if unassigned variables V are worth checking
    # ------------------------------------------------------------------
    if consider_variables(V, G, x_hq, X1_bar, x2_bar, f, ns, n, rng, pool):
        # Add each unassigned variable as its own singleton group
        for v in V:
            G.append([v])
        logger.debug(f"consider_variables returned True — added {len(V)} singletons to G")
    else:
        logger.debug("consider_variables returned False — skipping V this pass")

    if len(G) == 0:
        return NonSeps

    # ------------------------------------------------------------------
    # Step 4 — Initialise group queues
    # ------------------------------------------------------------------
    G_shuffled = list(G)
    rng.shuffle(G_shuffled)

    # G1 is the "seed" — the group we're trying to extend by finding interacting groups
    G1: List[List[int]] = [G_shuffled[0]]
    G2: List[List[int]] = G_shuffled[1:]

    # ------------------------------------------------------------------
    # Step 5 — Main interaction discovery loop
    # ------------------------------------------------------------------
    while len(G2) > 0:

        # Create first ranking for the current flattened G1
        X1_flat = [idx for group in G1 for idx in group]
        y1_bar, r1 = create_first_ranking(X1_flat, x_hq, X1_bar, f, ns, pool)

        # Try to extend G1 by finding interacting groups in G2
        G1_star = interact(G1, G2, x_hq, X1_bar, x2_bar, y1_bar, r1, f, ns, n, pool)

        if len(G1_star) == len(G1):
            # --- G1 was NOT extended ---
            if len(G1) == 1:
                # G1 is a single group. Try the missing-linkage mitigation:
                # if G1's group is large, some G2 variables might be getting
                # "drowned out" by G1's dominant effect on rankings. Shrink G1
                # by randomly removing half its variables so weaker signals emerge.
                min_g2_size = min(len(g) for g in G2)
                G1_single = G1[0]
                if len(G1_single) >= max(min_g2_size, 2):
                    # Remove half of G1's variables randomly
                    keep_n = len(G1_single) // 2
                    kept = list(rng.choice(G1_single, size=keep_n, replace=False))
                    logger.debug(
                        f"Missing-linkage mitigation: shrinking G1 from "
                        f"{len(G1_single)} to {len(kept)} variables"
                    )
                    G1 = [kept]
                else:
                    # G1 is small and still not connecting — move on
                    G1 = [G2[0]]
                    G2 = G2[1:]
            else:
                # G1 has multiple groups — we found interactions before but can't
                # extend further. Commit this cluster and start fresh.
                NonSeps.append([idx for group in G1 for idx in group])
                G1 = [G2[0]]
                G2 = G2[1:]

        else:
            # --- G1 WAS extended ---
            # Remove newly-added groups from G2
            new_groups = G1_star[len(G1):]
            G2 = [g for g in G2 if g not in new_groups]
            G1 = G1_star

    # ------------------------------------------------------------------
    # Step 6 — Finalise: if G1 ended with multiple groups, commit them
    # ------------------------------------------------------------------
    if len(G1) > 1:
        NonSeps.append([idx for group in G1 for idx in group])

    logger.debug(f"RRG found {len(NonSeps)} interacting groups this pass")
    return NonSeps


# ---------------------------------------------------------------------------
# IRRG — INCREMENTAL RECURSIVE RANKING GROUPING
# ---------------------------------------------------------------------------

def irrg(
    f: IRRGFitnessWrapper,
    n: int,
    lb: np.ndarray,
    ub: np.ndarray,
    theta_init: np.ndarray,
    config: IRRGConfig,
    rng: np.random.Generator,
    logger: logging.Logger,
    n_workers: Optional[int] = None
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Full IRRG algorithm: incrementally run RRG until no new interactions are found.

    IRRG calls RRG repeatedly with different random x2_bar vectors. Each call
    gets a fresh perspective on the fitness landscape. Interactions missed in
    one pass (because x2_bar happened to be in an insensitive region) may be
    caught in the next pass.

    The algorithm terminates when `sti` consecutive RRG calls produce zero new
    interactions, or immediately if the very first call produces none (in which
    case repeating is pointless).

    Parameters
    ----------
    f : IRRGFitnessWrapper
        Fitness oracle (wraps SUMO via TrafficFitnessWrapper).
    n : int
        Problem dimension.
    lb : np.ndarray of shape (n,)
        Lower bounds.
    ub : np.ndarray of shape (n,)
        Upper bounds.
    theta_init : np.ndarray of shape (n, n)
        Initial interaction matrix (identity = no warm start).
    config : IRRGConfig
        Algorithm parameters.
    rng : np.random.Generator
        Random generator.
    logger : logging.Logger
        For progress logging.
    n_workers : int, optional
        Number of parallel worker processes.  Defaults to os.cpu_count().

    Returns
    -------
    Seps : List[List[int]]
        Groups of separable variables, each of size config.s (last group may
        be smaller). Feed these to CMA-ES as independent subproblems.
    NonSeps : List[List[int]]
        Groups of interacting variables. Each group must be optimised together.
    """

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    logger.info(f"  Using {n_workers} parallel workers for fitness evaluations.")

    # Create a single process pool for the entire IRRG run
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        # ------------------------------------------------------------------
        # Step 1 — Find a high-quality solution x_hq
        # ------------------------------------------------------------------
        logger.info("Step 1: Running SHADE global search...")
        fes_before_init = f.fes
        x_hq = run_shade(f, n, lb, ub, config.shade_fes, rng, logger, pool)
        logger.info(f"  SHADE done. FFEs used: {f.fes - fes_before_init}. Best x_hq fitness: {f(x_hq):.4f}")

        logger.info("Step 2: Refining x_hq with MTS-LS1 local search...")
        fes_before_mts = f.fes
        x_hq = run_mts_ls1(f, x_hq, lb, ub, config.mts_fes, rng, logger)
        logger.info(f"  MTS-LS1 done. FFEs used: {f.fes - fes_before_mts}. Refined fitness: {f(x_hq):.4f}")

        # ------------------------------------------------------------------
        # Step 2 — Warm-start the interaction matrix
        # ------------------------------------------------------------------
        logger.info("Step 3: Initialising interaction matrix (identity = no warm start)...")
        theta = theta_init.copy().astype(np.int32)
        np.fill_diagonal(theta, 1)  # every variable interacts with itself
        theta = enforce_transitivity(theta)

        n_init_interactions = int(np.sum(theta) - n)  # subtract diagonal
        logger.info(f"  Initial off-diagonal interactions: {n_init_interactions}")

        # ------------------------------------------------------------------
        # Step 3 — Incremental RRG loop
        # ------------------------------------------------------------------
        logger.info("Step 4: Starting incremental RRG loop...")
        stale_count = 0
        first_iter = True
        iteration = 0

        while True:
            iteration += 1

            # Generate a fresh random alternative solution for this RRG pass
            x2_bar = rng.uniform(lb, ub)

            fes_before_rrg = f.fes
            NonSepsTmp = rrg(x_hq, x2_bar, theta, f, n, lb, ub, config.ns, rng, logger, pool)
            fes_this_rrg = f.fes - fes_before_rrg

            # Count interactions before update
            interactions_before = int(np.sum(theta))

            # Update theta from newly discovered groups
            for group in NonSepsTmp:
                for i in group:
                    for j in group:
                        theta[i, j] = 1
                        theta[j, i] = 1

            # Enforce transitivity after each update
            theta = enforce_transitivity(theta)

            interactions_after = int(np.sum(theta))
            new_interactions = interactions_after - interactions_before
            found_new = new_interactions > 0

            logger.info(
                f"  Iter {iteration:3d} | FFEs this pass: {fes_this_rrg:5d} | "
                f"Total FFEs: {f.fes:7d} | New interactions: {new_interactions:4d} | "
                f"Stale count: {stale_count}"
            )

            if not found_new:
                stale_count += 1
                # Terminate immediately if first iteration found nothing
                if first_iter or stale_count >= config.sti:
                    logger.info(f"  Terminating: {'first iteration found nothing' if first_iter else f'stale for {config.sti} iterations'}")
                    break
            else:
                stale_count = 0

            first_iter = False

        # ------------------------------------------------------------------
        # Step 4 — Build output groups from final theta
        # ------------------------------------------------------------------
        logger.info("Step 5: Building output groups from final interaction matrix...")

        assigned: set = set()
        NonSeps: List[List[int]] = []
        separable_vars: List[int] = []

        for v in range(n):
            if v in assigned:
                continue
            group = [j for j in range(n) if theta[v, j] == 1 and j != v]
            if group:
                full_group = sorted(set([v] + group))
                NonSeps.append(full_group)
                assigned.update(full_group)
            else:
                separable_vars.append(v)

        # Chunk separable variables into groups of size config.s
        Seps: List[List[int]] = []
        while len(separable_vars) > 0:
            chunk = separable_vars[:config.s]
            Seps.append(chunk)
            separable_vars = separable_vars[config.s:]

        logger.info(
            f"  Decomposition complete: {len(NonSeps)} interacting groups, "
            f"{len(Seps)} separable groups."
        )
        return Seps, NonSeps


# ---------------------------------------------------------------------------
# INITIAL OPTIMIZER STUBS
# ---------------------------------------------------------------------------

def run_shade(
    f: IRRGFitnessWrapper,
    n: int,
    lb: np.ndarray,
    ub: np.ndarray,
    max_fes: int,
    rng: np.random.Generator,
    logger: logging.Logger,
    pool: Optional[ProcessPoolExecutor] = None
) -> np.ndarray:
    """
    Run SHADE (Success-History Based Parameter Adaptation for Differential
    Evolution) to find a globally promising starting solution x_hq.

    CURRENT IMPLEMENTATION:
        Batched random search fallback — evaluates max_fes random solutions in
        parallel batches and returns the best.  Replace this with a full SHADE
        implementation for production use.

    Parameters
    ----------
    f : IRRGFitnessWrapper
        Fitness oracle.
    n : int
        Problem dimension.
    lb, ub : np.ndarray
        Variable bounds.
    max_fes : int
        Maximum number of fitness evaluations.
    rng : np.random.Generator
        Random generator.
    logger : logging.Logger
    pool : ProcessPoolExecutor, optional
        If provided, evaluations are dispatched in parallel batches.

    Returns
    -------
    np.ndarray of shape (n,)
        Best solution found within the FFE budget.
    """
    logger.debug(f"  run_shade: batched random search, max_fes={max_fes}")

    best_x = rng.uniform(lb, ub)
    best_fit = f(best_x)
    fes_used = 1

    # Evaluate in batches for parallel speedup
    batch_size = max(1, (pool._max_workers * 2) if pool is not None else 1)

    while fes_used < max_fes:
        n_remaining = max_fes - fes_used
        batch_n = min(batch_size, n_remaining)
        candidates = [rng.uniform(lb, ub) for _ in range(batch_n)]

        if pool is not None:
            fits = f.batch_evaluate(candidates, pool)
        else:
            fits = [f(c) for c in candidates]
        fes_used += batch_n

        for cand, fit in zip(candidates, fits):
            if fit < best_fit:
                best_fit = fit
                best_x = cand.copy()

    logger.debug(f"  run_shade done. Best fitness: {best_fit:.4f}")
    return best_x


def run_mts_ls1(
    f: IRRGFitnessWrapper,
    x_start: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_fes: int,
    rng: np.random.Generator,
    logger: logging.Logger
) -> np.ndarray:
    """
    Run MTS-LS1 (Multiple Trajectory Search Local Search 1) to refine x_hq.

    WHAT MTS-LS1 DOES (for implementation):
        MTS-LS1 is a coordinate-wise local search. It maintains a per-variable
        step size SR[i] (search range). At each step:
          1. For each variable i in a random order:
             a. Try x[i] + SR[i]. If improvement, accept. Else:
             b. Try x[i] - SR[i]. If improvement, accept. Else:
             c. SR[i] *= 0.5 (shrink the step size for this variable).
          2. Repeat until max_fes is exhausted.

        Initial step size: SR[i] = 0.2 * (ub[i] - lb[i]) (20% of range).
        Minimum step size: SR[i] >= 1e-15 (stop shrinking at machine precision).

    ROLE IN IRRG:
        MTS-LS1 refines the SHADE output to land close to a local optimum.
        Interaction detection with rankings is most sensitive near optima because
        small perturbations to X1 variables create meaningful fitness differences
        that can be reversed by changing X2. Far from an optimum, the gradient
        dominates and the signal drowns out.

    CURRENT IMPLEMENTATION:
        Coordinate-wise greedy local search with step halving — a simplified but
        functionally similar version of MTS-LS1.

    Parameters
    ----------
    f : IRRGFitnessWrapper
        Fitness oracle.
    x_start : np.ndarray of shape (n,)
        Starting point (output of SHADE).
    lb, ub : np.ndarray
        Variable bounds.
    max_fes : int
        Maximum number of fitness evaluations.
    rng : np.random.Generator
        Random generator.
    logger : logging.Logger

    Returns
    -------
    np.ndarray of shape (n,)
        Refined solution.
    """
    logger.debug(f"  run_mts_ls1: coordinate search, max_fes={max_fes}")

    n = len(x_start)
    x = x_start.copy()
    best_fit = f(x)
    fes_used = 1

    # Initial step sizes: 20% of each variable's range
    SR = 0.2 * (ub - lb)

    while fes_used < max_fes:
        improved_any = False
        var_order = list(range(n))
        rng.shuffle(var_order)

        for i in var_order:
            if fes_used >= max_fes:
                break

            # Try positive step
            x_trial = x.copy()
            x_trial[i] = np.clip(x[i] + SR[i], lb[i], ub[i])
            fit = f(x_trial)
            fes_used += 1

            if fit < best_fit:
                x = x_trial
                best_fit = fit
                improved_any = True
                continue

            if fes_used >= max_fes:
                break

            # Try negative step
            x_trial = x.copy()
            x_trial[i] = np.clip(x[i] - SR[i], lb[i], ub[i])
            fit = f(x_trial)
            fes_used += 1

            if fit < best_fit:
                x = x_trial
                best_fit = fit
                improved_any = True
            else:
                # No improvement in either direction — shrink step
                SR[i] = max(SR[i] * 0.5, 1e-15)

        if not improved_any:
            # No variable improved this sweep — done
            break

    logger.debug(f"  run_mts_ls1 done. Best fitness: {best_fit:.4f}")
    return x


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("IRRG — interaction detection on real baseline traffic data")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load baseline data (same pattern as DG2_grouping.py)
    # ------------------------------------------------------------------
    with open(BASELINE_TRAFFIC_DATA, "r") as _fh:
        baseline_data = json.load(_fh)

    # Build the picklable fitness wrapper — derives TLS mapping, bounds,
    # and gene labels directly from the baseline data.
    traffic_wrapper, n, x_lower, x_upper, labels = build_traffic_fitness_wrapper(
        baseline_data    = baseline_data,
        fitness_function = _traffic_fitness,
    )

    # Identity theta_init: no prior interactions known (IRRG is independent)
    theta_init = np.eye(n, dtype=np.int32)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    config = IRRGConfig(
        ns=10,
        sti=15,
        s=100,
        shade_fes=5000,
        mts_fes=15000,
        seed=42,
        log_level="INFO"
    )

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("IRRG")

    rng = np.random.default_rng(config.seed)

    n_workers = NUM_PROCESSORS if NUM_PROCESSORS is not None else os.cpu_count()

    # ------------------------------------------------------------------
    # Run IRRG
    # ------------------------------------------------------------------
    f = IRRGFitnessWrapper(traffic_wrapper, x_lower, x_upper)

    logger.info(f"Starting IRRG | n={n} variables | ns={config.ns} | sti={config.sti} | workers={n_workers}")
    t_start = time.time()

    Seps, NonSeps = irrg(f, n, x_lower, x_upper, theta_init, config, rng, logger, n_workers)

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("IRRG DECOMPOSITION COMPLETE")
    print("=" * 60)
    print(f"Total FFEs consumed by decomposition : {f.fes}")
    print(f"Workers used                         : {n_workers}")
    print(f"Wall-clock time                      : {elapsed:.2f}s")
    print(f"Separable groups   ({len(Seps):3d} groups)  : ", end="")
    print(", ".join(f"[{len(g)} vars]" for g in Seps[:6]), ("..." if len(Seps) > 6 else ""))
    print(f"Interacting groups ({len(NonSeps):3d} groups)  : ", end="")
    print(", ".join(f"[{len(g)} vars]" for g in NonSeps[:6]), ("..." if len(NonSeps) > 6 else ""))

    total_vars_accounted = sum(len(g) for g in Seps) + sum(len(g) for g in NonSeps)
    print(f"Variables accounted for              : {total_vars_accounted} / {n}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Save groups to JSON for downstream optimizer
    # ------------------------------------------------------------------
    output_dir = Path("src/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "n": n,
        "total_fes": f.fes,
        "elapsed_seconds": round(elapsed, 3),
        "n_workers_used": n_workers,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "ns": config.ns,
            "sti": config.sti,
            "s": config.s,
            "shade_fes": config.shade_fes,
            "mts_fes": config.mts_fes,
            "seed": config.seed,
        },
        "gene_labels": labels,
        "separable_groups": Seps,
        "interacting_groups": NonSeps,
    }

    output_path = output_dir / "irrg_decomposition.json"
    with open(output_path, "w") as fp:
        json.dump(output, fp, indent=2)

    print(f"\nResults saved to: {output_path}")