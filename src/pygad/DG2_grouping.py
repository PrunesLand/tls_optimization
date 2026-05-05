"""
DG2 — Differential Grouping 2 for variable interaction detection.
Based on Omidvar et al. (2017), IEEE Trans. Evolutionary Computation.

Detects which traffic light phase durations interact with each other
using parallel SUMO evaluations, then groups them for co-optimization.
"""

import json, math, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Machine epsilon constants
MACHINE_EPSILON = np.finfo(float).eps
GAMMA2 = 2 * MACHINE_EPSILON / (1 - 2 * MACHINE_EPSILON)

CYCLE_LENGTH = 90

PHASE_BOUNDS = {
    "green":  (24, 85),
    "yellow": ( 3,  6),
    "red":    ( 5, 85),
}


def _phase_type(state: str) -> str:
    """Infer phase type (green/yellow/red) from SUMO state string."""
    s = state.lower()
    counts = {"green": s.count("g"), "yellow": s.count("y"), "red": s.count("r")}
    for ptype in ("green", "yellow", "red"):
        if counts[ptype] == max(counts.values()):
            return ptype
    return "red"


def _eval_probe(args):
    """Worker function for parallel evaluation. Must be module-level for pickling."""
    f, x, tag = args
    return tag, float(f(x))


# ---------------------------------------------------------------------------
# Step 1 — Interaction Structure Matrix (parallel)
# ---------------------------------------------------------------------------

def _compute_ISM(f, n, x_lower, x_upper, n_workers):
    """Build the raw ISM (Lambda) by evaluating all probe points in parallel."""
    m = 0.5 * (x_lower + x_upper)
    tasks = [(f, m.copy(), ("base",))]

    for i in range(n):
        x = m.copy(); x[i] = x_upper[i]
        tasks.append((f, x, ("single", i)))

    for i in range(n - 1):
        for j in range(i + 1, n):
            x = m.copy(); x[i] = x_upper[i]; x[j] = x_upper[j]
            tasks.append((f, x, ("double", i, j)))

    evals = len(tasks)
    f_base = np.nan
    f_hat = np.full(n, np.nan)
    F = np.full((n, n), np.nan)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_eval_probe, task): task[2] for task in tasks}
        for future in as_completed(futures):
            tag, value = future.result()
            if tag[0] == "base":       f_base = value
            elif tag[0] == "single":   f_hat[tag[1]] = value
            else:                      F[tag[1], tag[2]] = value

    Lambda = np.zeros((n, n))
    for i in range(n - 1):
        delta1_i = f_hat[i] - f_base
        for j in range(i + 1, n):
            lam = abs(delta1_i - (F[i, j] - f_hat[j]))
            Lambda[i, j] = Lambda[j, i] = lam

    return Lambda, F, f_hat, f_base, evals


# ---------------------------------------------------------------------------
# Step 2 — Design Structure Matrix
# ---------------------------------------------------------------------------

def _compute_DSM(Lambda, F, f_hat, f_base, n):
    """Convert raw ISM into binary interaction matrix (Theta)."""
    gamma_sqrt_n = MACHINE_EPSILON * math.sqrt(n) / (1 - math.sqrt(n) * MACHINE_EPSILON)
    Theta = np.full((n, n), np.nan)
    eta0, eta1 = 0, 0

    # First pass: classify clear-cut pairs
    for i in range(n - 1):
        for j in range(i + 1, n):
            fij = float(F[i, j]) if not np.isnan(F[i, j]) else 0.0
            fi  = float(f_hat[i]) if not np.isnan(f_hat[i]) else 0.0
            fj  = float(f_hat[j]) if not np.isnan(f_hat[j]) else 0.0
            fb  = float(f_base)

            einf = GAMMA2 * max(abs(fb) + abs(fij), abs(fi) + abs(fj))
            esup = gamma_sqrt_n * max(abs(fb), abs(fij), abs(fi), abs(fj))

            if Lambda[i, j] < einf:
                Theta[i, j] = Theta[j, i] = 0; eta0 += 1
            elif Lambda[i, j] > esup:
                Theta[i, j] = Theta[j, i] = 1; eta1 += 1

    # Second pass: resolve ambiguous pairs
    total_reliable = eta0 + eta1
    for i in range(n - 1):
        for j in range(i + 1, n):
            if not np.isnan(Theta[i, j]):
                continue
            fij = float(F[i, j]) if not np.isnan(F[i, j]) else 0.0
            fi  = float(f_hat[i]) if not np.isnan(f_hat[i]) else 0.0
            fj  = float(f_hat[j]) if not np.isnan(f_hat[j]) else 0.0
            fb  = float(f_base)

            einf = GAMMA2 * max(abs(fb) + abs(fij), abs(fi) + abs(fj))
            esup = gamma_sqrt_n * max(abs(fb), abs(fij), abs(fi), abs(fj))

            if total_reliable == 0:
                epsilon = 0.5 * (einf + esup)
            else:
                epsilon = (eta0 / total_reliable) * einf + (eta1 / total_reliable) * esup
            Theta[i, j] = Theta[j, i] = 1 if Lambda[i, j] > epsilon else 0

    np.fill_diagonal(Theta, 0)
    return np.nan_to_num(Theta, nan=0.0).astype(int)


# ---------------------------------------------------------------------------
# Step 3 — Connected components (BFS)
# ---------------------------------------------------------------------------

def _connected_components(Theta, n):
    """Partition variables into interacting groups and separable singletons."""
    visited = [False] * n
    groups, separable = [], []

    for start in range(n):
        if visited[start]:
            continue
        component, queue = [], [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(node)
            for nb in range(n):
                if not visited[nb] and Theta[node, nb] == 1:
                    visited[nb] = True
                    queue.append(nb)

        if len(component) == 1:
            separable.append(component[0])
        else:
            groups.append(sorted(component))

    return groups, separable


# ---------------------------------------------------------------------------
# Public API — run_dg2
# ---------------------------------------------------------------------------

def run_dg2(f, n, x_lower, x_upper, gene_labels=None,
            output_path="dg2_results.json", n_workers=None, verbose=True):
    """Run DG2 interaction detection. Returns results dict and saves to JSON."""
    if gene_labels is None:
        gene_labels = [f"gene_{i}" for i in range(n)]
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    x_lower = np.asarray(x_lower, dtype=float)
    x_upper = np.asarray(x_upper, dtype=float)

    if verbose:
        print(f"[DG2] Genes: {n} | Workers: {n_workers} | Probes: {n*(n+1)//2+1}")

    t_start = time.perf_counter()
    Lambda, F, f_hat, f_base, evals = _compute_ISM(f, n, x_lower, x_upper, n_workers)
    Theta = _compute_DSM(Lambda, F, f_hat, f_base, n)
    groups, separable = _connected_components(Theta, n)
    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"[DG2] Groups: {len(groups)} | Separable: {len(separable)} | "
              f"Evals: {evals} | Time: {elapsed:.2f}s")

    results = {
        "groups": [{"group_id": i, "size": len(g), "gene_indices": g,
                     "gene_labels": [gene_labels[k] for k in g]}
                   for i, g in enumerate(groups)],
        "separable_genes": [{"gene_index": i, "gene_label": gene_labels[i]}
                            for i in separable],
        "interaction_matrix": Theta.tolist(),
        "lambda_matrix": Lambda.tolist(),
        "gene_labels": gene_labels,
        "n_genes": n, "n_groups": len(groups), "n_separable": len(separable),
        "function_evaluations": evals, "n_workers_used": n_workers,
        "computation_time_seconds": round(elapsed, 6),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=4)
    if verbose:
        print(f"[DG2] Saved -> {output_path}")

    return results


# ---------------------------------------------------------------------------
# Traffic-light fitness wrapper (picklable for multiprocessing)
# ---------------------------------------------------------------------------

class TrafficFitnessWrapper:
    """Converts a gene vector into per-TLS durations and calls the SUMO fitness function."""

    def __init__(self, fitness_function, tls_mapping):
        self.fitness_function = fitness_function
        self.tls_mapping = tls_mapping

    def __call__(self, vector: np.ndarray) -> float:
        tls_durations = {}

        for tls in self.tls_mapping:
            tls_id = tls["tls_id"]
            phase_types = tls["phase_types"]
            raw = list(vector[tls["start_idx"]: tls["end_idx"]])

            # Clamp each gene to its per-type bounds
            durations = []
            for raw_val, ptype in zip(raw, phase_types):
                lo, hi = PHASE_BOUNDS[ptype]
                durations.append(int(round(max(lo, min(hi, raw_val)))))

            # Adjust for 90s cycle length (only green/red absorb remainder)
            remainder = CYCLE_LENGTH - sum(durations)
            if remainder != 0:
                adjustable = [i for i, pt in enumerate(phase_types) if pt in ("green", "red")]
                if adjustable:
                    target_idx = min(adjustable, key=lambda i: durations[i])
                    durations[target_idx] += remainder

                    lo, _ = PHASE_BOUNDS[phase_types[target_idx]]
                    if durations[target_idx] < lo:
                        durations[target_idx] = lo
                        fallback = max(adjustable, key=lambda i: durations[i])
                        durations[fallback] += CYCLE_LENGTH - sum(durations)

            tls_durations[tls_id] = durations

        return self.fitness_function(tls_durations)


def build_traffic_fitness_wrapper(baseline_data, fitness_function):
    """Build a picklable fitness wrapper from baseline data. Returns (wrapper, n, lb, ub, labels)."""
    tls_mapping = []
    gene_idx = 0
    x_lower_list, x_upper_list, labels = [], [], []

    for tls_id in sorted(baseline_data["tls_data"].keys()):
        phase_keys = sorted(baseline_data["tls_data"][tls_id].keys())
        phase_types = []

        for pk in phase_keys:
            state = baseline_data["tls_data"][tls_id][pk].get("state", "")
            ptype = _phase_type(state)
            phase_types.append(ptype)

            lo, hi = PHASE_BOUNDS[ptype]
            x_lower_list.append(float(lo))
            x_upper_list.append(float(hi))
            labels.append(f"{tls_id}_{pk}")

        tls_mapping.append({
            "tls_id": tls_id, "num_phases": len(phase_keys),
            "phase_types": phase_types,
            "start_idx": gene_idx, "end_idx": gene_idx + len(phase_keys),
        })
        gene_idx += len(phase_keys)

    f = TrafficFitnessWrapper(fitness_function, tls_mapping)
    return f, gene_idx, np.array(x_lower_list), np.array(x_upper_list), labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import BASELINE_TRAFFIC_DATA
    from src.genetic_algorithm.fitness_evaluation import fitness_function as _traffic_fitness

    with open(BASELINE_TRAFFIC_DATA) as fh:
        baseline_data = json.load(fh)

    f, n, x_lower, x_upper, labels = build_traffic_fitness_wrapper(
        baseline_data=baseline_data, fitness_function=_traffic_fitness,
    )

    results = run_dg2(
        f=f, n=n, x_lower=x_lower, x_upper=x_upper,
        gene_labels=labels, output_path="src/outputs/dg2_results.json",
    )