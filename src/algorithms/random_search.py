import json, sys, time, os, copy
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    BASELINE_TRAFFIC_DATA, NUM_PROCESSORS, GAUSSIAN_NOISE,
    MAX_EVALS, GREEN_FLOOR, INSTANCES, SEED_BASE
)
from src.sumo_setup.fitness_evaluation import (
    fitness_function as _traffic_fitness,
    build_traffic_fitness_wrapper,
)

def _eval_worker(args):
    """Picklable worker for parallel execution."""
    wrapper, sol_idx, rep, solution = args
    t0 = time.time()
    fitness = float(wrapper(solution))
    elapsed = time.time() - t0
    return sol_idx, rep, fitness, elapsed


def init_population(strategy, n, num_genes, baseline_vec, noise_std, rng, ub):
    """Create initial population: 'random', 'baseline', or 'mixed'.

    ``ub`` is the per-gene upper bound (dynamic per-TLS green/red ceiling).
    """
    # Per-gene lower bound: yellow phases have ub=6 < GREEN_FLOOR, so cap the
    # lower at ub to keep uniform()/clip() valid (yellow collapses to 6).
    lo = np.minimum(GREEN_FLOOR, ub)
    if strategy == "random":
        return rng.uniform(lo, ub, (n, num_genes))

    elif strategy == "baseline":
        pop = np.tile(baseline_vec, (n, 1))
        pop += rng.normal(0, noise_std, pop.shape) * pop
        return np.clip(pop, lo, ub)

    elif strategy == "mixed":
        half = n // 2
        rand = rng.uniform(lo, ub, (half, num_genes))
        base = np.tile(baseline_vec, (n - half, 1))
        base += rng.normal(0, noise_std, base.shape) * base
        return np.vstack([rand, np.clip(base, lo, ub)])

    raise ValueError(f"Unknown strategy: {strategy}")


def build_gene_map(baseline_data):
    """Builds a gene map and baseline vector for population initialization."""
    tls_to_genes = {}
    idx = 0
    baseline = []

    for tls_id in sorted(baseline_data["tls_data"]):
        phases = sorted(baseline_data["tls_data"][tls_id])
        tls_to_genes[tls_id] = (idx, idx + len(phases))
        for pk in phases:
            baseline.append(float(baseline_data["tls_data"][tls_id][pk]["duration"]))
        idx += len(phases)

    return tls_to_genes, idx, np.array(baseline)


def run_single_search(strategy, baseline_data, wrapper, num_genes, baseline_vec, tls_to_genes, ub, out_dir, rng):
    """Run a single Random Search and write its result file."""
    print(f"\n{'='*60}")
    print(f"Random Search | Strategy: {strategy} | Solutions: {MAX_EVALS}")
    print(f"{'='*60}")

    solutions = init_population(strategy, MAX_EVALS, num_genes, baseline_vec, GAUSSIAN_NOISE, rng, ub)
    
    tasks = []
    for i, solution in enumerate(solutions):
        tasks.append((wrapper, i, 1, solution))

    n_workers = NUM_PROCESSORS or os.cpu_count() or 1
    t_start = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_eval_worker, task) for task in tasks]
        
        for idx, future in enumerate(as_completed(futures)):
            sol_idx, rep, fitness, elapsed = future.result()
            results.append({
                "solution": sol_idx,
                "repeat": rep,
                "fitness": fitness,
                "time_s": round(elapsed, 4),
            })
            
    total_time = time.time() - t_start
    results.sort(key=lambda x: (x["solution"], x["repeat"]))

    best_entry = min(results, key=lambda x: x["fitness"])
    best_sol_idx = best_entry["solution"]
    best_fitness = best_entry["fitness"]
    best_solution = solutions[best_sol_idx]

    print(f"Done in {total_time:.1f}s | Best: Solution {best_sol_idx + 1} | Fitness: {best_fitness:.2f}")

    # Reconstruct best configuration JSON
    best_json = copy.deepcopy(baseline_data)
    for tls_id in sorted(best_json["tls_data"]):
        if tls_id not in tls_to_genes:
            continue
        s, e = tls_to_genes[tls_id]
        raw = best_solution[s:e]
        keys = sorted(best_json["tls_data"][tls_id])
        num_phases = len(keys)

        total = sum(raw)
        if total <= 0:
            dur = [90 // num_phases] * num_phases
            dur[-1] += 90 - sum(dur)
        else:
            dur = [max(1, int(round(d * 90 / total))) for d in raw]
            diff = 90 - sum(dur)
            if diff:
                dur[int(np.argmax(dur))] += diff

        for i, pk in enumerate(keys):
            best_json["tls_data"][tls_id][pk]["duration"] = int(dur[i])

    best_json["composite_cost"] = best_fitness

    output = {
        "evaluations": results,
        "best_configuration": best_json,
        "best_fitness": best_fitness,
        "best_solution_index": int(best_sol_idx),
        "total_time_s": round(total_time, 2),
        "MAX_EVALS": MAX_EVALS,
        "total_evaluations": len(results),
        "algorithm": "random_search",
        "strategy": strategy,
        "seed": int(os.environ.get("TLS_RS_SEED", SEED_BASE)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = out_dir / os.environ.get("TLS_RS_OUTFILE", "random_search.json")
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    return best_fitness, total_time


def _resolve_out_dir():
    """Output dir for this run: ``<active instance out_dir>/random_search``.

    The active instance is whichever ``INSTANCES`` entry matches the baseline
    traffic data ``config`` resolved (honours ``TLS_BASELINE_DATA``); falls back
    to the default ``src/outputs`` when nothing matches.

    ``TLS_RS_OUTDIR`` overrides this entirely (used by the multi-repetition
    runner to collect all reps + summaries under one dedicated folder).
    """
    override = os.environ.get("TLS_RS_OUTDIR")
    if override:
        return Path(override)
    root = Path(__file__).resolve().parent.parent.parent
    active = Path(BASELINE_TRAFFIC_DATA).resolve()
    base = root / "src" / "outputs"
    for spec in INSTANCES.values():
        if Path(spec["baseline_data"]).resolve() == active:
            base = spec["out_dir"]
            break
    return base / "random_search"


def run_all_experiments():
    """Run a single random search (random init, no tree strategies).

    Writes ``random_search.json`` into the active instance's
    ``<out_dir>/random_search`` folder.

    Two environment variables let a multi-repetition runner drive this without
    touching the module:
      * ``TLS_RS_SEED`` — RNG seed for the initial population (default
        ``SEED_BASE``).  Distinct seeds across repetitions make the runs differ;
        reusing one seed would make them byte-identical and any average over
        them meaningless.
      * ``TLS_RS_OUTFILE`` — basename to write instead of ``random_search.json``
        (e.g. ``random_search_rep1.json``), so reps never clobber each other.
    """
    with open(BASELINE_TRAFFIC_DATA) as fh:
        baseline_data = json.load(fh)

    wrapper, num_genes, _, ub, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline_data,
        fitness_function=_traffic_fitness,
    )

    tls_to_genes, _, baseline_vec = build_gene_map(baseline_data)

    out_dir = _resolve_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional MAX_EVALS override (smoke tests). MAX_EVALS is read as a module
    # global by run_single_search, so reassign it here before the run.
    global MAX_EVALS
    if "TLS_RS_MAX_EVALS" in os.environ:
        MAX_EVALS = int(os.environ["TLS_RS_MAX_EVALS"])

    strategy = "random"
    seed = int(os.environ.get("TLS_RS_SEED", SEED_BASE))
    rng = np.random.default_rng(seed)

    try:
        best_cost, elapsed = run_single_search(
            strategy, baseline_data, wrapper, num_genes,
            baseline_vec, tls_to_genes, ub, out_dir, rng
        )
        info = {"best": best_cost, "time_s": elapsed}
    except Exception as e:
        print(f"ERROR [{strategy}]: {e}")
        import traceback; traceback.print_exc()
        info = {"error": str(e)}

    # Print results
    print(f"\n{'Strategy':<10} {'Best':>14} {'Time':>9}")
    print("─" * 35)
    if "error" in info:
        print(f"{strategy:<10} {'ERROR':>14}")
    else:
        print(f"{strategy:<10} {info['best']:>14.2f} {info['time_s']:>8.1f}s")


if __name__ == "__main__":
    run_all_experiments()
