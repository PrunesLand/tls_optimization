import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import BASELINE_TRAFFIC_DATA
from src.genetic_algorithm.fitness_evaluation import fitness_function as _traffic_fitness
from src.decomposition.DG2_grouping import build_traffic_fitness_wrapper

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

def evaluate_baseline():
    print(f"Loading baseline data from {BASELINE_TRAFFIC_DATA}")
    with open(BASELINE_TRAFFIC_DATA) as fh:
        baseline_data = json.load(fh)

    print("Building fitness wrapper...")
    wrapper, num_genes, _, _, _ = build_traffic_fitness_wrapper(
        baseline_data=baseline_data,
        fitness_function=_traffic_fitness,
    )
    
    tls_to_genes, _, baseline_vec = build_gene_map(baseline_data)

    print("Evaluating baseline configuration...")
    t0 = time.time()
    fitness = float(wrapper(baseline_vec))
    elapsed = time.time() - t0

    print(f"Baseline fitness: {fitness}")
    print(f"Evaluation took {elapsed:.2f} seconds")

    output = {
        "algorithm": "baseline",
        "best_fitness": fitness,
        "total_time_s": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "best_configuration": baseline_data
    }

    # We update composite_cost in the config data just for completeness
    output["best_configuration"]["composite_cost"] = fitness

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_results.json"

    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    evaluate_baseline()
