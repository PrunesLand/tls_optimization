import jax
import jax.numpy as jnp
from evox.core.problem import Problem
from evox import algorithms, workflows, monitors
import numpy as np
import copy
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import configurations and shared functions
from config import (
    SUMO_ARGS,
    EVOX_POP_SIZE,
    EVOX_GENERATIONS,
    EVOX_MUTATION_RATE,
    EVOX_LB,
    EVOX_UB,
    EVOX_NUM_WORKERS,   # Add this to config.py — see note below
)
from src.sumo_setup.extraction import extract_traffic_light_data
from src.genetic_algorithm.fitness_evaluation import fitness_function


# ---------------------------------------------------------------------------
# Top-level worker function (MUST be at module level for multiprocessing pickle)
# ---------------------------------------------------------------------------
def _evaluate_individual(args):
    """
    Evaluates a single individual in its own process.
    Each call gets an isolated Python interpreter with its own libsumo instance,
    which is the only safe way to parallelise libsumo simulations.

    Args:
        args: tuple of (index, individual_tls_data)
              index is returned so results can be reassembled in order.
    Returns:
        (index, fitness_value)
    """
    idx, individual_data = args
    # Re-import here is intentional: each worker process needs its own
    # sys.path setup since it starts fresh.
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.genetic_algorithm.fitness_evaluation import fitness_function

    wrapped_data = {"tls_data": individual_data}
    fit = fitness_function(wrapped_data)
    return idx, float(fit)


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------
def check_gpu():
    """Checks for GPU availability and reports it."""
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.device_kind.lower() == 'gpu']
    if gpu_devices:
        print(f"Compatible GPU(s) found: {gpu_devices}. Utilizing GPU for EvoX operations.")
    else:
        print("No compatible GPU found. Utilizing CPU for EvoX operations.")
    return devices[0]


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
class TrafficLightProblem(Problem):
    """
    Traffic Light Optimization Problem for EvoX v0.9.0 (JAX).
    Evaluates the entire population in parallel using ProcessPoolExecutor,
    with one independent libsumo instance per worker process.
    """
    def __init__(self, base_tls_data, num_workers=None):
        super().__init__()
        self.base_tls_data = base_tls_data
        self.tls_ids = sorted(base_tls_data.keys())
        self.num_workers = num_workers or os.cpu_count()

        # Identify modifiable phases (those with 'g' in state string)
        self.modifiable_phases = []  # list of (tls_id, phase_key)
        for tid in self.tls_ids:
            phases = base_tls_data[tid]
            sorted_keys = sorted(phases.keys(), key=lambda x: int(x.split('_')[1]))
            for pkey in sorted_keys:
                if 'g' in phases[pkey]['state'].lower():
                    self.modifiable_phases.append((tid, pkey))

        self.total_dims = len(self.modifiable_phases)
        if self.total_dims == 0:
            raise ValueError("No green phases found to optimize.")

        print(f"TrafficLightProblem initialized: {self.total_dims} green phases, "
              f"{self.num_workers} parallel workers.")

    def evaluate(self, state, X):
        """
        EvoX v0.9.0 Problem API: accepts (state, X), returns (fitness, state).
        Uses jax.pure_callback to safely escape JAX tracing for the
        parallel SUMO evaluation.
        """
        fitness = jax.pure_callback(
            self._parallel_evaluate,
            jax.ShapeDtypeStruct((X.shape[0],), jnp.float32),
            X
        )
        return fitness, state

    def _parallel_evaluate(self, X_jax):
        """
        Builds per-individual TLS data configs and dispatches them to a
        process pool. Each worker runs a fully independent libsumo instance,
        which is required because libsumo is NOT thread-safe.
        """
        X_np = np.array(X_jax)
        pop_size = X_np.shape[0]
        fitnesses = np.zeros(pop_size, dtype=np.float32)

        # Build args list: one (index, tls_data_dict) tuple per individual
        args_list = []
        for i in range(pop_size):
            individual_data = copy.deepcopy(self.base_tls_data)
            for j, (tid, pkey) in enumerate(self.modifiable_phases):
                duration = int(max(1, round(float(X_np[i, j]))))
                individual_data[tid][pkey]['duration'] = duration
            args_list.append((i, individual_data))

        # Dispatch to process pool
        # max_workers is capped at pop_size — no point spinning up more
        # workers than there are individuals to evaluate.
        max_workers = min(self.num_workers, pop_size)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_evaluate_individual, args): args[0]
                       for args in args_list}
            for future in as_completed(futures):
                idx, fit = future.result()
                fitnesses[idx] = fit

        return fitnesses.astype(np.float32)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_evox_ga(pop_size=None, generations=None, num_workers=None):
    """
    Configures and runs the EvoX DE optimizer (v0.9.0 JAX API)
    with parallelised SUMO fitness evaluation.
    """
    final_pop_size = pop_size or EVOX_POP_SIZE
    final_generations = generations or EVOX_GENERATIONS
    final_workers = num_workers or EVOX_NUM_WORKERS

    # 1. GPU check
    check_gpu()

    # 2. Extract traffic light data from SUMO
    print("Extracting traffic light data...")
    initial_data_dict = extract_traffic_light_data()
    if not initial_data_dict or "tls_data" not in initial_data_dict:
        print("Error: Could not extract TLS data.")
        return None

    # 3. Instantiate the problem (with parallel workers)
    problem = TrafficLightProblem(initial_data_dict["tls_data"], num_workers=final_workers)

    # 4. Configure DE algorithm
    lb = jnp.full((problem.total_dims,), float(EVOX_LB))
    ub = jnp.full((problem.total_dims,), float(EVOX_UB))

    algorithm = algorithms.DE(
        lb=lb,
        ub=ub,
        pop_size=final_pop_size,
        batch_size=final_pop_size,
        differential_weight=float(EVOX_MUTATION_RATE),
        cross_probability=0.9,
    )

    # 5. Monitor and Workflow
    monitor = monitors.EvalMonitor()
    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitors=[monitor],
    )

    # 6. Init
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    # 7. Optimization loop
    print(f"Starting EvoX DE: {final_pop_size} population, "
          f"{final_generations} generations, {final_workers} parallel workers.")
    for gen in range(final_generations):
        state = workflow.step(state)
        best_fitness, state = monitor.get_best_fitness(state)
        print(f"Gen {gen + 1}/{final_generations} | Best Fitness: {float(best_fitness):.4f}")

    print("Optimization finished.")
    return float(best_fitness)


if __name__ == "__main__":
    run_evox_ga(pop_size=10, generations=5)