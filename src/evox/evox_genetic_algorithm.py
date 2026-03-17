import jax
import jax.numpy as jnp
from evox.core.problem import Problem
from evox import algorithms, workflows, monitors
import numpy as np
import copy
import os
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import configurations and shared functions
from config import (
    SUMO_ARGS,
    EVOX_POP_SIZE,
    EVOX_GENERATIONS,
    EVOX_MUTATION_RATE,
    EVOX_LB,
    EVOX_UB
)
from src.sumo_setup.extraction import extract_traffic_light_data
from src.genetic_algorithm.fitness_evaluation import fitness_function


def check_gpu():
    """Checks for GPU availability and reports it."""
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.device_kind.lower() == 'gpu']
    if gpu_devices:
        print(f"Compatible GPU(s) found: {gpu_devices}. Utilizing GPU for EvoX operations.")
    else:
        print("No compatible GPU found. Utilizing CPU for EvoX operations.")
    return devices[0]


class TrafficLightProblem(Problem):
    """
    Traffic Light Optimization Problem for EvoX v0.9.0 (JAX).
    Optimizes durations of phases containing 'green' signals.
    """
    def __init__(self, base_tls_data):
        super().__init__()
        self.base_tls_data = base_tls_data
        self.tls_ids = sorted(base_tls_data.keys())

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

        print(f"TrafficLightProblem initialized with {self.total_dims} modifiable green phases.")

    def evaluate(self, state, X):
        """
        EvoX v0.9.0 Problem API: must accept (state, X) and return (fitness, state).
        X is a JAX array of shape (pop_size, total_dims).

        Because SUMO is an external subprocess (not a JAX op), we use
        jax.pure_callback to safely call Python/NumPy code from within
        the JAX-traced workflow.
        """
        fitness = jax.pure_callback(
            self._external_evaluate,
            jax.ShapeDtypeStruct((X.shape[0],), jnp.float32),
            X
        )
        return fitness, state

    def _external_evaluate(self, X_jax):
        """
        Pure Python evaluation — runs SUMO for each individual.
        Called via jax.pure_callback so JAX can trace around it.
        """
        X_np = np.array(X_jax)
        pop_size = X_np.shape[0]
        fitnesses = np.zeros(pop_size, dtype=np.float32)

        for i in range(pop_size):
            individual_data = copy.deepcopy(self.base_tls_data)
            for j, (tid, pkey) in enumerate(self.modifiable_phases):
                duration = int(max(1, round(float(X_np[i, j]))))
                individual_data[tid][pkey]['duration'] = duration

            wrapped_data = {"tls_data": individual_data}
            fit = fitness_function(wrapped_data)
            fitnesses[i] = float(fit)

        return fitnesses.astype(np.float32)


def run_evox_ga(pop_size=None, generations=None):
    """
    Configures and runs the EvoX DE optimizer (v0.9.0 JAX API).

    Note: EvoX v0.9.0 has no 'GA' or 'GeneticAlgorithm' class.
    'DE' (Differential Evolution) is the correct single-objective,
    real-valued optimizer — it uses mutation + crossover + selection,
    the same core operations as a standard GA.
    """
    final_pop_size = pop_size or EVOX_POP_SIZE
    final_generations = generations or EVOX_GENERATIONS

    # 1. GPU check
    check_gpu()

    # 2. Extract traffic light data from SUMO
    print("Extracting traffic light data...")
    initial_data_dict = extract_traffic_light_data()
    if not initial_data_dict or "tls_data" not in initial_data_dict:
        print("Error: Could not extract TLS data.")
        return None

    # 3. Instantiate the problem
    problem = TrafficLightProblem(initial_data_dict["tls_data"])

    # 4. Configure DE algorithm
    # algorithms.DE is confirmed present in evox 0.9.0.
    # differential_weight (F) controls mutation scale — mapped from EVOX_MUTATION_RATE.
    lb = jnp.full((problem.total_dims,), float(EVOX_LB))
    ub = jnp.full((problem.total_dims,), float(EVOX_UB))

    algorithm = algorithms.DE(
        lb=lb,
        ub=ub,
        pop_size=final_pop_size,
        batch_size=final_pop_size,  # must be <= pop_size; defaults to 100 which breaks small pop sizes
        differential_weight=float(EVOX_MUTATION_RATE),
        cross_probability=0.9,
    )

    # 5. Monitor and Workflow
    # Both confirmed present in evox 0.9.0 from your dir() output.
    monitor = monitors.EvalMonitor()
    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitors=[monitor],
    )

    # 6. Init — v0.9.0 uses a JAX PRNGKey, not init_step()
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    # 7. Optimization loop
    print(f"Starting EvoX DE: {final_pop_size} population, {final_generations} generations.")
    for gen in range(final_generations):
        state = workflow.step(state)

        # EvalMonitor in v0.9.0: fitness history is stored per-step
        best_fitness = monitor.get_best_fitness()
        print(f"Gen {gen + 1}/{final_generations} | Best Fitness: {float(best_fitness):.4f}")

    print("Optimization finished.")
    return float(monitor.get_best_fitness())


if __name__ == "__main__":
    run_evox_ga(pop_size=10, generations=5)