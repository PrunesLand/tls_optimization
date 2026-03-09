import jax
import jax.numpy as jnp
from evox import algorithms, problems, workflows, monitors

# --- NEW METHODOLOGY: Custom Uniform Mutation ---
def custom_uniform_mutation(key, individual, lb, ub, mutation_rate=0.1):
    """
    Methodology: Replaces random genes with new values from the search space.
    """
    key_mask, key_values = jax.random.split(key)
    
    # Create a mask: True for genes that will be mutated
    mask = jax.random.uniform(key_mask, individual.shape) < mutation_rate
    
    # Generate new random values within the bounds [lb, ub]
    new_values = jax.random.uniform(key_values, individual.shape, minval=lb, maxval=ub)
    
    # Apply the methodology: if mask is true, take new_value, else keep individual
    return jnp.where(mask, new_values, individual)

class MyCustomGA(algorithms.GeneticAlgorithm):
    def __init__(self, lb, ub, pop_size, mutation_rate=0.05):
        super().__init__(lb=lb, ub=ub, pop_size=pop_size)
        self.mutation_rate = mutation_rate

    def ask(self, state):
        # 1. Standard Crossover (handled by parent class or custom logic)
        state, offspring = super().ask(state)
        
        # 2. Injecting our NEW Mutation Methodology
        key, mutation_key = jax.random.split(state.key)
        
        # We 'vmap' our custom mutation to apply it to the whole population at once
        vmapped_mutation = jax.vmap(custom_uniform_mutation, in_axes=(0, 0, None, None, None))
        
        # Split keys for each individual in the population
        pop_keys = jax.random.split(mutation_key, offspring.shape[0])
        mutated_offspring = vmapped_mutation(pop_keys, offspring, self.lb, self.ub, self.mutation_rate)
        
        return state.update(key=key), mutated_offspring

# --- RUNNING THE SCRIPT ---
problem = problems.numerical.Sphere()
monitor = monitors.HPOFitnessMonitor()

# Initialize our custom methodology GA
my_ga = MyCustomGA(lb=jnp.full(5, -5.0), ub=jnp.full(5, 5.0), pop_size=100)

workflow = workflows.StdWorkflow(algorithm=my_ga, problem=problem, monitors=[monitor])
state = workflow.init(jax.random.PRNGKey(42))

for i in range(50):
    state = workflow.step(state)
    if i % 10 == 0:
        print(f"Gen {i} Best Fitness: {monitor.get_best_fitness():.4f}")