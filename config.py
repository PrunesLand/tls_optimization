import os
import multiprocessing

# configuration parameters for the genetic algorithm
POPULATION = 300
GENERATIONS = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7

# pygad configuration parameters
PYGAD_POPULATION_SIZE = 3
PYGAD_NUM_GENERATIONS = 2
PYGAD_MUTATION_PERCENT_GENES = 10
PYGAD_NUM_PARENTS_MATING = 2
PYGAD_KEEP_PARENTS = 2

# Parallel processing configuration
# Set to an integer to use a specific number of CPUs, or something else (like None) to use all.
NUM_PROCESSORS = None

#configuration file for sumo
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "src/sumo_setup/osm.sumocfg")
SUMO_ARGS = [
    "sumo", 
    "-c", CONFIG_FILE, 
    "--no-step-log", "true", 
    "--no-warnings", "true",
    "--time-to-teleport", "-1",  
]
BASELINE_TRAFFIC_DATA = os.path.join(os.path.dirname(__file__), "src/outputs/baseline_traffic_data.json")
OUTPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), "src/outputs/check_traffic_cycles.json")