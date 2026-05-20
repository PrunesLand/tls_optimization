import os

MAX_EVALS = 1000

# pygad configuration parameters
PYGAD_POPULATION_SIZE = 100
PYGAD_NUM_GENERATIONS = MAX_EVALS // PYGAD_POPULATION_SIZE
PYGAD_MUTATION_PERCENT_GENES = 5
PYGAD_NUM_PARENTS_MATING = 20
PYGAD_KEEP_PARENTS = 2

# Parallel processing configuration
# Set to an integer to use a specific number of CPUs, or something else (like None) to use all.
NUM_PROCESSORS = None

#configuration file for sumo
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "src/sumo_setup/osm.sumocfg")
SUMO_ARGS = [
    "sumo", 
    "-c", CONFIG_FILE, 
    "--no-step-log",          
    "--no-warnings",          
    "--time-to-teleport", "-1", 
    "--duration-log.disable",
    "--duration-log.statistics", "false",   
]
BASELINE_TRAFFIC_DATA = os.path.join(os.path.dirname(__file__), "src/outputs/baseline_traffic_data.json")
OUTPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), "src/outputs/check_traffic_cycles.json")

# Custom Optimizer configuration
POPULATION_SIZE = 100
NUM_GENERATIONS = MAX_EVALS // POPULATION_SIZE
GAUSSIAN_NOISE = 0.10  # 10% Gaussian perturbation for baseline-perturbed init
NOVEL_MUTATION = False  # Enable or disable pair-cluster mutation

# Clustering thresholds
CLUSTER_THRESHOLD_FASTEST = 300
CLUSTER_THRESHOLD_SHORTEST = 4000
CLUSTER_THRESHOLD_EUCLIDIAN = 3500

# Custom Optimizer  Bounds and Mutation
GENE_LOW = 24.0
GENE_HIGH = 82.0
MUTATION_RATE = 0.15

CYCLE_LENGTH = 90

PHASE_BOUNDS = {
    "green":  (24, 82),
    "yellow": ( 3,  6),
    "red":    ( 5, 63),
}