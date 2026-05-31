import os

MAX_EVALS = 5000

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

# Base random seed (used by distance-matrix plotting / reproducible runs).
SEED_BASE = 42

# Clustering thresholds
CLUSTER_THRESHOLD_FASTEST = 300
CLUSTER_THRESHOLD_SHORTEST = 4000
CLUSTER_THRESHOLD_EUCLIDIAN = 3500

# Custom Optimizer  Bounds and Mutation
# Per-type phase-duration floors (minimums within the fixed cycle). Green/red
# ceilings are no longer static: they come from the dynamic per-TLS
# phase_upper_bounds in fitness_evaluation.py.
GREEN_FLOOR  = 24.0   # green phase minimum
RED_FLOOR    = 5.0    # red phase minimum
# Yellow phases are frozen to a fixed band (both bounds still used to clamp).
YELLOW_FLOOR = 3.0
YELLOW_CEIL  = 6.0
MUTATION_RATE = 0.15

CYCLE_LENGTH = 90

# Only TLSs whose phase count is in this set are optimised. Other TLSs are
# filtered out at baseline-JSON generation time (see src/sumo_setup/generation.py)
# and keep whatever durations the SUMO network file holds during simulation.
OPTIMIZE_PHASE_COUNTS = {3, 4}