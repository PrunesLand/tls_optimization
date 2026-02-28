import os

# configuration parameters for the genetic algorithm
POPULATION = 300
GENERATIONS = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7

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