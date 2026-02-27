from config import *
from src.genetic_algorithm.fitness_evaluation import evaluate_individual
from src.genetic_algorithm.initialization import generate_individual

def genetic_algorithm():
    individual = generate_individual(BASELINE_TRAFFIC_DATA, OUTPUT_JSON_PATH)
    evaluate_individual(individual)

if __name__ == "__main__":
    genetic_algorithm()