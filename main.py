from config import *
from genetic_algorithm.fitness_evaluation import evaluate_population
from src.genetic_algorithm.initialization import generate_population

def genetic_algorithm():
    population = generate_population(BASELINE_TRAFFIC_DATA, OUTPUT_JSON_PATH)
    evaluate_population(population)


if __name__ == "__main__":
    genetic_algorithm()