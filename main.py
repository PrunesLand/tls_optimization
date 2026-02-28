from config import *
from src.genetic_algorithm.fitness_evaluation import evaluate_population
from src.genetic_algorithm.initialization import generate_population

def genetic_algorithm():
    population = []
    population = generate_population(BASELINE_TRAFFIC_DATA, OUTPUT_JSON_PATH)
    evaluate_population(population)
    for generation in range(GENERATIONS):
        print(f"Generation {generation + 1}/{GENERATIONS}")
        
        evaluate_population(population)


if __name__ == "__main__":
    genetic_algorithm()