from config import *
from src.genetic_algorithm.initialization import generate_population

def genetic_algorithm():
    population = generate_population(INPUT_JSON_PATH, OUTPUT_JSON_PATH)
    print(population)
if __name__ == "__main__":
    genetic_algorithm()