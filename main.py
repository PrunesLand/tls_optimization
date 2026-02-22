from config import *
from genetic_algorithm.initialization import generate_population


def genetic_algorithm():
    population = generate_population(POPULATION)
    print(population)
if __name__ == "__main__":
    genetic_algorithm()