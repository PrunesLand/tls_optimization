import libsumo as traci

from config import SUMO_ARGS


def fitness_function(individual):

    return

def evaluate_population(population):

    for individual in population:
        fitness = fitness_function(individual)
        individual['fitness'] = fitness
    return