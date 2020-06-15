import matplotlib.pyplot as plt
import fiona
import init
import mutation
import numpy as np
import pandas as pd
import pickle
import random
import recombination
import solution

from deap.benchmarks.tools import hypervolume
from deap import base, creator, tools
from plot.plot_on_GSHHS import plot_on_gshhs
from rtree import index
from shapely.geometry import shape
from shapely.prepared import prep


class Vessel:
    def __init__(self, name, speeds, fuel_rates):
        self.name = name
        self.speeds = speeds
        self.fuel_rates = fuel_rates


max_distance = 200  # nautical miles
width_ratio = 0.5
# swaps = ['insert', 'move', 'delete', 'speed']
swaps = ['insert', 'move', 'delete']

# Vessel characteristics
vessel_name = 'Fairmaster'
table = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name=vessel_name)
speeds = [round(speed, 1) for speed in table['Speed']]
fuel_rates = {speed: round(table['Fuel'][i], 1) for i, speed in enumerate(speeds)}
vessel = Vessel(vessel_name, speeds, fuel_rates)

# Route characteristics and navigation area
start, end = (-5.352121, 48.021295), (53.131866, 13.521350)  # (longitude, latitude)
polygons = [shape(shoreline['geometry']) for shoreline in
            iter(fiona.open('C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp'))]
prepared_polygons = [prep(polygon) for polygon in polygons]

# Populate R-tree index with bounds of polygons
rtree_idx = index.Index()
for pos, polygon in enumerate(polygons):
    rtree_idx.insert(pos, polygon.bounds)

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("edge_feasible", solution.edge_feasible, rtree_idx, polygons, max_distance)
toolbox.register("insert", mutation.insert_waypoint, toolbox)
toolbox.register("delete", mutation.delete_random_waypoint, toolbox)
toolbox.register("speed", mutation.change_speed, vessel)
toolbox.register("move", mutation.move_waypoint, toolbox)
toolbox.register("evaluate", solution.evaluate, vessel)
toolbox.register("mate", recombination.crossover, toolbox,)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mutate", mutation.mutate, toolbox, swaps)
toolbox.register("individual", init.init_individual, creator.Individual, toolbox, init.graph_route(start, end, vessel))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def main(seed=None):
    random.seed(seed)

    NGEN = 500
    MU = 20
    CXPB = 1

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    # print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop))

    return pop, logbook


if __name__ == "__main__":
    population, statistics = main(1)
    population.sort(key=lambda x: x.fitness.values)

    output_file_name = 'sorted_population'
    with open('output/' + output_file_name, 'wb') as file:
        pickle.dump(population, file)

    # print(statistics)

    # Plot first individual of population
    print(population[0].fitness.values)
    plot_on_gshhs(population[-1])
    plt.show()

    front = np.array([ind.fitness.values for ind in population])
    plt.scatter(front[:, 0], front[:, 1], c="b")
    plt.axis("tight")

