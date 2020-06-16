import matplotlib.pyplot as plt
import fiona
import init
import mutation
import numpy as np
import pandas as pd
import pickle
import random
import recombination

from deap.benchmarks.tools import hypervolume
from deap import base, creator, tools
from plot.plot_on_GSHHS import plot_on_gshhs
from rtree import index
from shapely.geometry import shape, LineString
from shapely.prepared import prep
from haversine import haversine


class Vessel:
    def __init__(self, name, _speeds, _fuel_rates):
        self.name = name
        self.speeds = _speeds
        self.fuel_rates = _fuel_rates


# Vessel characteristics
vessel_name = 'Fairmaster'
table = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name=vessel_name)
speeds = [round(speed, 1) for speed in table['Speed']]
fuel_rates = {speed: round(table['Fuel'][i], 1) for i, speed in enumerate(speeds)}
vessel = Vessel(vessel_name, speeds, fuel_rates)

distances_cache = dict()
feasible_cache = dict()

max_edge_length = 500  # nautical miles


def distance(individual):
    global distances_cache
    waypoints = [item[0] for item in individual]
    distances = []

    for u, v in zip(waypoints[:-1], waypoints[1:]):
        if u[0] < v[0]:
            key = (u, v)
        else:
            key = (v, u)

        if key in distances_cache:
            edge_distance = distances_cache[key]
        else:
            edge_distance = haversine(u, v, unit='nmi')
            distances_cache[key] = edge_distance

        distances.append(edge_distance)



    return sum(distances), max_edge_length


def evaluate(individual):
    global distances_cache
    global vessel

    # Initialize variables
    travel_time, fuel_consumption = 0, 0

    for i in range(len(individual) - 1):
        u = individual[i][0]
        v = individual[i+1][0]
        speed = individual[i][1]

        if u[0] < v[0]:
            key = (u, v)
        else:
            key = (v, u)

        # If distance in cache, get distance. Otherwise, calculate distance and save to cache
        if key in distances_cache:
            edge_distance = distances_cache[key]
        else:
            edge_distance = haversine(u, v, unit='nmi')
            distances_cache[key] = edge_distance

        edge_travel_time = edge_distance / speed  # Hours
        fuel_rate = vessel.fuel_rates[speed]  # Tons / Hour
        # fuel_rate = 30
        edge_fuel_consumption = fuel_rate * edge_travel_time  # Tons

        # Increment objective values
        travel_time += edge_travel_time
        fuel_consumption += edge_fuel_consumption

    return travel_time, fuel_consumption


def feasible(tb, individual):
    global feasible_cache
    waypoints = [item[0] for item in individual]

    for u, v in zip(waypoints[:-1], waypoints[1:]):
        if u[0] < v[0]:
            key = (u, v)
        else:
            key = (v, u)
        if key in feasible_cache:
            if not feasible_cache[key]:
                return False
        else:
            if not tb.edge_feasible(u, v):
                return False
    return True


def edge_feasible(rtree_i, prep_geoms, max_distance, u, v):
    global distances_cache
    global feasible_cache

    if u[0] < v[0]:
        key = (u, v)
    else:
        key = (v, u)

    # If distance in cache, get distance. Otherwise, calculate distance and save to cache
    if key in distances_cache:
        distance = distances_cache[key]
    else:
        distance = haversine(u, v, unit='nmi')
        distances_cache[key] = distance

    # If distance is larger than maximum edge length, return infeasible
    if distance > max_distance:
        feasible_cache[key] = False
        return False

    # Compute line bounds
    u_x, u_y = u
    v_x, v_y = v
    line_bounds = (min(u_x, v_x), min(u_y, v_y),
                   max(u_x, v_x), max(u_y, v_y))

    # Returns the geometry indices of the minimum bounding rectangles of polygons that intersect the edge bounds
    mbr_intersections = rtree_i.intersection(line_bounds)
    if mbr_intersections:
        # Create LineString if there is at least one minimum bounding rectangle intersection
        shapely_line = LineString([u, v])

        # For every mbr intersection check if its polygon is actually intersect by the edge
        for i in mbr_intersections:
            if prep_geoms[i].intersects(shapely_line):
                feasible_cache[key] = False
                return False
    feasible_cache[key] = True
    return True


width_ratio = 0.5
radius = 1
swaps = ['insert', 'move', 'delete', 'speed']

# Route characteristics and navigation area
start, end = (-5.352121, 48.021295), (53.131866, 13.521350)  # (longitude, latitude)
polygons = [shape(shoreline['geometry']) for shoreline in
            iter(fiona.open('C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp'))]
prepared_polygons = [prep(polygon) for polygon in polygons]

# Populate R-tree index with bounds of polygons
rtree_idx = index.Index()
for pos, polygon in enumerate(polygons):
    rtree_idx.insert(pos, polygon.bounds)


# Create Fitness and Indiviul types
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Register function aliases
toolbox = base.Toolbox()
toolbox.register("distance", distance)
toolbox.register("edge_feasible", edge_feasible, rtree_idx, polygons, max_edge_length)
toolbox.register("insert", mutation.insert_waypoint, toolbox, width_ratio)
toolbox.register("delete", mutation.delete_random_waypoints, toolbox)
toolbox.register("speed", mutation.change_speed, vessel)
toolbox.register("move", mutation.move_waypoints, toolbox, radius)
toolbox.register("feasible", feasible, toolbox)
toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenalty(toolbox.feasible, [99999999999, 9999999999]))
toolbox.register("mate", recombination.crossover, toolbox,)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mutate", mutation.mutate, toolbox, swaps)
toolbox.register("individual", init.init_individual, toolbox, init.graph_route(creator.Individual, start, end, vessel))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def main(seed=None):
    random.seed(seed)

    NGEN = 100
    MU = 4 * 20
    CXPB = 0.9

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
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        # offspring = toolbox.map(toolbox.clone, offspring)
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

    output_file_name = 'sorted_population1'
    with open('output/' + output_file_name, 'wb') as file:
        pickle.dump(population, file)

    # print(statistics)
    waypoints = [item[0] for item in population[0]]
    distances = []
    for u, v in zip(waypoints[:-1], waypoints[1:]):
        edge_distance = haversine(u, v, unit='nmi')
        distances.append(edge_distance)
    print(sum(distances) / len(distances))

    # Plot first individual of population
    print(population[0].fitness.values)
    plot_on_gshhs(population[-1])
    plt.show()

    front = np.array([ind.fitness.values for ind in population])
    plt.scatter(front[:, 0], front[:, 1], c="b")
    plt.axis("tight")

