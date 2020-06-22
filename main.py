import datetime
import matplotlib.pyplot as plt
import init
import mutation
import numpy as np
import pandas as pd
import pickle
import random
import recombination
import os
import fiona
from katana import get_split_polygons

from deap import base, creator, tools
from rtree import index
from shapely.geometry import shape, LineString
from shapely.prepared import prep
from haversine import haversine


class Vessel:
    def __init__(self, name, _speeds, _fuel_rates):
        self.name = name
        self.speeds = _speeds
        self.fuel_rates = _fuel_rates


class RoutePlanner:
    def __init__(self,
                 start,
                 end,
                 max_edge_length=500,
                 width_ratio=0.5,
                 radius=1,
                 mutation_ops=None,
                 resolution='c',
                 max_poly_size=4,
                 n_gen=100,
                 mu=4 * 20,
                 cxpb=0.9,
                 vessel=None
                 ):
        self.start = start                                              # Start location
        self.end = end                                                   # End location
        self.distances_cache = dict()
        self.feasible_cache = dict()
        self.max_edge_length = max_edge_length                          # TO BE REMOVED
        self.width_ratio = width_ratio                                  # Parameter for insert_random_waypoint
        self.radius = radius                                            # Parameter for move_random_waypoint

        if mutation_ops is None:                                        # To be performed mutation operators
            self.mutation_ops = ['insert', 'move', 'delete', 'speed']
        else:
            self.mutation_ops = mutation_ops
        self.resolution = resolution                                    # Resolution of shorelines
        self.max_poly_size = max_poly_size                              # Parameter for split_polygon

        if vessel is None:
            # Set vessel characteristics
            vessel_name = 'Fairmaster'
            table = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name=vessel_name)
            speeds = [round(speed, 1) for speed in table['Speed']]
            fuel_rates = {speed: round(table['Fuel'][i], 1) for i, speed in enumerate(speeds)}
            self.vessel = Vessel(vessel_name, speeds, fuel_rates)
        else:
            self.vessel = vessel

        # Import land obstacles as polygons
        try:
            with open('output/split_polygons/res_{0}_treshold_{1}'.format(self.resolution, self.max_poly_size),
                      'rb') as f:
                split_polys = pickle.load(f)
        except FileNotFoundError:
            split_polys = get_split_polygons(self.resolution, self.max_poly_size)

        self.prepared_polygons = [prep(polygon) for polygon in split_polys]

        # Populate R-tree index with bounds of polygons
        self.rtree_idx = index.Index()
        for pos, polygon in enumerate(split_polys):
            self.rtree_idx.insert(pos, polygon.bounds)

        # Create Fitness and Individual types
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # --Register function aliases
        self.toolbox = base.Toolbox()

        # Feasibility
        self.toolbox.register("edge_feasible", self.edge_feasible)

        # Mutation
        self.toolbox.register("insert", mutation.insert_waypoint, self.toolbox, self.width_ratio)
        self.toolbox.register("delete", mutation.delete_random_waypoints, self.toolbox)
        self.toolbox.register("speed", mutation.change_speed, self.vessel)
        self.toolbox.register("move", mutation.move_waypoints, self.toolbox, self.radius)
        self.toolbox.register("mutate", mutation.mutate, self.toolbox, self.mutation_ops)

        # Evaluation
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.decorate("evaluate", tools.DeltaPenalty(self.feasible, [99999999999, 99999999999]))

        # Selection and crossover
        self.toolbox.register("mate", recombination.crossover, self.toolbox, )
        self.toolbox.register("select", tools.selNSGA2)

        # Initialization
        self.toolbox.register("global_routes", init.get_global_routes, creator.Individual,
                              self.start, self.end, self.vessel)
        self.toolbox.register("population", tools.initRepeat, list)

        # Genetic/Evolutionary algorithm parameter settings
        self.n_gen = n_gen
        self.mu = mu
        self.cxpb = cxpb

    def nsga2(self, seed=None):
        random.seed(seed)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        glob_routes, n_paths = self.toolbox.global_routes()

        all_pops, all_logs = [], []
        for gl, glob_route in enumerate(glob_routes):
            print('----------------- Computing path {0} of {1} -----------------'.format(gl + 1, len(glob_routes)))
            path_pops, path_logs = [], []
            for su, sub_route in enumerate(glob_route):
                print('----------------- Computing sub path {0} of {1} -----------------'.format(su + 1,
                                                                                                 len(glob_route)))

                self.toolbox.register("individual", init.init_individual, self.toolbox, sub_route)

                logbook = tools.Logbook()
                logbook.header = "gen", "evals", "std", "min", "avg", "max"

                pop = self.toolbox.population(self.toolbox.individual, self.mu)

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # This is just to assign the crowding distance to the individuals
                # no actual selection is done
                pop = self.toolbox.select(pop, len(pop))

                record = stats.compile(pop)
                logbook.record(gen=0, evals=len(invalid_ind), **record)
                print(logbook.stream)

                # Begin the generational process
                for gen in range(1, self.n_gen):
                    # Vary the population
                    offspring = tools.selTournamentDCD(pop, len(pop))
                    # offspring = list(toolbox.map(toolbox.clone, offspring))
                    offspring = [self.toolbox.clone(ind) for ind in offspring]

                    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() <= self.cxpb:
                            self.toolbox.mate(ind1, ind2)
                        self.toolbox.mutate(ind1)
                        self.toolbox.mutate(ind2)
                        del ind1.fitness.values, ind2.fitness.values

                    # Evaluate the individuals with an invalid fitness
                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit

                    # Select the next generation population
                    pop = self.toolbox.select(pop + offspring, self.mu)
                    record = stats.compile(pop)
                    logbook.record(gen=gen, evals=len(invalid_ind), **record)
                    print(logbook.stream)

                path_pops.append(pop)
                path_logs.append(logbook)
                self.toolbox.unregister("individual")

            all_pops.append(path_pops)
            all_logs.append(path_logs)

        return all_pops, all_logs

    def evaluate(self, individual):
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
            if key in self.distances_cache:
                edge_distance = self.distances_cache[key]
            else:
                edge_distance = haversine(u, v, unit='nmi')
                self.distances_cache[key] = edge_distance

            edge_travel_time = edge_distance / speed  # Hours
            fuel_rate = self.vessel.fuel_rates[speed]  # Tons / Hour
            # fuel_rate = 30
            edge_fuel_consumption = fuel_rate * edge_travel_time  # Tons

            # Increment objective values
            travel_time += edge_travel_time
            fuel_consumption += edge_fuel_consumption

        return travel_time, fuel_consumption

    def feasible(self, individual):
        waypoints = [item[0] for item in individual]

        for u, v in zip(waypoints[:-1], waypoints[1:]):
            key = tuple(sorted([u, v]))
            # If edge in cache, get feasibility. Otherwise, calculate feasibility and save to cache
            if key in self.feasible_cache:
                if not self.feasible_cache[key]:
                    return False
            else:
                if not self.edge_feasible(u, v):
                    return False
        return True

    def edge_feasible(self, u, v):
        key = tuple(sorted([u, v]))
        # If distance in cache, get distance. Otherwise, calculate distance and save to cache
        if key in self.distances_cache:
            distance = self.distances_cache[key]
        else:
            distance = haversine(u, v, unit='nmi')
            self.distances_cache[key] = distance

        # If distance is larger than maximum edge length, return infeasible
        if distance > self.max_edge_length:
            self.feasible_cache[key] = False
            return False

        # Compute line bounds
        u_x, u_y = u
        v_x, v_y = v
        line_bounds = (min(u_x, v_x), min(u_y, v_y),
                       max(u_x, v_x), max(u_y, v_y))

        # Returns the geometry indices of the minimum bounding rectangles of polygons that intersect the edge bounds
        mbr_intersections = self.rtree_idx.intersection(line_bounds)
        if mbr_intersections:
            # Create LineString if there is at least one minimum bounding rectangle intersection
            shapely_line = LineString([u, v])

            # For every mbr intersection check if its polygon is actually intersect by the edge
            for i in mbr_intersections:
                if self.prepared_polygons[i].intersects(shapely_line):
                    self.feasible_cache[key] = False
                    return False
        self.feasible_cache[key] = True
        return True


if __name__ == "__main__":
    # Route characteristics and navigation area
    start, end = (-5.352121, 48.021295), (53.131866, 13.521350)  # (longitude, latitude)
    # start, end = (34.252773, 43.461197), (53.131866, 13.521350)  # (longitude, latitude)

    planner = RoutePlanner(start, end)
    paths_populations, paths_statistics = planner.nsga2(1)
    all_best_individuals = []
    timestamp = datetime.datetime.now()
    for p, path_population in enumerate(paths_populations):
        best_individual = []
        for s, sub_path_pop in enumerate(path_population):
            statistics = paths_statistics[p][s]
            sub_path_pop.sort(key=lambda x: x.fitness.values)

            output_file_name = 'path_{0}/{1:%H_%M_%S}_population_sub_path_{2}'.format(p, timestamp, s)
            try:
                with open('output/' + output_file_name, 'wb') as file:
                    pickle.dump(sub_path_pop, file)
            except FileNotFoundError:
                os.mkdir('output/path_{}'.format(p))
                with open('output/' + output_file_name, 'wb') as file:
                    pickle.dump(sub_path_pop, file)

            # print(statistics)

            # Select sub path with least fitness values
            best_individual.append(sub_path_pop[0])

        print('Path {}: '.format(p), tuple(sum([np.array(path.fitness.values) for path in best_individual])))
        all_best_individuals.extend(best_individual)

    # plot_paths(all_best_individuals, vessel)
    # plt.show()

    # front = np.array([ind.fitness.values for ind in sub_path_pop])
    # plt.scatter(front[:, 0], front[:, 1], c="b")
    # plt.axis("tight")
