import great_circle
import init
import matplotlib.pyplot as plt
import mutation
import numpy as np
import ocean_current
import os
import pandas as pd
import pickle
import pyproj
import random
import recombination
import rtree

from datetime import datetime
from deap import base, creator, tools
from katana import get_split_polygons
from shapely.geometry import LineString
from shapely.prepared import prep


class Vessel:
    def __init__(self, name, _speeds, _fuel_rates):
        self.name = name
        self.speeds = _speeds
        self.fuel_rates = _fuel_rates


class RoutePlanner:
    def __init__(self,
                 start,
                 end,
                 start_date='20160101',
                 del_s=67,
                 width_ratio=0.5,
                 radius=1,
                 mutation_ops=None,
                 resolution='c',
                 max_poly_size=4,
                 n_gen=100,
                 mu=4 * 20,
                 cx_prob=0.9,
                 vessel=None,
                 g_density=6,
                 g_var_density=4,
                 include_currents=True,
                 a=6378137.0,
                 f=0.0033528106647475126
                 ):
        self.start = start                                              # Start location
        self.end = end                                                  # End location
        self.date = start_date                                          # Start date for initializing ocean currents
        self.dist_cache = dict()
        self.feas_cache = dict()
        self.points_cache = dict()
        self.geod = pyproj.Geod(a=a, f=f)
        self.del_s = del_s                                              # Maximum segment length (nautical miles)
        self.g_density = g_density                                      # Density recursion number, graph
        self.g_var_density = g_var_density                              # Variable density recursion number, graph
        self.width_ratio = width_ratio                                  # Parameter for insert_random_waypoint
        self.radius = radius                                            # Parameter for move_random_waypoint
        if mutation_ops is None:                                        # To be performed mutation operators
            self.mutation_ops = ['insert', 'move', 'delete', 'speed']
        else:
            self.mutation_ops = mutation_ops
        self.resolution = resolution                                    # Resolution of shorelines
        self.max_poly_size = max_poly_size                              # Parameter for split_polygon
        if vessel is None:                                              # Set vessel characteristics
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
        self.rtree_idx = rtree.index.Index()
        for pos, polygon in enumerate(split_polys):
            self.rtree_idx.insert(pos, polygon.bounds)

        # Get ocean current variables
        self.include_currents = include_currents
        if include_currents:
            self.u, self.v, self.lons, self.lats = ocean_current.read_netcdf(self.date)
        else:
            self.u, self.v, self.lons, self.lats = None, None, None, None

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
        self.toolbox.register("global_routes", init.get_global_routes, creator.Individual, self.resolution,
                              self.g_density, self.g_var_density, self.prepared_polygons, self.rtree_idx,
                              self.start, self.end, self.vessel)
        self.toolbox.register("population", tools.initRepeat, list)

        # Genetic/Evolutionary algorithm parameter settings
        self.n_gen = n_gen
        self.mu = mu
        self.cx_prob = cx_prob

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
                    offspring = list(self.toolbox.map(self.toolbox.clone, offspring))
                    # offspring = [self.toolbox.clone(ind) for ind in offspring]

                    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() <= self.cx_prob:
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
        travel_time = fuel_consumption = 0.0

        for e in range(len(individual) - 1):
            p1, p2, boat_speed = individual[e][0], individual[e+1][0], individual[e][1]
            k = tuple(sorted([p1, p2]))

            e_dist = self.dist_cache.get(k, False)
            if not e_dist:  # Never steps in IF-statement
                print('computes distance')
                e_dist = great_circle.distance(p1[0], p1[1], p2[0], p2[1], self.geod)
                self.dist_cache[k] = e_dist

            if self.include_currents:
                # Split edge in segments (seg) of max seg_length in km
                points = self.points_cache.get(k, False)
                if not points:  # Never steps in IF-statement
                    print('computes points')
                    points = great_circle.points(p1[0], p1[1], p2[0], p2[1], e_dist, self.geod, self.del_s)
                    self.points_cache[k] = points
                lons, lats = points[0], points[1]
                e_travel_time = 0.0
                for i in range(len(lons)-1):
                    p1, p2 = (lons[i], lats[i]), (lons[i+1], lats[i+1])
                    seg_travel_time = ocean_current.get_edge_travel_time(p1, p2, boat_speed, e_dist, self.u, self.v,
                                                                         self.lons, self.lats)
                    e_travel_time += seg_travel_time
            else:
                e_travel_time = e_dist / boat_speed
            edge_fuel_consumption = self.vessel.fuel_rates[boat_speed] * e_travel_time  # Tons

            # Increment objective values
            travel_time += e_travel_time
            fuel_consumption += edge_fuel_consumption

        return travel_time, fuel_consumption

    def feasible(self, individual):
        for i in range(len(individual) - 1):
            p1, p2 = individual[i][0], individual[i+1][0]
            if not self.edge_feasible(p1, p2):
                return False
        return True

    def edge_feasible(self, p1, p2):
        # First check if feasibility check is already performed
        k = tuple(sorted([p1, p2]))
        feasible = self.feas_cache.get(k, None)
        if feasible == 1:
            return True
        elif feasible == 0:
            return False

        dist = self.dist_cache.get(k, False)
        if not dist:
            dist = great_circle.distance(p1[0], p1[1], p2[0], p2[1], self.geod)
            self.dist_cache[k] = dist

        points = self.points_cache.get(k, False)
        if not points:
            points = great_circle.points(p1[0], p1[1], p2[0], p2[1], dist, self.geod, self.del_s)
            self.points_cache[k] = points
        lons, lats = points[0], points[1]
        for i in range(len(lons)-1):
            # Compute line bounds
            q1_x, q1_y = lons[i], lats[i]
            q2_x, q2_y = lons[i+1], lats[i+1]
            line_bounds = (min(q1_x, q2_x), min(q1_y, q2_y), max(q1_x, q2_x), max(q1_y, q2_y))

            # Returns the geometry indices of the minimum bounding rectangles of polygons that intersect the edge bounds
            mbr_intersections = self.rtree_idx.intersection(line_bounds)
            if mbr_intersections:
                # Create LineString if there is at least one minimum bounding rectangle intersection
                shapely_line = LineString([(q1_x, q1_y), (q2_x, q2_y)])

                # For every mbr intersection check if its polygon is actually intersect by the edge
                for idx in mbr_intersections:
                    if self.prepared_polygons[idx].intersects(shapely_line):
                        self.feas_cache[k] = 0
                        return False
        self.feas_cache[k] = 1
        return True


if __name__ == "__main__":
    # start, end = (3.14516, 4.68508), (-94.5968, 26.7012)  # Gulf of Guinea, Gulf of Mexico
    # start, end = (-23.4166, -7.2574), (-72.3352, 12.8774)  # South Atlantic (Brazil), Caribbean Sea
    # start, end = (77.7962, 4.90087), (48.1425, 12.5489)  # Laccadive Sea, Gulf of Aden
    # start, end = (-5.352121, 48.021295), (53.131866, 13.521350)  #
    start, end = (34.252773, 43.461197), (53.131866, 13.521350)  # Black of Sea, Gulf of Aden

    planner = RoutePlanner(start, end, include_currents=False, n_gen=200)
    paths_populations, paths_statistics = planner.nsga2(seed=1)

    all_best_individuals, sub_path_pop = [], None
    timestamp = datetime.now()
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

            # Select sub path with least fitness values
            best_individual.append(sub_path_pop[0])

        print('Path {}: '.format(p), tuple(sum([np.array(path.fitness.values) for path in best_individual])))
        all_best_individuals.extend(best_individual)

    front = np.array([ind.fitness.values for ind in sub_path_pop])
    plt.scatter(front[:, 0], front[:, 1], c="b")
    plt.axis("tight")
    plt.show()
