import evaluation
import initialization
import katana
import math
import matplotlib.pyplot as plt
import operations
import numpy as np
import pandas as pd
import pyproj
import pickle
import random
import rtree

from datetime import datetime
from deap import base, creator, tools, algorithms
from shapely.prepared import prep


class Vessel:
    def __init__(self, name='Fairmaster'):
        self.name = name
        table = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name=self.name)
        self.speeds = [round(speed, 1) for speed in table['Speed']]
        self.fuel_rates = {speed: round(table['Fuel'][i], 1) for i, speed in enumerate(self.speeds)}


Nbar = 50
N = 100
GEN = 250
CXPB = 0.9
MUTPB = 0.9


class RoutePlanner:
    def __init__(self,
                 start=None,
                 end=None,
                 seca_factor=1.2,
                 resolution='c',
                 max_poly_size=4,
                 vessel_name='Fairmaster',
                 include_currents=True
                 ):

        # Create Fitness and Individual types
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Create toolbox for function aliases
        self.toolbox = base.Toolbox()

        self.start = start
        self.end = end
        self.seca_factor = seca_factor
        self.geod = pyproj.Geod(a=3443.918467, f=0.0033528106647475126)
        self.vessel = Vessel(vessel_name)                                   # Create vessel class
        self.resolution = resolution                                        # Resolution of shorelines
        self.max_poly_size = max_poly_size                                  # Parameter for split_polygon
        try:                                                                # Import land obstacles as polygons
            with open('output/split_polygons/res_{0}_treshold_{1}'.format(self.resolution, self.max_poly_size),
                      'rb') as f:
                split_polys = pickle.load(f)
        except FileNotFoundError:
            split_polys = katana.get_split_polygons(self.resolution, self.max_poly_size)
        self.prep_polys = [prep(poly) for poly in split_polys]              # Prepared and split land polygons

        # Populate R-tree index with bounds of polygons
        self.rtree_idx = rtree.index.Index()
        for idx, poly in enumerate(split_polys):
            self.rtree_idx.insert(idx, poly.bounds)

        # Initialize "Evaluator" and register it's functions
        self.evaluator = evaluation.Evaluator(self.vessel, self.prep_polys, self.rtree_idx, self.geod, self.seca_factor,
                                              include_currents=include_currents)
        self.toolbox.register("edge_feasible", self.evaluator.edge_feasible)
        self.toolbox.register("feasible", self.evaluator.feasible)
        self.toolbox.register("evaluate", self.evaluator.evaluate)
        self.toolbox.decorate("evaluate", tools.DeltaPenalty(self.toolbox.feasible, [math.inf, math.inf]))

        # Initialize "Initializer" and register it's functions
        if start and end:
            self.initializer = initialization.Initializer(self.start, self.end, self.geod, self.vessel, self.resolution,
                                                          self.prep_polys, self.rtree_idx, self.toolbox)
            self.toolbox.register("init_routes", self.initializer.get_init_routes, creator.Individual)
        else:
            print('No start and endpoint given')

        # Initialize "Operator" and register it's functions
        self.operators = operations.Operators(self.toolbox, self.vessel, self.geod, width_ratio=3, radius=5)
        self.toolbox.register("mutate", self.operators.mutate)
        self.toolbox.register("mate", self.operators.crossover)

        self.toolbox.register("population", tools.initRepeat, list)

    def spea2(self, seed=None):
        # Register SPEA2 selection function
        self.toolbox.register("select", tools.selSPEA2)

        random.seed(seed)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(key=len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean, axis=0)
        mstats.register("std", np.std, axis=0)
        mstats.register("min", np.min, axis=0)
        mstats.register("max", np.max, axis=0)

        init_routes = self.toolbox.init_routes()

        paths, path_logs = {}, {}
        for p_idx, init_route in enumerate(init_routes):
            print('----------------- Path {0}/{1} -----------------'.format(p_idx+1, len(init_routes)))
            sub_paths, sub_path_logs = {}, {}
            for sp_idx, sub_route in enumerate(init_route):
                print('----------------- Sub path {0}/{1} -----------------'.format(sp_idx+1, len(init_route)))

                self.toolbox.register("individual", initialization.init_individual, self.toolbox, sub_route)

                log = tools.Logbook()
                log.header = "gen", "evals", "fitness", "size"
                log.chapters["fitness"].header = "min", "avg", "max"
                log.chapters["size"].header = "min", "avg", "max"

                # Step 1: Initialization
                pop = self.toolbox.population(self.toolbox.individual, N)
                archive = []
                curr_gen = 1

                # Step 2: Fitness assignment
                invalid_inds1, invalid_inds2 = pop, []
                fits = self.toolbox.map(self.toolbox.evaluate, invalid_inds1)
                for ind, fit in zip(invalid_inds1, fits):
                    ind.fitness.values = fit

                # Begin the generational process
                while True:
                    # Step 3: Environmental selection
                    archive = self.toolbox.select(pop + archive, k=Nbar)

                    # Record statistics
                    record = mstats.compile(archive)
                    log.record(gen=curr_gen, evals=(len(invalid_inds1) + len(invalid_inds2)), **record)
                    print(log.stream)

                    # Step 4: Termination
                    if curr_gen >= GEN:
                        sub_paths[sp_idx] = archive
                        sub_path_logs[sp_idx] = log
                        self.toolbox.unregister("individual")
                        break

                    # Step 5: Mating Selection
                    mating_pool = tools.selTournament(archive, k=N, tournsize=2)

                    # Step 6: Variation
                    pop = algorithms.varAnd(mating_pool, self.toolbox, CXPB, MUTPB)

                    # Step 2: Fitness assignment
                    # Population
                    invalid_inds1 = [ind for ind in pop if not ind.fitness.valid]
                    fits = self.toolbox.map(self.toolbox.evaluate, invalid_inds1)
                    for ind, fit in zip(invalid_inds1, fits):
                        ind.fitness.values = fit

                    # Archive
                    invalid_inds2 = [ind for ind in archive if not ind.fitness.valid]
                    fits = self.toolbox.map(self.toolbox.evaluate, invalid_inds2)
                    for ind, fit in zip(invalid_inds2, fits):
                        ind.fitness.values = fit

                    curr_gen += 1

            paths[p_idx] = sub_paths
            path_logs[p_idx] = sub_path_logs

        return paths, path_logs, init_routes

    def nsga2(self, seed=None):
        # Register NSGA2 selection function
        self.toolbox.register("select", tools.selNSGA2)

        random.seed(seed)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(key=len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean, axis=0)
        mstats.register("std", np.std, axis=0)
        mstats.register("min", np.min, axis=0)
        mstats.register("max", np.max, axis=0)

        init_routes = self.toolbox.init_routes()

        paths, path_logs = {}, {}
        for p_idx, init_route in enumerate(init_routes):
            print('----------------- Path {0}/{1} -----------------'.format(p_idx+1, len(init_routes)))
            sub_paths, sub_path_logs = {}, {}
            for sp_idx, sub_route in enumerate(init_route):
                print('----------------- Sub path {0}/{1} -----------------'.format(sp_idx+1, len(init_route)))

                self.toolbox.register("individual", initialization.init_individual, self.toolbox, sub_route)

                log = tools.Logbook()
                log.header = "gen", "evals", "fitness", "size"
                log.chapters["fitness"].header = "min", "avg", "max"
                log.chapters["size"].header = "min", "avg", "max"

                # Step 1 Initialization
                pop = self.toolbox.population(self.toolbox.individual, N)
                offspring = []
                curr_gen = 1

                # Step 2: Fitness assignment
                fits = self.toolbox.map(self.toolbox.evaluate, pop)
                for ind, fit in zip(pop, fits):
                    ind.fitness.values = fit

                # Begin the generational process
                while True:
                    # Step 3: Environmental selection
                    pop = self.toolbox.select(pop + offspring, N)

                    # Record statistics
                    record = mstats.compile(pop)
                    log.record(gen=curr_gen, evals=len(pop), **record)
                    print(log.stream)

                    # Step 4: Termination
                    if curr_gen >= GEN:
                        sub_paths[sp_idx] = pop
                        sub_path_logs[sp_idx] = log
                        self.toolbox.unregister("individual")
                        break

                    # Step 5: Variation
                    offspring = algorithms.varAnd(pop, self.toolbox, CXPB, MUTPB)

                    # Step 2: Fitness assignment
                    invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
                    fits = self.toolbox.map(self.toolbox.evaluate, invalid_inds)
                    for ind, fit in zip(invalid_inds, fits):
                        ind.fitness.values = fit

                    curr_gen += 1

            paths[p_idx] = sub_paths
            path_logs[p_idx] = sub_path_logs

        return paths, path_logs, init_routes


if __name__ == "__main__":
    _start, _end = (3.14516, 4.68508), (-94.5968, 26.7012)  # Gulf of Guinea, Gulf of Mexico
    # _start, _end = (-23.4166, -7.2574), (-72.3352, 12.8774)  # South Atlantic (Brazil), Caribbean Sea
    # _start, _end = (77.7962, 4.90087), (48.1425, 12.5489)  # Laccadive Sea, Gulf of Aden
    # _start, _end = (-5.352121, 48.021295), (-53.306878, 46.423969)  # Normandy, Canada
    # _start, _end = (34.252773, 43.461197), (53.131866, 13.521350)  # Black of Sea, Gulf of Aden
    # _start, _end = (20.891193, 58.464147), (-85.063585, 29.175463)  # Gulf of Bothnia, Gulf of Mexico
    # _start, _end = (3.891292, 60.088472), (-7.562237, 47.403357)  # North UK, South UK

    _route_planner = RoutePlanner(_start, _end, seca_factor=1.2, include_currents=False)
    _paths, _path_logs, _init_routes = _route_planner.spea2(seed=1)

    # Save parameters
    timestamp = datetime.now()

    # Save initial routes
    init_routes_fn = '{0:%H_%M_%S}_init_routes'.format(timestamp)
    with open('output/glob_routes/' + init_routes_fn, 'wb') as file:
        pickle.dump(_init_routes, file)
    print('Saved initial routes to: ' + 'output/glob_routes/' + init_routes_fn)

    # Save logs
    logs_fn = '{:%H_%M_%S}_logs'.format(timestamp)
    with open('output/logs/' + logs_fn, 'wb') as file:
        pickle.dump(_path_logs, file)
    print('Saved logs to output/logs/{}'.format(logs_fn))

    # Save paths
    paths_fn = '{0:%H_%M_%S}_paths'.format(timestamp)
    with open('output/paths/' + paths_fn, 'wb') as file:
        pickle.dump(_paths, file)
    print('Saved paths to "output/paths/{}"'.format(paths_fn))

    # Plot statistics
    for p, path_log in enumerate(_path_logs.values()):
        for sp, sub_path_log in enumerate(path_log.values()):
            _gen = sub_path_log.select("gen")
            fit_mins = sub_path_log.chapters["fitness"].select("min")
            time_mins = [fit_min[0] for fit_min in fit_mins]
            fuel_mins = [fit_min[1] for fit_min in fit_mins]
            path_time_min = fit_mins[time_mins.index(min(time_mins))]
            path_fuel_min = fit_mins[fuel_mins.index(min(fuel_mins))]
            print('P{0} SP{1}: Min Time {2}'.format(p, sp, path_time_min))
            print('P{0} SP{1}: Min Fuel {2}'.format(p, sp, path_fuel_min))
            size_avgs = sub_path_log.chapters["size"].select("avg")

            fig, ax1 = plt.subplots()
            line1 = ax1.plot(_gen, fit_mins, "b-", label="Minimum Fitness")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Fitness", color="b")
            for tl in ax1.get_yticklabels():
                tl.set_color("b")

            ax2 = ax1.twinx()
            line2 = ax2.plot(_gen, size_avgs, "r-", label="Average Size")
            ax2.set_ylabel("Size", color="r")
            for tl in ax2.get_yticklabels():
                tl.set_color("r")

            lines = line1 + line2
            labs = [line.get_label() for line in lines]
            ax1.legend(lines, labs, loc="center right")

    # front = np.array([ind.fitness.values for ind in sub_path_pop])
    # plt.scatter(front[:, 0], front[:, 1], c="b")
    # plt.axis("tight")
    plt.show()
