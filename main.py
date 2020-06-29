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
from deap import base, creator, tools
from shapely.prepared import prep


class Vessel:
    def __init__(self, name='Fairmaster'):
        self.name = name
        table = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name=self.name)
        self.speeds = [round(speed, 1) for speed in table['Speed']]
        self.fuel_rates = {speed: round(table['Fuel'][i], 1) for i, speed in enumerate(self.speeds)}


class RoutePlanner:
    def __init__(self,
                 start,
                 end,
                 resolution='c',
                 max_poly_size=4,
                 n_gen=100,
                 mu=4 * 20,
                 vessel_name='Fairmaster',
                 cx_prob=0.9
                 ):

        # Create Fitness and Individual types
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Create toolbox for function aliases
        self.toolbox = base.Toolbox()

        self.start = start
        self.end = end
        self.geod = pyproj.Geod(a=6378137.0, f=0.0033528106647475126)
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
        self.evaluator = evaluation.Evaluator(self.vessel, self.prep_polys, self.rtree_idx, self.geod)
        self.toolbox.register("edge_feasible", self.evaluator.edge_feasible)
        self.toolbox.register("feasible", self.evaluator.feasible)
        self.toolbox.register("evaluate", self.evaluator.evaluate)
        self.toolbox.decorate("evaluate", tools.DeltaPenalty(self.toolbox.feasible, [math.inf, math.inf]))

        # Initialize "Initializer" and register it's functions
        self.initializer = initialization.Initializer(self.start, self.end, self.geod, self.vessel, self.resolution,
                                                      self.prep_polys, self.rtree_idx, self.toolbox)
        self.toolbox.register("global_routes", self.initializer.get_global_routes, creator.Individual)

        # Initialize "Operator" and register it's functions
        self.operators = operations.Operators(self.toolbox, self.vessel)
        self.toolbox.register("mutate", self.operators.mutate)
        self.toolbox.register("mate", self.operators.crossover)

        # Register NSGA2 functions and set parameter settings
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("population", tools.initRepeat, list)
        self.cx_prob = cx_prob
        self.n_gen = n_gen
        self.mu = mu

    def nsga2(self, seed=None):
        random.seed(seed)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(key=len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean, axis=0)
        mstats.register("std", np.std, axis=0)
        mstats.register("min", np.min, axis=0)
        mstats.register("max", np.max, axis=0)

        glob_routes, n_paths = self.toolbox.global_routes()

        all_pops, all_logs = [], []
        for gl, glob_route in enumerate(glob_routes):
            print('----------------- Computing path {0} of {1} -----------------'.format(gl + 1, len(glob_routes)))
            path_pops, path_logs = [], []
            for su, sub_route in enumerate(glob_route):
                print('----------------- Computing sub path {0} of {1} -----------------'.format(su + 1,
                                                                                                 len(glob_route)))

                self.toolbox.register("individual", initialization.init_individual, self.toolbox, sub_route)

                logbook = tools.Logbook()
                logbook.header = "gen", "evals", "fitness", "size"
                logbook.chapters["fitness"].header = "min", "avg", "max"
                logbook.chapters["size"].header = "min", "avg", "max"

                pop = self.toolbox.population(self.toolbox.individual, self.mu)

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # This is just to assign the crowding distance to the individuals
                # no actual selection is done
                pop = self.toolbox.select(pop, len(pop))

                record = mstats.compile(pop)
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
                    record = mstats.compile(pop)
                    logbook.record(gen=gen, evals=len(invalid_ind), **record)
                    print(logbook.stream)

                path_pops.append(pop)
                path_logs.append(logbook)
                self.toolbox.unregister("individual")

            all_pops.append(path_pops)
            all_logs.append(path_logs)

        return all_pops, all_logs


if __name__ == "__main__":
    # _start, _end = (3.14516, 4.68508), (-94.5968, 26.7012)  # Gulf of Guinea, Gulf of Mexico
    # _start, _end = (-23.4166, -7.2574), (-72.3352, 12.8774)  # South Atlantic (Brazil), Caribbean Sea
    # _start, _end = (77.7962, 4.90087), (48.1425, 12.5489)  # Laccadive Sea, Gulf of Aden
    # _start, _end = (-5.352121, 48.021295), (-53.306878, 46.423969)  # Normandy, Canada
    _start, _end = (34.252773, 43.461197), (53.131866, 13.521350)  # Black of Sea, Gulf of Aden

    planner = RoutePlanner(_start, _end)
    paths_populations, paths_statistics = planner.nsga2(seed=1)

    all_best_individuals, sub_path_pop = [], None
    timestamp = datetime.now()
    for p, path_population in enumerate(paths_populations):
        best_individual = []
        for s, sub_path_pop in enumerate(path_population):
            statistics = paths_statistics[p][s]
            sub_path_pop.sort(key=lambda x: x.fitness.values)

            output_file_name = '{0:%H_%M_%S}_population_p{1}_sp{2}'.format(timestamp, p, s)
            with open('output/paths/' + output_file_name, 'wb') as file:
                pickle.dump(sub_path_pop, file)
            print('Saved path {}, sub path {} to output/{}'.format(p, s, output_file_name))
            # Select sub path with least fitness values
            best_individual.append(sub_path_pop[0])
            print(sub_path_pop[0].fitness.values)

        print('Path {}: '.format(p), tuple(sum([np.array(path.fitness.values) for path in best_individual])))
        all_best_individuals.extend(best_individual)

    for logs in paths_statistics:
        for logbook in logs:
            gen = logbook.select("gen")
            fit_mins = logbook.chapters["fitness"].select("min")
            size_avgs = logbook.chapters["size"].select("avg")

            fig, ax1 = plt.subplots()
            line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Fitness", color="b")
            for tl in ax1.get_yticklabels():
                tl.set_color("b")

            ax2 = ax1.twinx()
            line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
            ax2.set_ylabel("Size", color="r")
            for tl in ax2.get_yticklabels():
                tl.set_color("r")

            lns = line1 + line2
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc="center right")

    plt.show()

    # front = np.array([ind.fitness.values for ind in sub_path_pop])
    # plt.scatter(front[:, 0], front[:, 1], c="b")
    # plt.axis("tight")
    # plt.show()
