import evaluation
import great_circle
import initialization
import katana
import matplotlib.pyplot as plt
import ocean_current
import operations
import os
import math
import numpy as np
import pandas as pd
import pickle
import random
import rtree
import time

from datetime import datetime
from deap import base, creator, tools, algorithms
from shapely.prepared import prep


class Vessel:
    def __init__(self, name='Fairmaster'):
        self.name = name
        table = pd.read_excel('C:/dev/data/speed_table.xlsx',
                              sheet_name=self.name)
        self.speeds = [round(speed, 1) for speed in table['Speed']]
        self.fuel_rates = {speed: round(table['Fuel'][i], 1)
                           for i, speed in enumerate(self.speeds)}


Nbar = 50
# GEN = 250
# N = 100
N = 20
GEN = 20
CXPB = 0.9
MUTPB = 0.9


class RoutePlanner:
    def __init__(self,
                 start=None,
                 end=None,
                 start_date=datetime(2016, 1, 2),
                 eca_f=1.2,
                 res='c',
                 spl_th=4,
                 vessel_name='Fairmaster',
                 incl_curr=True
                 ):

        # Create Fitness and Individual types
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.tb = base.Toolbox()              # Function toolbox
        self.start = start                    # Start point
        self.end = end                        # End point
        self.start_date = start_date          # Start date of voyage
        self.eca_f = eca_f                  # Multiplication factor ECA fuel
        self.vessel = Vessel(vessel_name)     # Vessel class instance
        self.res = res                        # Resolution of shorelines
        self.spl_th = spl_th                  # Threshold for split_polygon
        self.incl_curr = incl_curr            # Boolean for including currents
        self.gc = great_circle.GreatCircle()  # Geod class instance

        # Import land obstacles as polygons
        try:
            spl_dir = 'output/split_polygons/'
            spl_fn = 'res_{0}_threshold_{1}'.format(res, spl_th)
            spl_fp = os.path.join(spl_dir, spl_fn)
            with open(spl_fp, 'rb') as f:
                spl_polys = pickle.load(f)
        except FileNotFoundError:
            spl_polys = katana.get_split_polygons(self.res, self.spl_th)

        # Prepared and split land polygons
        self.prep_polys = [prep(poly) for poly in spl_polys]

        # Populate R-tree index with bounds of polygons
        self.rtree_idx = rtree.index.Index()
        for idx, poly in enumerate(spl_polys):
            self.rtree_idx.insert(idx, poly.bounds)

        # Load eca polygons
        with open('C:/dev/data/seca_areas_csv', 'rb') as f:
            self.ecas = pickle.load(f)

        # Populate R-tree index with bounds of ECA polygons
        self.rtree_idx_eca = rtree.index.Index()
        for idx, eca in enumerate(self.ecas):
            self.rtree_idx_eca.insert(idx, eca.bounds)

        # Initialize "Evaluator" and register it's functions
        self.evaluator = evaluation.Evaluator(self.vessel,
                                              self.prep_polys,
                                              self.rtree_idx,
                                              self.ecas,
                                              self.rtree_idx_eca,
                                              eca_f,
                                              start_date,
                                              self.gc,
                                              incl_curr)
        self.tb.register("e_feasible", self.evaluator.e_feasible)
        self.tb.register("feasible", self.evaluator.feasible)
        self.tb.register("evaluate", self.evaluator.evaluate)
        self.tb.decorate("evaluate", tools.DeltaPenalty(self.tb.feasible,
                                                        [1e+20, 1e+20]))

        # Initialize "Initializer" and register it's functions
        if start and end:
            self.initializer = initialization.Initializer(start, end,
                                                          self.vessel, res,
                                                          self.prep_polys,
                                                          self.rtree_idx,
                                                          self.ecas,
                                                          self.rtree_idx_eca,
                                                          self.tb,
                                                          self.gc)
            self.tb.register("get_shortest_paths",
                             self.initializer.get_shortest_paths,
                             creator.Individual)
        else:
            print('No start and endpoint given')

        # Initialize "Operator" and register it's functions
        self.operators = operations.Operators(self.tb, self.vessel)
        self.tb.register("mutate", self.operators.mutate)
        self.tb.register("mate", self.operators.cx_one_point)

        self.tb.register("population", initialization.init_repeat_list)

        # Initialize Statistics
        stats_fit = tools.Statistics(lambda _ind: _ind.fitness.values)
        stats_size = tools.Statistics(key=len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", np.mean, axis=0)
        self.mstats.register("std", np.std, axis=0)
        self.mstats.register("min", np.min, axis=0)
        self.mstats.register("max", np.max, axis=0)

        # Initialize Logbook
        self.log = tools.Logbook()
        self.log.header = "gen", "evals", "fitness", "size"
        self.log.chapters["fitness"].header = "min", "avg", "max"
        self.log.chapters["size"].header = "min", "avg", "max"

        # Initialize Hall of Fame
        self.hof = tools.HallOfFame(N)

    def spea2(self, seed=None):
        random.seed(seed)

        # Register SPEA2 selection function
        self.tb.register("select", tools.selSPEA2)

        print('Get shortest paths on graph:', end='\n ')
        sps_dict = self.tb.get_shortest_paths()
        print('done')

        paths, path_logs, path_hofs = {}, {}, {}
        for sp_key, sp_dict in sps_dict.items():
            print('Path {0}/{1}'.format(sp_key+1, len(sps_dict)))
            sub_paths, sub_path_logs, sub_path_hofs = {}, {}, {}
            for sub_sp_key, sub_sp_dict in sp_dict.items():
                print('Sub path {0}/{1}'.format(sub_sp_key+1, len(sp_dict)))
                self.tb.register("individual", initialization.init_individual, self.tb, sub_sp_dict)
                self.hof.clear()
                self.log.clear()

                # Step 1: Initialization
                print('Initializing population from shortest path:', end=' ')
                pop = self.tb.population(self.tb.individual,
                                         int(N / len(sub_sp_dict.values()))
                                         )
                archive = []
                curr_gen = 1
                print('done')

                if self.incl_curr:
                    self.evaluator.current_data = ocean_current.CurrentOperator(
                        self.start_date, self.get_n_days(pop))

                # Step 2: Fitness assignment
                inv_inds1, inv_inds2 = pop, []
                fits = self.tb.map(self.tb.evaluate, inv_inds1)
                for ind, fit in zip(inv_inds1, fits):
                    ind.fitness.values = fit

                # Begin the generational process
                while True:
                    # Step 3: Environmental selection
                    archive = self.tb.select(pop + archive, k=Nbar)
                    self.hof.update(archive)

                    # Record statistics
                    record = self.mstats.compile(archive)
                    self.log.record(gen=curr_gen,
                                    evals=(len(inv_inds1) + len(inv_inds2)),
                                    **record)
                    print('\r', self.log.stream)

                    # Step 4: Termination
                    if curr_gen >= GEN:
                        sub_paths[sub_sp_key] = archive
                        sub_path_logs[sub_sp_key] = self.log
                        sub_path_hofs[sub_sp_key] = self.hof
                        self.tb.unregister("individual")
                        break

                    # Step 5: Mating Selection
                    mating_pool = tools.selTournament(archive, k=N, tournsize=2)

                    # Step 6: Variation
                    pop = algorithms.varAnd(mating_pool, self.tb, CXPB, MUTPB)

                    # Step 2: Fitness assignment
                    # Population
                    inv_inds1 = [ind for ind in pop if not ind.fitness.valid]
                    fits = self.tb.map(self.tb.evaluate, inv_inds1)
                    for ind, fit in zip(inv_inds1, fits):
                        ind.fitness.values = fit

                    # Archive
                    inv_inds2 = [ind for ind in archive
                                 if not ind.fitness.valid]
                    fits = self.tb.map(self.tb.evaluate, inv_inds2)
                    for ind, fit in zip(inv_inds2, fits):
                        ind.fitness.values = fit

                    curr_gen += 1

            paths[sp_key] = sub_paths
            path_logs[sp_key] = sub_path_logs
            path_hofs[sp_key] = sub_path_hofs
        return paths, path_logs, path_hofs, sps_dict

    def nsga2(self, seed=None):
        random.seed(seed)

        # Register NSGA2 selection function
        self.tb.register("select", tools.selNSGA2)

        print('Get shortest paths on graph:', end='\n ')
        sps_dict = self.tb.get_shortest_paths()
        print('done')

        paths, path_logs, path_hofs = {}, {}, {}
        for sp_key, sp_dict in sps_dict.items():
            print('Path {0}/{1}'.format(sp_key + 1, len(sps_dict)))
            sub_paths, sub_path_logs, sub_path_hofs = {}, {}, {}
            for sub_sp_key, sub_sp_dict in sp_dict.items():
                print('Sub path {0}/{1}'.format(sub_sp_key+1, len(sp_dict)))
                self.tb.register("individual", initialization.init_individual, self.tb, sub_sp_dict)
                self.hof.clear()
                self.log.clear()

                # Step 1: Initialization
                print('Initializing population from shortest path:', end='\n ')
                pop = self.tb.population(self.tb.individual,
                                         int(N / len(sub_sp_dict.values())))
                offspring = []
                curr_gen = 1
                print('done')

                if self.incl_curr:
                    self.evaluator.current_data = ocean_current.CurrentOperator(
                        self.start_date, self.get_n_days(pop))

                # Step 2: Fitness assignment
                print('Fitness assignment:', end=' ')
                fits = self.tb.map(self.tb.evaluate, pop)
                for ind, fit in zip(pop, fits):
                    ind.fitness.values = fit
                print('assigned')

                # Begin the generational process
                while True:
                    # Step 3: Environmental selection (and update HoF)
                    pop = self.tb.select(pop + offspring, N)
                    self.hof.update(pop)

                    # Record statistics
                    record = self.mstats.compile(pop)
                    self.log.record(gen=curr_gen, evals=len(pop), **record)
                    print('\r', self.log.stream)

                    # Step 4: Termination
                    if curr_gen >= GEN:
                        sub_paths[sub_sp_key] = pop
                        sub_path_logs[sub_sp_key] = self.log
                        sub_path_hofs[sub_sp_key] = self.hof
                        self.tb.unregister("individual")
                        break

                    # Step 5: Variation
                    offspring = algorithms.varAnd(pop, self.tb, CXPB, MUTPB)

                    # Step 2: Fitness assignment
                    inv_inds = [ind for ind in offspring if not ind.fitness.valid]
                    fits = self.tb.map(self.tb.evaluate, inv_inds)
                    for ind, fit in zip(inv_inds, fits):
                        ind.fitness.values = fit

                    curr_gen += 1
            paths[sp_key] = sub_paths
            path_logs[sp_key] = sub_path_logs
            path_hofs[sp_key] = sub_path_hofs

        return paths, path_logs, path_hofs, sps_dict

    def get_n_days(self, pop):
        """
        Get estimate of max travel time of inds in *pop* in whole days
        """
        boat_speed = min(self.vessel.speeds)
        max_travel_time = 0
        for ind in pop:
            travel_time = 0.0
            for e in range(len(ind) - 1):
                p1, p2 = sorted((ind[e][0], ind[e + 1][0]))
                e_dist = self.gc.distance(p1, p2)
                e_travel_time = e_dist / boat_speed
                travel_time += e_travel_time
            if travel_time > max_travel_time:
                max_travel_time = travel_time
        days = int(math.ceil(max_travel_time / 24))
        print('Number of days:', days)
        return days


if __name__ == "__main__":
    start_time = time.time()
    # # Gulf of Guinea, Gulf of Mexico
    # _start, _end = (3.14516, 4.68508), (-94.5968, 26.7012)
    # South Atlantic (Brazil), Caribbean Sea
    # _start, _end = (-23.4166, -7.2574), (-72.3352, 12.8774)
    # Mediterranean Sea, Gulf of Aden
    # _start, _end = (29.188952, 32.842985), (48.1425, 12.5489)
    # Normandy, Canada
    _start, _end = (-5.352121, 48.021295), (-53.306878, 46.423969)
    # Normandy, Canada
    # _start, _end = (-95, 10), (89, 9)
    # Gulf of Bothnia, Gulf of Mexico
    # _start, _end = (20.891193, 58.464147), (-85.063585, 29.175463)
    # North UK, South UK
    # _start, _end = (3.891292, 60.088472), (-7.562237, 47.403357)

    _route_planner = RoutePlanner(_start, _end, eca_f=1.2, incl_curr=True)
    _paths, _path_logs, _hofs, _init_routes = _route_planner.nsga2(seed=1)

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

    # Save Hall-of-Fames
    hofs_fn = '{0:%H_%M_%S}_hofs'.format(timestamp)
    with open('output/hofs/' + hofs_fn, 'wb') as file:
        pickle.dump(_hofs, file)
    print('Saved hofs to "output/hofs/{}"'.format(hofs_fn))

    print("--- %s seconds ---" % (time.time() - start_time))

    sub_path_pop = _paths[0][0]

    front = np.array([_ind.fitness.values for _ind in sub_path_pop])
    plt.scatter(front[:, 0], front[:, 1], c="b")
    plt.axis("tight")
    plt.show()
