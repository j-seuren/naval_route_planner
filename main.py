import initialization
import ocean_current
import operations
import math
import numpy as np
import pickle
import random
import support
import time

from copy import deepcopy
from datetime import datetime
from deap import base, creator, tools, algorithms
from evaluation import route_evaluation, geodesic


class RoutePlanner:
    def __init__(self,
                 start=None,
                 end=None,
                 start_date=datetime(2016, 1, 1),
                 vessel_name='Fairmaster',
                 incl_curr=False,
                 eca_f=1.05,
                 vlsfo_price=0.3,
                 par=None):

        if par is None:
            par = {'res': 'c',
                   'spl_th': 4,
                   'n_bar': 50,
                   'gen': 200,
                   'n': 250,
                   'cxpb': 0.9,
                   'mutpb': 0.9,
                   'recomb': 5,
                   'l_fails': 5,
                   'l_moves': 10,
                   'width_ratio': 0.5,
                   'radius': 1,
                   'shape': 3,
                   'scale_factor': 0.1,
                   'del_factor': 2
                   }

        # Create Fitness and Individual types
        creator.create("FitnessMin", base.Fitness, weights=(-10.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.creator = creator

        self.tb = base.Toolbox()                # Function toolbox
        self.start = start                      # Start point
        self.end = end                          # End point
        self.start_date = start_date            # Start date of voyage
        self.vessel = support.Vessel(vessel_name)  # Vessel class instance
        self.incl_curr = incl_curr              # Boolean for including currents
        self.eca_f = eca_f                      # Multiplication factor ECA fuel
        self.vlsfo_price = vlsfo_price          # VLSFO price [1000 dollars / mt]
        self.res = par['res']                   # Resolution of shorelines
        self.spl_th = par['spl_th']             # Threshold for split_polygon
        self.geod = geodesic.Geodesic()         # Geodesic class instance

        # Parameter settings
        self.n_bar = par['n_bar']      # Local archive size (M-PAES, SPEA2)
        self.gen = par['gen']         # Number of generations
        self.n = par['n']             # Population size
        self.cxpb = par['cxpb']       # Crossover probability (NSGAII, SPEA2)
        self.mutpb = par['mutpb']     # Mutation probability (NSGAII, SPEA2)
        self.recomb = par['recomb']   # Max recombination trials (M-PAES)

        # Load and pre-process shoreline and ECA polygons
        fp = 'output/split_polygons/res_{0}_threshold_{1}'.format(par['res'],
                                                                  par['spl_th'])
        self.tree = support.populate_rtree(fp, par['res'], par['spl_th'])
        fp = 'data/seca_areas'
        self.eca_tree = support.populate_rtree(fp)

        # Initialize "Evaluator" and register it's functions
        self.evaluator = route_evaluation.Evaluator(self.vessel,
                                                    self.tree,
                                                    self.eca_tree,
                                                    eca_f,
                                                    self.vlsfo_price,
                                                    start_date,
                                                    self.geod,
                                                    incl_curr)
        self.tb.register("e_feasible", self.evaluator.e_feasible)
        self.tb.register("feasible", self.evaluator.feasible)
        self.tb.register("evaluate", self.evaluator.evaluate)
        self.tb.decorate("evaluate", tools.DeltaPenalty(self.tb.feasible,
                                                        [1e+20, 1e+20]))

        # Initialize "Initializer" and register it's functions
        if start and end:
            self.initializer = initialization.Initializer(start, end,
                                                          self.vessel,
                                                          par['res'],
                                                          self.tree,
                                                          self.eca_tree,
                                                          self.tb,
                                                          self.geod)
            self.tb.register("get_shortest_paths",
                             self.initializer.get_shortest_paths,
                             creator.Individual)
        else:
            print('No start and endpoint given')

        # Initialize "Operator" and register it's functions
        self.operators = operations.Operators(self.tb, self.vessel, self.geod,
                                              par)
        self.tb.register("mutate", self.operators.mutate)
        self.tb.register("mate", self.operators.cx_one_point)

        self.tb.register("population", initialization.init_repeat_list)

        # Initialize Statistics and logbook
        self.mstats = support.statistics()
        self.log = support.logbook()

        # Initialize ParetoFront
        self.front = tools.ParetoFront()

        # Initialize algorithm classes
        self.mpaes = self.MPAES(self.tb, self.evaluator, self.mstats, self.log,
                                self.front, self.incl_curr, self.start_date,
                                self.get_n_days, par)
        self.spea2 = self.SPEA2(self.tb, self.evaluator, self.mstats, self.log,
                                self.front, self.incl_curr, self.start_date,
                                self.get_n_days, par)
        self.nsgaii = self.NSGAII(self.tb, self.evaluator, self.mstats,
                                  self.front, self.incl_curr, self.start_date,
                                  self.get_n_days, par)

    class MPAES:
        def __init__(self,
                     tb,
                     evaluator,
                     mstats,
                     log,
                     G,
                     incl_curr,
                     start_date,
                     get_n_days,
                     params):

            self.tb = tb
            self.evaluator = evaluator
            self.mstats = mstats
            self.log = log
            self.G = G

            self.incl_curr = incl_curr
            self.start_date = start_date
            self.get_n_days = get_n_days

            # Parameters
            self.gen = params['gen']
            self.n = params['n']
            self.nbar = params['n_bar']
            self.recomb = params['recomb']
            self.n_m = int(self.n / 2 / (self.recomb + 1))
            self.l_fails = params['l_fails']
            self.l_moves = params['l_moves']

            self.evals = 0

        def test(self, c, m, H):
            # If the archive is not full
            if len(H) < self.nbar:
                H.append(m)
                support.assign_crowding_dist(H)
                if m.fitness.crowding_dist < c.fitness.crowding_dist:
                    c_out = m
                else:  # Maintain c as current solution
                    c_out = c
            else:
                # If m is in a less crowded region of the archive than x for
                # some member x on the archive
                x = random.choice(H)
                support.assign_crowding_dist(H + [m])
                if m.fitness.crowding_dist < x.fitness.crowding_dist:
                    # Remove a member of the archive from the most crowded region
                    # and add m to the archive
                    remove_i = np.argmax([ind_.fitness.crowding_dist for ind_ in H])
                    del H[remove_i]
                    H.append(m)
                    support.assign_crowding_dist(H)
                    # If m is in a less crowded region of the archive than c
                    if m.fitness.crowding_dist < c.fitness.crowding_dist:
                        # Accept m as the new current solution
                        c_out = m
                    else:  # Maintain c as the current solution
                        c_out = c
                else:
                    # If m is in a less crowded region of the archive than c
                    if m.fitness.crowding_dist < c.fitness.crowding_dist:
                        # Accept m as the new current solution
                        c_out = m
                    else:  # Maintain c as the current solution
                        c_out = c
            return c_out

        def paes(self, c, H):
            fails = moves = 0
            while fails < self.l_fails and moves < self.l_moves:
                # Mutate c to produce m, evaluate m
                m, = self.tb.mutate(self.tb.clone(c))
                fit_m = self.tb.evaluate(m)
                m.fitness.values = fit_m
                self.evals += 1

                # If c dominates m, discard m
                if c.fitness.dominates(m.fitness):
                    fails += 1
                # Else if m dominates c
                elif m.fitness.dominates(c.fitness):
                    # Ensure c stays in H
                    c_copy = self.tb.clone(c)
                    H.append(c_copy)
                    # Replace c with m, add m to H
                    c = m
                    H.append(m)
                    fails = 0
                else:
                    # If m is dominated by any member of H, discard m
                    dominated = False
                    for ind_ in H:
                        if ind_.fitness.dominates(m.fitness):
                            dominated = True
                            break
                    # Else apply test(c,m, H) to determine which becomes the new
                    # current solution and whether to add m to the archive
                    if not dominated:
                        c = self.test(c, m, H)
                    # Archive m in G as necessary
                    self.G.update([m])
                moves += 1
            return c

        def compute(self, seed=None):
            random.seed(seed)

            print('Get shortest paths on graph:', end='\n ')
            sps_dict = self.tb.get_shortest_paths()
            print('done')

            result = {'shortest_paths': sps_dict, 'logs': {}, 'fronts': {}}
            for sp_key, sp_dict in sps_dict.items():
                result['logs'][sp_key], result['fronts'][sp_key] = {}, {}
                print('Path {0}/{1}'.format(sp_key+1, len(sps_dict)))
                for sub_sp_key, sub_sp_dict in sp_dict.items():
                    print('Sub path {0}/{1}'.format(sub_sp_key+1, len(sp_dict)))
                    self.tb.register("individual", initialization.init_individual, self.tb, sub_sp_dict)
                    self.G.clear()
                    self.log.clear()

                    # Step 1: Initialization
                    print('Initializing population from shortest path:', end=' ')
                    pop = self.tb.population(self.tb.individual,
                                             int(self.n_m / len(sub_sp_dict.values()))
                                             )

                    curr_gen = 1
                    print('done')

                    if self.incl_curr:
                        self.evaluator.current_data = ocean_current.CurrentOperator(
                            self.start_date, self.get_n_days(pop))

                    # Step 2: Fitness and crowding distance assignment
                    inv_inds = pop
                    fits = self.tb.map(self.tb.evaluate, inv_inds)
                    self.evals = len(inv_inds)
                    for ind, fit in zip(inv_inds, fits):
                        ind.fitness.values = fit

                    # Begin the generational process
                    while True:
                        # Step 3: Update global archive
                        self.G.update(pop)

                        # Record statistics
                        record = self.mstats.compile(self.G)
                        self.log.record(gen=curr_gen, evals=self.evals, **record)
                        print('\r', self.log.stream)

                        self.evals = 0

                        # Step 4: Local search
                        for idx, candidate in enumerate(pop):
                            # Fill local archive (H) with any solutions from pareto front
                            # that do not dominate c
                            local_archive = [hofer for hofer in self.G
                                             if candidate.fitness.dominates(hofer.fitness)]

                            # Copy candidate into H
                            local_archive.append(candidate)

                            # Perform local search using procedure PAES(c, G, H)
                            impr_ind = self.paes(candidate, local_archive)

                            # Replace improved solution back into population
                            pop[idx] = impr_ind

                        # Step 5: Recombination
                        pop_i = []  # Initialize intermediate population
                        for g in range(self.n_m):
                            child = c_dominated = None
                            for r in range(self.recomb):
                                # Randomly choose two parents from P + G
                                mom, dad = tools.selRandom(pop + self.G.items, 2)
                                mom, dad = self.tb.clone(mom), self.tb.clone(dad)

                                # Recombine to form offspring, evaluate
                                child, _ = self.tb.mate(mom, dad)
                                fit_c = self.tb.evaluate(child)
                                child.fitness.values = fit_c
                                self.evals += 1

                                support.assign_crowding_dist(pop + self.G.items + [child])

                                c_dominated = True
                                # Check if c is dominated by G
                                for hofer in self.G:
                                    if child.fitness.dominates(hofer.fitness):
                                        c_dominated = False

                                c_more_crowded = True
                                if not c_dominated:
                                    # Check if c is in more crowded grid location than both parents
                                    if child.fitness.crowding_dist > mom.fitness.crowding_dist and\
                                            child.fitness.crowding_dist > dad.fitness.crowding_dist:
                                        c_more_crowded = True

                                # Update G with c as necessary
                                self.G.update([child])

                                if not (c_more_crowded or c_dominated):
                                    break

                            if c_dominated:
                                child = tools.selTournament(self.G.items,
                                                            k=1, tournsize=2)
                            else:
                                child = [child]

                            pop_i.extend(self.tb.clone(child))

                        # Step 4: Termination
                        if curr_gen >= self.gen:
                            result['logs'][sp_key][sub_sp_key] = self.log[:]
                            result['fronts'][sp_key][sub_sp_key] = self.G[:]
                            self.tb.unregister("individual")
                            break

                        pop = pop_i

                        curr_gen += 1

            return result

    class NSGAII:
        def __init__(self,
                     tb,
                     evaluator,
                     mstats,
                     front,
                     incl_curr,
                     start_date,
                     get_n_days,
                     params):
            self.tb = tb
            self.evaluator = evaluator
            self.mstats = mstats
            self.front = front

            self.incl_curr = incl_curr
            self.start_date = start_date
            self.get_n_days = get_n_days

            # Parameter settings
            self.gen = params['gen']
            self.n = params['n']
            self.cxpb = params['cxpb']
            self.mutpb = params['mutpb']

            # Register NSGA2 selection function
            self.tb.register("select", tools.selNSGA2)

        def compute(self, seed=None):
            random.seed(seed)

            print('Get shortest paths on graph:', end='\n ')
            sps_dict = self.tb.get_shortest_paths()
            print('done')

            result = {'shortest_paths': sps_dict, 'logs': {}, 'fronts': {}}
            for sp_key, sp_dict in sps_dict.items():
                result['logs'][sp_key], result['fronts'][sp_key] = {}, {}
                print('Path {0}/{1}'.format(sp_key + 1, len(sps_dict)))
                for sub_sp_key, sub_sp_dict in sp_dict.items():
                    print('Sub path {0}/{1}'.format(sub_sp_key+1, len(sp_dict)))

                    # Reset functions and caches
                    self.tb.register("individual", initialization.init_individual, self.tb, sub_sp_dict)
                    self.front.clear()
                    log = support.logbook()

                    # Step 1: Initialization
                    print('Initializing population from shortest path:', end='\n ')
                    pop = self.tb.population(self.tb.individual,
                                             int(self.n / len(sub_sp_dict.values())))
                    offspring = []
                    curr_gen = 1
                    print('done')

                    if self.incl_curr:
                        self.evaluator.current_data = ocean_current.CurrentOperator(
                            self.start_date, self.get_n_days(pop))

                    # Step 2: Fitness assignment
                    print('Fitness assignment:', end=' ')
                    evals = len(pop)
                    fits = self.tb.map(self.tb.evaluate, pop)
                    for ind, fit in zip(pop, fits):
                        ind.fitness.values = fit
                    print('assigned')

                    # Begin the generational process
                    while True:
                        # Step 3: Environmental selection (and update HoF)
                        pop = self.tb.select(pop + offspring, self.n)
                        self.front.update(pop)

                        # Record statistics
                        record = self.mstats.compile(self.front)
                        log.record(gen=curr_gen, evals=evals, **record)
                        print('\r', log.stream)

                        # Step 4: Termination
                        if curr_gen >= self.gen:
                            result['logs'][sp_key][sub_sp_key] = deepcopy(log)
                            result['fronts'][sp_key][sub_sp_key] = deepcopy(self.front)
                            self.tb.unregister("individual")
                            break

                        # Step 5: Variation
                        offspring = algorithms.varAnd(pop, self.tb, self.cxpb, self.mutpb)

                        # Step 2: Fitness assignment
                        inv_inds = [ind for ind in offspring if not ind.fitness.valid]
                        fits = self.tb.map(self.tb.evaluate, inv_inds)
                        for ind, fit in zip(inv_inds, fits):
                            ind.fitness.values = fit

                        evals = len(inv_inds)

                        curr_gen += 1

            return result

    class SPEA2:
        def __init__(self,
                     tb,
                     evaluator,
                     mstats,
                     log,
                     front,
                     incl_curr,
                     start_date,
                     get_n_days,
                     params):
            self.tb = tb
            self.evaluator = evaluator
            self.mstats = mstats
            self.log = log
            self.front = front

            self.incl_curr = incl_curr
            self.start_date = start_date
            self.get_n_days = get_n_days

            # Parameter settings
            self.gen = params['gen']
            self.n = params['n']
            self.n_bar = params['n_bar']
            self.cxpb = params['cxpb']
            self.mutpb = params['mutpb']

            # Register SPEA2 selection function
            self.tb.register("select", tools.selSPEA2)

        def compute(self, seed=None):
            random.seed(seed)

            print('Get shortest paths on graph:', end='\n ')
            sps_dict = self.tb.get_shortest_paths()
            print('done')

            result = {'shortest_paths': sps_dict, 'logs': {}, 'fronts': {}}
            for sp_key, sp_dict in sps_dict.items():
                result['logs'][sp_key], result['fronts'][sp_key] = {}, {}
                print('Path {0}/{1}'.format(sp_key+1, len(sps_dict)))
                for sub_sp_key, sub_sp_dict in sp_dict.items():
                    print('Sub path {0}/{1}'.format(sub_sp_key+1, len(sp_dict)))
                    self.tb.register("individual", initialization.init_individual, self.tb, sub_sp_dict)
                    self.front.clear()
                    self.log.clear()

                    # Step 1: Initialization
                    print('Initializing population from shortest path:', end=' ')
                    pop = self.tb.population(self.tb.individual,
                                             int(self.n / len(sub_sp_dict.values()))
                                             )
                    archive = []
                    curr_gen = 1
                    print('done')

                    if self.incl_curr:
                        self.evaluator.current_data = ocean_current.CurrentOperator(
                            self.start_date, self.get_n_days(pop))

                    # Step 2: Fitness assignment
                    inv_inds = pop
                    evals = len(inv_inds)
                    fits = self.tb.map(self.tb.evaluate, inv_inds)
                    for ind, fit in zip(inv_inds, fits):
                        ind.fitness.values = fit

                    # Begin the generational process
                    while True:
                        # Step 3: Environmental selection
                        archive = self.tb.select(pop + archive, k=self.n_bar)
                        self.front.update(archive)

                        # Record statistics
                        record = self.mstats.compile(self.front)
                        self.log.record(gen=curr_gen, evals=evals, **record)
                        print('\r', self.log.stream)

                        # Step 4: Termination
                        if curr_gen >= self.gen:
                            result['logs'][sp_key][sub_sp_key] = self.log
                            result['fronts'][sp_key][sub_sp_key] = self.front[:]
                            self.tb.unregister("individual")
                            break

                        # Step 5: Mating Selection
                        mating_pool = tools.selTournament(archive, k=self.n, tournsize=2)

                        # Step 6: Variation
                        pop = algorithms.varAnd(mating_pool, self.tb, self.cxpb, self.mutpb)

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

                        evals = len(inv_inds1) + len(inv_inds2)

                        curr_gen += 1

            return result

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
                e_dist = self.geod.distance(p1, p2)
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
    # _start, _end = (-5.352121, 48.021295), (-53.306878, 46.423969)
    # Normandy, Canada
    # _start, _end = (-95, 10), (89, 9)
    # Gulf of Bothnia, Gulf of Mexico
    # _start, _end = (20.891193, 58.464147), (-85.063585, 29.175463)
    # North UK, South UK
    _start, _end = (3.3, 60), (-7.5, 47)
    # Sri Lanka, Yemen
    # _start, _end = (78, 5), (49, 12)
    # # Rotterdam, Tokyo
    # _start, _end = (3.79, 51.98), (139.53, 34.95)
    # # Rotterdam, Tokyo
    # _start, _end = (3.79, 51.98), (103.60, 1.08)

    planner = RoutePlanner(_start, _end, incl_curr=False)
    _result = planner.nsgaii.compute(seed=1)

    # Save parameters
    timestamp = datetime.now()

    # Save result
    result_fn = '{0:%H_%M_%S}'.format(timestamp)
    with open('output/result/' + result_fn, 'wb') as file:
        pickle.dump(_result, file)
    print('Saved result to: ' + 'output/result/' + result_fn)

    print("--- %s seconds ---" % (time.time() - start_time))
