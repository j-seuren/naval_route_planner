import functools
import initialization
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
                 vesselName='Fairmaster',
                 ecaFactor=1.2,
                 fuelPrice=0.3,
                 resolution='c',
                 parameters=None,
                 weights=(-10.0, -1.0)):

        defaultParameters = {'splits': 4,          # Threshold for split_polygon
                             'nBar': 50,           # Local archive size (M-PAES, SPEA2)
                             'gen': 110,           # Number of generations
                             'n': 100,             # Population size
                             'cxpb': 0.85,         # Crossover probability (NSGAII, SPEA2)
                             'mutpb': 0.66,        # Mutation probability (NSGAII, SPEA2)
                             'recomb': 5,          # Max recombination trials (M-PAES)
                             'fails': 5,
                             'moves': 10,
                             'widthRatio': 0.5,
                             'radius': 2.1,
                             'shape': 3,
                             'scaleFactor': 0.1,
                             'delFactor': 1.2,
                             'mutationOperators': ['speed', 'insert', 'move', 'delete'],
                             'gauss': False,
                             'stopCriterionRate': 5,
                             }

        # Set parameters
        if parameters:
            self.parameters = {**defaultParameters, **parameters}
        else:
            self.parameters = defaultParameters

        # Create Fitness and Individual types
        creator.create("FitnessMin", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.tb = base.Toolbox()                # Function toolbox
        self.vessel = support.Vessel(vesselName)  # Vessel class instance
        self.ecaFactor = ecaFactor              # Multiplication factor ECA fuel
        self.fuelPrice = fuelPrice              # Fuel price [1000 dollars / mt]
        self.geod = geodesic.Geodesic()         # Geodesic class instance
        self.resolution = resolution
        self.stopIndicator = 1                  # Set stopping indicator

        # Load and pre-process shoreline and ECA polygons
        fp = 'output/split_polygons/res_{0}_threshold_{1}'.format(self.resolution,
                                                                  self.parameters['splits'])
        self.tree = support.populate_rtree(fp, self.parameters)
        fp = 'data/seca_areas'
        self.ecaTree = support.populate_rtree(fp)

        # Initialize "Initializer"
        self.initializer = initialization.Initializer(self.vessel,
                                                      self.resolution,
                                                      self.tree,
                                                      self.ecaTree,
                                                      self.tb,
                                                      self.geod,
                                                      creator.Individual)
        self.initialPaths = {}

        # Initialize "Evaluator" and register it's functions
        self.evaluator = route_evaluation.Evaluator(self.vessel,
                                                    self.tree,
                                                    self.ecaTree,
                                                    ecaFactor,
                                                    self.fuelPrice,
                                                    self.geod)
        self.tb.register("e_feasible", self.evaluator.e_feasible)
        self.tb.register("feasible", self.evaluator.feasible)
        self.tb.register("evaluate", self.evaluator.evaluate)
        self.tb.decorate("evaluate", tools.DeltaPenalty(self.tb.feasible,
                                                        [1e+20, 1e+20]))

        # Initialize "Operator" and register it's functions
        self.operators = operations.Operators(self.tb, self.vessel, self.geod,
                                              self.parameters)
        self.tb.register("mutate", self.operators.mutate)
        self.tb.register("mate", self.operators.cx_one_point)

        self.tb.register("population", initialization.init_repeat_list)

        # Initialize Statistics and logbook
        self.mstats = support.statistics()
        self.log = support.logbook()

        # Initialize ParetoFront and replace its update function
        self.front = tools.ParetoFront()
        self.front.update = functools.partial(support.update, self.front)

        # Initialize algorithm classes
        self.mpaes = self.MPAES(self.tb, self.evaluator, self.mstats, self.log,
                                self.front, self.get_days,
                                self.stopping_criterion, self.parameters)
        self.spea2 = self.SPEA2(self.tb, self.evaluator, self.mstats, self.log,
                                self.front, self.get_days,
                                self.stopping_criterion, self.parameters)
        self.nsgaii = self.NSGAII(self.tb, self.evaluator, self.mstats,
                                  self.front, self.get_days,
                                  self.stopping_criterion, self.parameters)

    class MPAES:
        def __init__(self,
                     tb,
                     evaluator,
                     mstats,
                     log,
                     G,
                     get_days,
                     stopping_criterion,
                     par):

            self.tb = tb
            self.evaluator = evaluator
            self.mstats = mstats
            self.log = log
            self.front = G

            self.get_days = get_days
            self.stopping_criterion = stopping_criterion

            # Parameters
            self.gen = par['gen']
            self.n = par['n']
            self.nBar = par['nBar']
            self.recomb = par['recomb']
            self.nM = int(self.n / 2 / (self.recomb + 1))
            self.fails = par['fails']
            self.moves = par['moves']
            self.nDomPC = self.nDomCP = 0

            self.evals = 0

        def test(self, c, m, H):
            # If the archive is not full
            if len(H) < self.nBar:
                H.append(m)
                support.assign_crowding_dist(H)
                if m.fitness.crowding_dist < c.fitness.crowding_dist:
                    cOut = m
                else:  # Maintain c as current solution
                    cOut = c
            else:
                # If m is in a less crowded region of the archive than x for
                # some member x on the archive
                x = random.choice(H)
                support.assign_crowding_dist(H + [m])
                if m.fitness.crowding_dist < x.fitness.crowding_dist:
                    # Remove a member of the archive from the most crowded region
                    # and add m to the archive
                    removeI = np.argmax([ind_.fitness.crowding_dist for ind_ in H])
                    del H[removeI]
                    H.append(m)
                    support.assign_crowding_dist(H)
                    # If m is in a less crowded region of the archive than c
                    if m.fitness.crowding_dist < c.fitness.crowding_dist:
                        # Accept m as the new current solution
                        cOut = m
                    else:  # Maintain c as the current solution
                        cOut = c
                else:
                    # If m is in a less crowded region of the archive than c
                    if m.fitness.crowding_dist < c.fitness.crowding_dist:
                        # Accept m as the new current solution
                        cOut = m
                    else:  # Maintain c as the current solution
                        cOut = c
            return cOut

        def paes(self, c, H):
            fails = moves = 0
            while fails < self.fails and moves < self.moves:
                # Mutate c to produce m, evaluate m
                m, = self.tb.mutate(self.tb.clone(c))
                fitM = self.tb.evaluate(m)
                m.fitness.values = fitM
                self.evals += 1

                # If c dominates m, discard m
                if c.fitness.dominates(m.fitness):
                    fails += 1
                # Else if m dominates c
                elif m.fitness.dominates(c.fitness):
                    # Ensure c stays in H
                    cCopy = self.tb.clone(c)
                    H.append(cCopy)
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
                    mDominates, mDominated, _, sizeCurr = self.front.update([m])
                    self.nDomPC += mDominates
                    self.nDomCP += mDominated
                moves += 1
            return c

        def compute(self, initPaths, startDate, inclCurr, inclWeather, seed=None):
            random.seed(seed)

            result = {'shortest_paths': initPaths, 'logs': {}, 'fronts': {}}
            for pathKey, path in initPaths.items():
                result['logs'][pathKey], result['fronts'][pathKey] = {}, {}
                print('Path {0}/{1}'.format(pathKey + 1, len(initPaths)))
                for subPathKey, subPath in path.items():
                    print('Sub path {0}/{1}'.format(subPathKey+1, len(path)))
                    self.tb.register("individual", initialization.init_individual, self.tb, subPath)
                    self.front.clear()
                    self.log.clear()

                    # Step 1: Initialization
                    print('Initializing population from shortest path:', end=' ')
                    pop = self.tb.population(self.tb.individual,
                                             int(self.nM / len(subPath.values()))
                                             )

                    currGen = 1
                    print('done')

                    self.evaluator.set_classes(inclCurr, inclWeather, startDate, self.get_days(pop))

                    # Step 2: Fitness and crowding distance assignment
                    invInds = pop
                    fits = self.tb.map(self.tb.evaluate, invInds)
                    self.evals = len(invInds)
                    for ind, fit in zip(invInds, fits):
                        ind.fitness.values = fit

                    self.nDomPC = self.nDomCP = 0

                    # Begin the generational process
                    while True:
                        # Step 3: Update global archive
                        self.nDomPC, self.nDomCP, sizePrev, sizeCurr = self.front.update(pop)

                        # Record statistics
                        record = self.mstats.compile(self.front)
                        self.log.record(gen=currGen, evals=self.evals, **record)
                        print('\r', self.log.stream)

                        self.evals = 0

                        # Step 4: Local search
                        for idx, candidate in enumerate(pop):
                            # Fill local archive (H) with any solutions from pareto front
                            # that do not dominate c
                            localArchive = [hofer for hofer in self.front
                                            if candidate.fitness.dominates(hofer.fitness)]

                            # Copy candidate into H
                            localArchive.append(candidate)

                            # Perform local search using procedure PAES(c, G, H)
                            improvedInd = self.paes(candidate, localArchive)

                            # Replace improved solution back into population
                            pop[idx] = improvedInd

                        # Step 5: Recombination
                        popInter = []  # Initialize intermediate population
                        for g in range(self.nM):
                            child = cDominated = None
                            for r in range(self.recomb):
                                # Randomly choose two parents from P + G
                                mom, dad = tools.selRandom(pop + self.front.items, 2)
                                mom, dad = self.tb.clone(mom), self.tb.clone(dad)

                                # Recombine to form offspring, evaluate
                                child, _ = self.tb.mate(mom, dad)
                                fitChild = self.tb.evaluate(child)
                                child.fitness.values = fitChild
                                self.evals += 1

                                support.assign_crowding_dist(pop + self.front.items + [child])

                                cDominated = True
                                # Check if c is dominated by G
                                for hofer in self.front:
                                    if child.fitness.dominates(hofer.fitness):
                                        cDominated = False

                                cMoreCrowded = True
                                if not cDominated:
                                    # Check if c is in more crowded grid location than both parents
                                    if child.fitness.crowding_dist > mom.fitness.crowding_dist and\
                                            child.fitness.crowding_dist > dad.fitness.crowding_dist:
                                        cMoreCrowded = True

                                # Update pareto front with c as necessary
                                childDominates, childDominated, _, sizeCurr = self.front.update([child])
                                self.nDomPC += childDominates
                                self.nDomCP += childDominated

                                if not (cMoreCrowded or cDominated):
                                    break

                            if cDominated:
                                child = tools.selTournament(self.front.items,
                                                            k=1, tournsize=2)
                            else:
                                child = [child]

                            popInter.extend(self.tb.clone(child))

                        # Step 4: Termination
                        if self.stopping_criterion(self.nDomPC, self.nDomCP, sizePrev, sizeCurr, currGen):
                            result['logs'][pathKey][subPathKey] = self.log[:]
                            result['fronts'][pathKey][subPathKey] = self.front[:]
                            self.tb.unregister("individual")
                            break

                        pop = popInter

                        currGen += 1

            return result

    class NSGAII:
        def __init__(self,
                     tb,
                     evaluator,
                     mstats,
                     front,
                     get_days,
                     stopping_criterion,
                     par):
            self.tb = tb
            self.evaluator = evaluator
            self.mstats = mstats
            self.front = front
            self.nonDominatedInds = []

            self.get_days = get_days
            self.stopping_criterion = stopping_criterion

            # Parameter settings
            self.gen = par['gen']
            self.n = par['n']
            self.cxpb = par['cxpb']
            self.mutpb = par['mutpb']

            # Register NSGA2 selection function
            self.tb.register("select", tools.selNSGA2)

        def compute(self, initPaths, startDate, inclCurr, inclWeather, seed):
            random.seed(seed)

            result = {'shortest_paths': initPaths, 'logs': {}, 'fronts': {}}
            for pathKey, path in initPaths.items():
                result['logs'][pathKey], result['fronts'][pathKey] = {}, {}
                print('Path {0}/{1}'.format(pathKey + 1, len(initPaths)))
                for subPathKey, subPath in path.items():
                    print('Sub path {0}/{1}'.format(subPathKey+1, len(path)))

                    # Reset functions and caches
                    self.tb.register("individual", initialization.init_individual, self.tb, subPath)
                    self.front.clear()
                    log = support.logbook()

                    # Step 1: Initialization
                    print('Initializing population from shortest path:', end='\n ')
                    pop = self.tb.population(self.tb.individual,
                                             int(self.n / len(subPath.values())))
                    offspring = []
                    currGen = 1
                    print('done')

                    self.evaluator.set_classes(inclCurr, inclWeather, startDate, self.get_days(pop))

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
                        sizePrev = len(self.nonDominatedInds)
                        self.nonDominatedInds, nDomPC, nDomCP = self.front.update(pop, self.nonDominatedInds)

                        # Record statistics
                        record = self.mstats.compile(self.front)
                        log.record(gen=currGen, evals=evals, **record)
                        print('\r', log.stream)

                        # Step 4: Termination
                        if self.stopping_criterion(nDomPC, nDomCP, sizePrev, len(self.nonDominatedInds), currGen):
                            result['logs'][pathKey][subPathKey] = deepcopy(log)
                            result['fronts'][pathKey][subPathKey] = deepcopy(self.front)
                            self.tb.unregister("individual")
                            break

                        # Step 5: Variation
                        offspring = algorithms.varAnd(pop, self.tb, self.cxpb, self.mutpb)

                        # Step 2: Fitness assignment
                        invInds = [ind for ind in offspring if not ind.fitness.valid]
                        fits = self.tb.map(self.tb.evaluate, invInds)
                        for ind, fit in zip(invInds, fits):
                            ind.fitness.values = fit

                        evals = len(invInds)

                        currGen += 1

            return result

    class SPEA2:
        def __init__(self,
                     tb,
                     evaluator,
                     mstats,
                     log,
                     front,
                     get_days,
                     stopping_criterion,
                     par):
            self.tb = tb
            self.evaluator = evaluator
            self.mstats = mstats
            self.log = log
            self.front = front

            self.get_days = get_days
            self.stopping_criterion = stopping_criterion

            # Parameter settings
            self.gen = par['gen']
            self.n = par['n']
            self.nBar = par['nBar']
            self.cxpb = par['cxpb']
            self.mutpb = par['mutpb']

            # Register SPEA2 selection function
            self.tb.register("select", tools.selSPEA2)

        def compute(self, initPaths, startDate, inclCurr, inclWeather, seed):
            random.seed(seed)

            result = {'shortest_paths': initPaths, 'logs': {}, 'fronts': {}}
            for pathKey, path in initPaths.items():
                result['logs'][pathKey], result['fronts'][pathKey] = {}, {}
                print('Path {0}/{1}'.format(pathKey + 1, len(initPaths)))
                for subPathKey, subPath in path.items():
                    print('Sub path {0}/{1}'.format(subPathKey+1, len(path)))
                    self.tb.register("individual", initialization.init_individual, self.tb, subPath)
                    self.front.clear()
                    self.log.clear()

                    # Step 1: Initialization
                    print('Initializing population from shortest path:', end=' ')
                    pop = self.tb.population(self.tb.individual,
                                             int(self.n / len(subPath.values()))
                                             )
                    archive = []
                    currGen = 1
                    print('done')

                    self.evaluator.set_classes(inclCurr, inclWeather, startDate, self.get_days(pop))

                    # Step 2: Fitness assignment
                    invInds = pop
                    evals = len(invInds)
                    fits = self.tb.map(self.tb.evaluate, invInds)
                    for ind, fit in zip(invInds, fits):
                        ind.fitness.values = fit

                    # Begin the generational process
                    while True:
                        # Step 3: Environmental selection
                        archive = self.tb.select(pop + archive, k=self.nBar)
                        nDomPC, nDomCP, sizePrev, sizeCurr = self.front.update(archive)

                        # Record statistics
                        record = self.mstats.compile(self.front)
                        self.log.record(gen=currGen, evals=evals, **record)
                        print('\r', self.log.stream)

                        # Step 4: Termination
                        if self.stopping_criterion(nDomPC, nDomCP, sizePrev, sizeCurr, currGen):
                            result['logs'][pathKey][subPathKey] = self.log
                            result['fronts'][pathKey][subPathKey] = self.front[:]
                            self.tb.unregister("individual")
                            break

                        # Step 5: Mating Selection
                        matingPool = tools.selTournament(archive, k=self.n, tournsize=2)

                        # Step 6: Variation
                        pop = algorithms.varAnd(matingPool, self.tb, self.cxpb, self.mutpb)

                        # Step 2: Fitness assignment
                        # Population
                        invInds1 = [ind for ind in pop if not ind.fitness.valid]
                        fits = self.tb.map(self.tb.evaluate, invInds1)
                        for ind, fit in zip(invInds1, fits):
                            ind.fitness.values = fit

                        # Archive
                        invInds2 = [ind for ind in archive if not ind.fitness.valid]
                        fits = self.tb.map(self.tb.evaluate, invInds2)
                        for ind, fit in zip(invInds2, fits):
                            ind.fitness.values = fit

                        evals = len(invInds1) + len(invInds2)

                        currGen += 1

            return result

    def compute(self, startEnds, startDate=None, inclCurr=False, inclWeather=False, algorithm='NSGA2', seed=None):
        self.evaluator.startDate = startDate  # Update start date
        if algorithm == 'MPAES':
            GA = self.mpaes.compute
        elif algorithm == 'SPEA2':
            GA = self.spea2.compute
        else:
            GA = self.nsgaii.compute

        results = []
        for start, end in startEnds:
            # Get initial paths
            initialPaths = self.initialPaths.get((start, end))
            if not initialPaths:
                initialPaths = self.initializer.get_initial_paths(start, end)
                self.initialPaths[(start, end)] = initialPaths

            # Append result of genetic algorithm to list
            results.append(GA(initialPaths, startDate, inclCurr, inclWeather, seed))

        return results

    def update_parameters(self, newParameters):

        self.parameters = {**self.parameters, **newParameters}

        # M-PAES class
        self.mpaes.gen = self.parameters['gen']
        self.mpaes.n = self.parameters['n']
        self.mpaes.nBar = self.parameters['nBar']
        self.mpaes.recomb = self.parameters['recomb']
        self.mpaes.nM = int(self.mpaes.n / 2 / (self.mpaes.recomb + 1))
        self.mpaes.fails = self.parameters['fails']
        self.mpaes.moves = self.parameters['moves']

        # SPEA2 class
        self.spea2.gen = self.parameters['gen']
        self.spea2.n = self.parameters['n']
        self.spea2.nBar = self.parameters['nBar']
        self.spea2.cxpb = self.parameters['cxpb']
        self.spea2.mutpb = self.parameters['mutpb']

        # NSGA-II class
        self.nsgaii.gen = self.parameters['gen']
        self.nsgaii.n = self.parameters['n']
        self.nsgaii.cxpb = self.parameters['cxpb']
        self.nsgaii.mutpb = self.parameters['mutpb']

        # Operator class
        self.operators.radius = self.parameters['radius']
        self.operators.cov = [[self.operators.radius, 0],
                              [0, self.operators.radius]]
        self.operators.widthRatio = self.parameters['widthRatio']
        self.operators.shape = self.parameters['shape']
        self.operators.scaleFactor = self.parameters['scaleFactor']
        self.operators.delFactor = self.parameters['delFactor']
        self.operators.ops = self.parameters['mutationOperators']
        self.operators.gauss = self.parameters['gauss']

        # Re-populate R-Tree structures
        fp = 'output/split_polygons/res_{0}_threshold_{1}'.format(self.resolution, self.parameters['splits'])
        self.tree = support.populate_rtree(fp, self.parameters)
        fp = 'data/seca_areas'
        self.ecaTree = support.populate_rtree(fp)

    def get_days(self, pop):
        """
        Get estimate of max travel time of inds in *pop* in whole days
        """
        boatSpeed = min(self.vessel.speeds)
        maxTravelTime = 0
        for ind in pop:
            travelTime = 0.0
            for e in range(len(ind) - 1):
                p1, p2 = sorted((ind[e][0], ind[e + 1][0]))
                edgeDist = self.geod.distance(p1, p2)
                edgeTravelTime = edgeDist / boatSpeed
                travelTime += edgeTravelTime
            if travelTime > maxTravelTime:
                maxTravelTime = travelTime
        days = int(math.ceil(maxTravelTime / 24))
        print('Number of days:', days)
        return days

    def stopping_criterion(self, nDomPC, nDomCP, sizePrev, sizeCurr, iteration):
        """The MDR indicator provides different types of information.
        MDR =  1:   New population is completely better than its predecessor.
        MDR =  0:   No substantial progress.
        MDR = -1:   New population does not improve any solution of its predecessor.
        """

        # Notation - nDomAB: number of elements of A dominated by any element of B
        if sizePrev > 0 and sizeCurr > 0:
            MDR = nDomPC / sizePrev - nDomCP / sizeCurr
            K = 1 / (1 + self.parameters['stopCriterionRate'])
            self.stopIndicator = self.stopIndicator + K * (MDR - self.stopIndicator)

            if self.stopIndicator < 0:
                print('STOPPING: MGBM criterion')
                return True

        if iteration > self.parameters['gen']:
            print('STOPPING: Max generations')
            return True

        return False


if __name__ == "__main__":
    startTime = time.time()
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
    # _start, _end = (20.89, 58.46), (-85.06, 29.18)
    # North UK, South UK
    # _start, _end = (3.3, 60), (-7.5, 47)
    # Sri Lanka, Yemen
    # _start, _end = (78, 5), (49, 12)
    # # Rotterdam, Tokyo
    # _start, _end = (3.79, 51.98), (139.53, 34.95)
    # Rotterdam, Tokyo
    # _start, _end = (3.79, 51.98), (151, -34)

    _startEnd = [((46.5, -42.5), (-46.3, -49))]

    _parameters = {'gauss': True}

    planner = RoutePlanner(parameters=_parameters)
    _results = planner.compute(_startEnd,
                               startDate=datetime(2020, 8, 16),
                               inclCurr=False,
                               inclWeather=False,
                               seed=1)

    # Save parameters
    timestamp = datetime.now()

    # Save result
    resultFN = '{0:%H_%M_%S}'.format(timestamp)
    with open('output/result/' + resultFN, 'wb') as file:
        pickle.dump(_results, file)
    print('Saved result to: ' + 'output/result/' + resultFN)

    print("--- %s seconds ---" % (time.time() - startTime))
