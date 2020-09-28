import evaluation
import initialization
import math
import numpy as np
import os
import indicators
import pickle
import random
import support
import uuid

from copy import deepcopy
from data_config.navigable_area import NavigableAreaGenerator
from deap import base, creator, tools, algorithms
from analysis import geodesic
from operations import Operators
from pathlib import Path
from shapely.geometry import Point, Polygon


_criteria = {'minimalTime': True, 'minimalCost': True}
timeWeight, costWeight = -5, -1

if _criteria['minimalTime'] and _criteria['minimalCost']:
    _criteria = {'minimalTime': timeWeight, 'minimalCost': timeWeight}
elif _criteria['minimalCost']:
    _criteria = {'minimalCost': -1}
else:
    _criteria = {'minimalTime': -1}

DIR = Path('D:/')
creator.create("FitnessMin", base.Fitness, weights=tuple(_criteria.values(),))
creator.create("Individual", list, fitness=creator.FitnessMin)
_tb = base.Toolbox()


class RoutePlanner:
    def __init__(self,
                 vesselName='Fairmaster_2',
                 shipLoading='normal',
                 ecaFactor=1.2,
                 fuelPrice=300,  # Fuel price per metric tonne
                 bathymetry=True,
                 inputParameters=None,
                 tb=None,
                 criteria=None):
        if criteria is None:
            criteria = _criteria

        # Set parameters
        defaultParameters = {
                             # Navigation area parameters
                             'avoidAntarctic': True,
                             'avoidArctic': True,
                             'res': 'l',           # Resolution of shorelines
                             'penaltyValue': {'time': criteria['minimalTime'],
                                              'cost': criteria['minimalCost']},
                             'graphDens': 4,       # Recursion level graph
                             'graphVarDens': 6,    # Variable recursion level graph
                             'splits': 3,          # Threshold for split_polygon (val 3 yields best performance)

                             # MOEA parameters
                             'n': 322,             # Population size
                             'nBar': 50,           # Local archive size (M-PAES, SPEA2)
                             'cxpb': 0.75,         # Crossover probability (NSGAII, SPEA2)
                             'mutpb': 0.51,        # Mutation probability (NSGAII, SPEA2)
                             'nMutations': 5,      # Max. number of mutations per selected individual
                             'recomb': 5,          # Max recombination trials (M-PAES)
                             'fails': 5,           # Max fails (M-PAES)
                             'moves': 10,          # Max moves (M-PAES)

                             # Stopping parameters
                             'gen': 247,           # Minimal number of generations
                             'maxGDs': 30,         # Max length of generational distance list
                             'minVar': 4.8e-5,       # Minimal variance of generational distance list

                             # Mutation parameters
                             'mutationOperators': ['speed', 'insert', 'move', 'delete'],  # Operators to be included
                             'widthRatio': 1.5,  # 7.5e-4 obtained from hyp param tuning
                             'radius': 0.4,       # 0.39 obtained from hyp param tuning
                             'shape': 3,           # Shape parameter for Gamma distribution
                             'scaleFactor': 0.1,   # Scale factor for Gamma and Exponential distribution
                             'delFactor': 1,       # Factor of deletions
                             'gauss': False,       # Use Gaussian mutation for insert and move operators

                             # Evaluation parameters
                             'segLengthF': 15,     # Length of linear approx. of great circle track for feasibility
                             'segLengthC': 15      # same for ocean currents and wind along route
                             }
        self.p = {**defaultParameters, **inputParameters} if inputParameters else defaultParameters
        self.tb = _tb if tb is None else tb
        self.criteria = criteria
        self.procResultsFP = None
        self.vessel = evaluation.Vessel(fuelPrice, vesselName, shipLoading, DIR=DIR)  # Vessel class instance
        self.fuelPrice = fuelPrice
        self.ecaFactor = ecaFactor              # Multiplication factor ECA fuel
        self.geod = geodesic.Geodesic()         # Geodesic class instance

        # Load and pre-process shoreline, ECA, and Bathymetry geometries
        navAreaGenerator = NavigableAreaGenerator(self.p, DIR=DIR)
        landTree = navAreaGenerator.get_shoreline_rtree()
        ecaTree = navAreaGenerator.get_eca_rtree()
        bathTree = navAreaGenerator.get_bathymetry_rtree() if bathymetry else None

        # Initialize "Evaluator" and register it's functions
        self.evaluator = evaluation.Evaluator(self.vessel,
                                              landTree,
                                              ecaTree,
                                              bathTree,
                                              ecaFactor,
                                              self.geod,
                                              criteria,
                                              self.p,
                                              DIR=DIR)

        # Initialize "Initializer"
        self.initializer = initialization.Initializer(self.evaluator,
                                                      self.vessel,
                                                      landTree,
                                                      ecaTree,
                                                      self.geod,
                                                      self.p,
                                                      creator.Individual,
                                                      DIR)

        # Load previously calculated initial paths
        self.initPathsDir = DIR / 'output/initialRoutes/RES_{}_D{}_VD_{}'.format(self.p['res'],
                                                                                 self.p['graphDens'],
                                                                                 self.p['graphVarDens'])
        if not os.path.exists(self.initPathsDir):
            os.mkdir(self.initPathsDir)
        self.initialPaths = []
        for fp in os.listdir(self.initPathsDir):
            with open(self.initPathsDir / fp, 'rb') as file:
                self.initialPaths.append(pickle.load(file))

        # Initialize "Operator" and register it's functions
        self.operators = Operators(self.evaluator.e_feasible, self.vessel, self.geod, self.p)
        self.tb.register("mutate", self.operators.mutate)
        self.tb.register("mate", self.operators.cx_one_point)
        self.tb.register("population", initialization.init_repeat_list)

        # Initialize algorithm classes
        self.mpaes = self.MPAES(self.tb, self.evaluator, self.get_days, self.p)
        self.spea2 = self.SPEA2(self.tb, self.evaluator, self.get_days, self.p)
        self.nsgaii = self.NSGAII(self.tb, self.evaluator, self.get_days, self.p)

    class MPAES:
        def __init__(self, tb, evaluator, get_days, par):
            self.tb = tb
            self.evaluator = evaluator
            self.get_days = get_days

            # Parameters
            self.nBar = par['nBar']
            self.recomb = par['recomb']
            self.n = par['n']
            self.nM = int(self.n / 2 / (self.recomb + 1))
            self.fails = par['fails']
            self.moves = par['moves']
            self.gen = par['gen']
            self.minVar = par['minVar']
            self.maxStops = par['maxGDs']
            self.gds = []
            self.nStops = 0
            self.evals = 0

            self.mstats = support.statistics()
            self.front = tools.ParetoFront()  # Initialize ParetoFront class

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
                fitM = self.evaluator.evaluate(m)
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
                    # Update global Pareto front with m, as necessary
                    self.front.update([m])
                moves += 1
            return c

        def termination(self, prevFront, t):
            gd = indicators.generational_distance(prevFront, self.front)
            self.gds.append(gd)
            if len(self.gds) > self.maxStops:
                self.gds.pop(0)
            if t >= self.gen:
                if np.var(self.gds) < self.minVar:
                    print('STOPPING: Generational distance')
                    return True
            return False

        def optimize(self, startEnd, initPaths, startDate, inclCurr, inclWeather, seed=None):
            random.seed(seed)

            result = {'startEnd': startEnd, 'initialRoutes': initPaths, 'logs': {}, 'fronts': {}}
            for pathKey, path in initPaths.items():
                result['logs'][pathKey], result['fronts'][pathKey] = {}, {}
                print('Path {0}/{1}'.format(pathKey + 1, len(initPaths)))
                for subPathKey, subPath in path.items():
                    print('Sub path {0}/{1}'.format(subPathKey+1, len(path)))
                    self.tb.register("individual", initialization.init_individual, self.tb,  subPath)
                    self.front.clear()
                    log = support.logbook()

                    # Step 1: Initialization
                    print('Initializing population from shortest path:', end=' ')
                    pop = self.tb.population(self.tb.individual,
                                             int(self.nM / len(subPath.values()))
                                             )

                    currGen = self.nStops = 0
                    print('done')

                    self.evaluator.set_classes(inclCurr, inclWeather, startDate, self.get_days(pop))

                    # Step 2: Fitness and crowding distance assignment
                    invInds = pop
                    fits = self.tb.map(self.evaluator.evaluate, invInds)
                    self.evals = len(invInds)
                    for ind, fit in zip(invInds, fits):
                        ind.fitness.values = fit

                    # Begin the generational process
                    while True:
                        # Step 3: Update global Pareto front
                        prevFront = deepcopy(self.front)
                        self.front.update(pop)

                        # Record statistics
                        record = self.mstats.compile(self.front)
                        log.record(gen=currGen, evals=self.evals, **record)
                        print('\r', log.stream)

                        self.evals = 0

                        # Step 4: Local search
                        for idx, candidate in enumerate(pop):
                            # Fill local Pareto front (H) with any solutions from pareto front
                            # that do not dominate c
                            localFront = [hofer for hofer in self.front
                                          if candidate.fitness.dominates(hofer.fitness)]

                            # Copy candidate into H
                            localFront.append(candidate)

                            # Perform local search using procedure PAES(c, G, H)
                            improvedInd = self.paes(candidate, localFront)

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
                                fitChild = self.evaluator.evaluate(child)
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
                                self.front.update([child])

                                if not (cMoreCrowded or cDominated):
                                    break

                            if cDominated:
                                child = tools.selTournament(self.front.items,
                                                            k=1, tournsize=2)
                            else:
                                child = [child]

                            popInter.extend(self.tb.clone(child))

                        # Step 4: Termination
                        if self.termination(prevFront, currGen):
                            result['logs'][pathKey][subPathKey] = log[:]
                            result['fronts'][pathKey][subPathKey] = self.front[:]
                            self.tb.unregister("individual")
                            break

                        pop = popInter

                        currGen += 1

            return result

    class NSGAII:
        def __init__(self, tb, evaluator, get_days, par):
            self.tb = tb
            self.evaluator = evaluator
            self.get_days = get_days

            # Parameters
            self.gen = par['gen']
            self.n = par['n']
            self.cxpb = par['cxpb']
            self.mutpb = par['mutpb']
            self.minVar = par['minVar']
            self.maxStops = par['maxGDs']
            self.gds = []

            # Initialize mutation selection weights, scores, and probabilities
            # self.mutW = {op: 0 for op in par['mutationOperators']}
            # self.mutP = {op: 0 for op in par['mutationOperators']}
            # self.mutS = {op: 0 for op in par['mutationOperators']}

            self.mstats = support.statistics()
            self.front = tools.ParetoFront()  # Initialize ParetoFront class
            self.tb.register("select", tools.selNSGA2)

        def termination(self, prevFront, t):
            gd = indicators.generational_distance(prevFront, self.front)
            self.gds.append(gd)
            if len(self.gds) > self.maxStops:
                self.gds.pop(0)
            if t >= self.gen:
                if np.var(self.gds) < self.minVar:
                    print('STOPPING: Generational distance')
                    return True
            # if t >= self.gen:
            #     print('STOPPING: Max generations')
            #     return True
            return False

        def optimize(self, startEnd, initRoutes, startDate, current, weather, seed):
            random.seed(seed)

            result = {'startEnd': startEnd, 'initialRoutes': initRoutes, 'indicators': [None] * len(initRoutes),
                      'logs': [None] * len(initRoutes), 'fronts': [None] * len(initRoutes)}
            for routeIdx, route in enumerate(initRoutes):
                print('Computing route {0}/{1}'.format(routeIdx + 1, len(initRoutes)))
                for subIdx, subRoute in enumerate(route['route']):
                    result['logs'][routeIdx], result['fronts'][routeIdx], result['indicators'][routeIdx] = [], [], {}
                    print('Computing sub route {0}/{1}'.format(subIdx+1, len(route['route'])))

                    # Reset functions and caches
                    self.tb.register("individual", initialization.init_individual, self.tb, subRoute)
                    self.front.clear()
                    log = support.logbook()

                    # Step 1: Initialization
                    print('Initializing population from shortest path:', end='\n ')
                    pop = self.tb.population(self.tb.individual, int(self.n / len(subRoute.values())))
                    offspring, currGen = [], 0
                    print('done')

                    self.evaluator.set_classes(current, weather, startDate, self.get_days(pop))

                    # Step 2: Fitness assignment
                    print('Fitness assignment:', end=' ')
                    evals = len(pop)
                    fits = self.tb.map(self.evaluator.evaluate, pop)
                    for ind, fit in zip(pop, fits):
                        ind.fitness.values = fit
                    print('assigned')

                    # Begin the generational process
                    while True:
                        # Step 3: Environmental selection (and update HoF)
                        pop = self.tb.select(pop + offspring, self.n)
                        prevFront = deepcopy(self.front)
                        self.front.update(pop)

                        # Record statistics
                        record = self.mstats.compile(self.front)
                        log.record(gen=currGen, evals=evals, **record)
                        print('\r', log.stream)

                        # Step 4: Termination
                        if self.termination(prevFront, currGen):
                            hypervolume = indicators.hypervolume(self.front)
                            print('hypervolume', hypervolume)
                            result['indicators'][routeIdx]['hypervolume'] = hypervolume
                            result['logs'][routeIdx].append(deepcopy(log))
                            result['fronts'][routeIdx].append(deepcopy(self.front))
                            self.tb.unregister("individual")
                            break

                        # Step 5: Variation
                        offspring = algorithms.varAnd(pop, self.tb, self.cxpb, self.mutpb)

                        # Step 2: Fitness assignment
                        invInds = [ind for ind in offspring if not ind.fitness.valid]
                        fits = self.tb.map(self.evaluator.evaluate, invInds)
                        for ind, fit in zip(invInds, fits):
                            ind.fitness.values = fit

                        evals = len(invInds)

                        currGen += 1

            return result

    class SPEA2:
        def __init__(self, tb, evaluator, get_days, par):
            self.tb = tb
            self.evaluator = evaluator
            self.get_days = get_days

            # Parameters
            self.gen = par['gen']
            self.n = par['n']
            self.nBar = par['nBar']
            self.cxpb = par['cxpb']
            self.mutpb = par['mutpb']
            self.minVar = par['minVar']
            self.maxStops = par['maxGDs']
            self.gds = []

            self.mstats = support.statistics()
            self.front = tools.ParetoFront()  # Initialize ParetoFront class
            self.tb.register("select", tools.selSPEA2)

        def termination(self, prevFront, t):
            gd = indicators.generational_distance(prevFront, self.front)
            self.gds.append(gd)
            if len(self.gds) > self.maxStops:
                self.gds.pop(0)
            if t >= self.gen:
                if np.var(self.gds) < self.minVar:
                    print('STOPPING: Generational distance')
                    return True
            return False

        def optimize(self, startEnd, initPaths, startDate, inclCurr, inclWeather, seed):
            random.seed(seed)

            result = {'startEnd': startEnd, 'initialRoutes': initPaths, 'logs': {}, 'fronts': {}}
            for pathKey, path in initPaths.items():
                result['logs'][pathKey], result['fronts'][pathKey] = {}, {}
                print('Path {0}/{1}'.format(pathKey + 1, len(initPaths)))
                for subPathKey, subPath in path.items():
                    print('Sub path {0}/{1}'.format(subPathKey+1, len(path)))
                    self.tb.register("individual", initialization.init_individual, self.tb, subPath)
                    self.front.clear()
                    log = support.logbook()

                    # Step 1: Initialization
                    print('Initializing population from shortest path:', end=' ')
                    pop = self.tb.population(self.tb.individual,
                                             int(self.n / len(subPath.values()))
                                             )
                    archive = []
                    currGen = 0
                    print('done')

                    self.evaluator.set_classes(inclCurr, inclWeather, startDate, self.get_days(pop))

                    # Step 2: Fitness assignment
                    invInds = pop
                    evals = len(invInds)
                    fits = self.tb.map(self.evaluator.evaluate, invInds)
                    for ind, fit in zip(invInds, fits):
                        ind.fitness.values = fit

                    # Begin the generational process
                    while True:
                        # Step 3: Environmental selection
                        archive = self.tb.select(pop + archive, k=self.nBar)
                        prevFront = deepcopy(self.front)
                        self.front.update(archive)

                        # Record statistics
                        record = self.mstats.compile(self.front)
                        log.record(gen=currGen, evals=evals, **record)
                        print('\r', log.stream)

                        # Step 4: Termination
                        if self.termination(prevFront, currGen):
                            result['logs'][pathKey][subPathKey] = log
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
                        fits = self.tb.map(self.evaluator.evaluate, invInds1)
                        for ind, fit in zip(invInds1, fits):
                            ind.fitness.values = fit

                        # Archive
                        invInds2 = [ind for ind in archive if not ind.fitness.valid]
                        fits = self.tb.map(self.evaluator.evaluate, invInds2)
                        for ind, fit in zip(invInds2, fits):
                            ind.fitness.values = fit

                        evals = len(invInds1) + len(invInds2)

                        currGen += 1

            return result

    def compute(self, startEnd, recompute=False, startDate=None, current=False,
                weather=False, algorithm='NSGA2', seed=None, avoidArctic=True, avoidAntarctic=True):
        support.clear_caches()  # Clear caches

        if startEnd[0] == startEnd[1]:
            return 'equal_start_end'

        antarctic_circle = Polygon([(-180, -66), (180, -66), (180, -89), (-180, -89)])
        arctic_circle = Polygon([(-180, 66), (180, 66), (180, 89), (-180, 89)])

        start, end = startEnd
        startPoint, endPoint = Point(start), Point(end)

        if antarctic_circle.contains(startPoint) or arctic_circle.contains(endPoint):
            avoidAntarctic = False
        if arctic_circle.contains(startPoint) or arctic_circle.contains(endPoint):
            avoidArctic = False

        if startDate is not None:
            dateString = startDate.strftime('%Y%m%d')
        else:
            dateString = None

        fn = "{}_C{}_W{}_d{}_inclS{}_inclN{}_V{}_T{}_C{}_FP{}_ECA{}".format(startEnd, current, weather,
                                                                            dateString, avoidAntarctic,
                                                                            avoidArctic, self.vessel.name,
                                                                            self.criteria['minimalTime'],
                                                                            self.criteria['minimalCost'],
                                                                            self.fuelPrice, self.ecaFactor)
        self.procResultsFP = DIR / "output/processedResults/" / fn

        if not recompute and os.path.exists(self.procResultsFP):
            return None

        newParameters, reinitialize = {}, False
        if self.p['avoidAntarctic'] != avoidAntarctic:
            reinitialize = True
            newParameters['avoidAntarctic'] = avoidAntarctic
        if self.p['avoidArctic'] != avoidArctic:
            newParameters['avoidArctic'] = avoidArctic
            reinitialize = True

        self.update_parameters(newParameters, reinitialize=reinitialize)

        key = tuple(sorted([start, end]))
        # Get initial paths
        initialPaths = None
        for path in self.initialPaths:
            if path['startEndKey'] == key and path['avoidAntarctic'] == avoidAntarctic and \
                    path['avoidArctic'] == avoidArctic:
                initialPaths = path['paths']
                break
        if not initialPaths:
            initialPaths = self.initializer.get_initial_routes(start, end)

            pathOutput = {'avoidAntarctic': avoidAntarctic,
                          'avoidArctic': avoidArctic,
                          'startEndKey': key,
                          'paths': initialPaths}

            self.initialPaths.append(pathOutput)

            fn = str(uuid.uuid4())
            with open(self.initPathsDir / fn, 'wb') as file:
                pickle.dump(pathOutput, file)

        if algorithm == 'MPAES':
            GA = self.mpaes.optimize
        elif algorithm == 'SPEA2':
            GA = self.spea2.optimize
        else:
            GA = self.nsgaii.optimize

        return GA(startEnd, initialPaths, startDate, current, weather, seed)

    def update_parameters(self, newParameters, reinitialize=False):
        def set_attrs(_self, **kwargs):
            for k, v in kwargs.items():
                setattr(_self, k, v)

        self.p = {**self.p, **newParameters}

        # M-PAES class
        set_attrs(self.mpaes,
                  gen=self.p['gen'],
                  n=self.p['n'],
                  nBar=self.p['nBar'],
                  recomb=self.p['recomb'],
                  nM=int(self.mpaes.n / 2 / (self.mpaes.recomb + 1)),
                  fails=self.p['fails'],
                  moves=self.p['moves'])

        # SPEA2 class
        set_attrs(self.spea2,
                  gen=self.p['gen'],
                  n=self.p['n'],
                  nBar=self.p['nBar'],
                  cxpb=self.p['cxpb'],
                  mutpb=self.p['mutpb'])

        # NSGA-II class
        set_attrs(self.nsgaii,
                  gen=self.p['gen'],
                  n=self.p['n'],
                  cxpb=self.p['cxpb'],
                  mutpb=self.p['mutpb'])

        # Operator class
        set_attrs(self.operators,
                  radius=self.p['radius'],
                  cov=[[self.operators.radius, 0],
                       [0, self.operators.radius]],
                  widthRatio=self.p['widthRatio'],
                  shape=self.p['shape'],
                  scaleFactor=self.p['scaleFactor'],
                  delFactor=self.p['delFactor'],
                  ops=self.p['mutationOperators'],
                  gauss=self.p['gauss'])

        if reinitialize:
            # Re-populate R-Tree structures
            navAreaGenerator = NavigableAreaGenerator(self.p, DIR=DIR)
            landTree = navAreaGenerator.get_shoreline_rtree()
            ecaTree = navAreaGenerator.get_eca_rtree()

            set_attrs(self.evaluator,
                      treeDict=landTree,
                      ecaTreeDict=ecaTree)
            self.initializer = initialization.Initializer(self.evaluator, self.vessel, landTree, ecaTree,
                                                          self.geod, self.p, creator.Individual, DIR)

    def get_days(self, pop):
        """
        Get estimate of max travel time of inds in *pop* in whole days
        """
        boatSpeed = min(self.vessel.speeds)
        maxTravelTime = 0
        for ind in pop:
            travelTime = 0.0
            for i in range(len(ind) - 1):
                p1, p2 = sorted((ind[i][0], ind[i+1][0]))
                edgeDist = self.geod.distance(p1, p2)
                edgeTravelTime = edgeDist / boatSpeed
                travelTime += edgeTravelTime
            if travelTime > maxTravelTime:
                maxTravelTime = travelTime
        days = int(math.ceil(maxTravelTime / 24))
        print('Number of days:', days)
        return days

    def create_route_response(self, obj, bestWeighted, wps, objValue, fitValue, xCanals):
        return {'optimizationCriterion': obj,
                'bestWeighted': bestWeighted,
                'distance': self.geod.total_distance(wps),
                'fuelCost': objValue[1],
                'travelTime': objValue[0],
                'fitValues': fitValue.tolist(),
                'waypoints': [{'lon': wp[0][0],
                               'lat': wp[0][1],
                               'speed': wp[1]} for wp in wps],
                'crossedCanals': xCanals}

    def post_process(self, result, ID=None):
        if result is None:
            with open(self.procResultsFP, 'rb') as file:
                processedResults = pickle.load(file)
            with open(self.procResultsFP.as_posix() + '_raw', 'rb') as file:
                result = pickle.load(file)
            return processedResults, result
        elif result == 'equal_start_end':
            processedResults = {'routeResponse': [],
                                'units': {'travelTime': 'days', 'fuelCost': 'euros', 'distance': 'nautical miles'}}

            for obj in [obj for obj, included in self.criteria.items() if included]:
                processedResults['routeResponse'].append({'optimizationCriterion': obj,
                                                          'bestWeighted': False,
                                                          'distance': 0.0,
                                                          'fuelCost': 0.0,
                                                          'travelTime': 0.0,
                                                          'fitValues': [0.0, 0.0],
                                                          'waypoints': [],
                                                          'crossedCanals': []})
            return processedResults

        nFits = len([included for included in self.criteria.values() if included])
        objKeys = [obj for obj, included in self.criteria.items() if included]
        objIndices = {'minimalTime': 0, 'minimalCost': 1}
        processedResults = {'routeResponse': [],
                            'initialRoutes': result['initialRoutes'],
                            'units': {'travelTime': 'days',
                                      'fuelCost': 'euros',
                                      'distance': 'nautical miles'}}
        if ID:
            processedResults['id'] = ID

        # Get minimum fuel route and minimum time route for each path
        # Then create output dictionary
        for i, pathFront in enumerate(result['fronts']):
            bestWeighted = {'bestWeighted': True}
            xCanals = result['initialRoutes'][i]['xCanals']

            # Get bestWeighted route
            wps, bestWeightedObjValue, fitValue = [], np.zeros(2), np.zeros(nFits)
            for subFront in pathFront:
                ind = subFront[0]
                bestWeightedObjValue += self.evaluator.evaluate(ind, revert=False, includePenalty=False)
                fitValue += ind.fitness.values
                wps.extend(ind)

            bestWeightedResponse = self.create_route_response('bestWeighted',
                                                              True,
                                                              wps,
                                                              bestWeightedObjValue,
                                                              fitValue,
                                                              xCanals)

            for obj in objKeys:
                i = objIndices[obj]

                # Initialize lists and arrays
                wps, objValue, fitValue = [], np.zeros(2), np.zeros(nFits)

                # A path is split into sub paths by canals, so we merge the sub path results
                for subFront in pathFront:
                    # Evaluate fuel and time for each individual in front
                    # NB: Using same parameter settings for Evaluator as in optimization
                    subObjValues = np.asarray([self.evaluator.evaluate(ind, revert=False, includePenalty=False)
                                               for ind in subFront])

                    # Get best individual
                    idx = np.argmin(subObjValues[:, i])
                    ind = subFront[idx]
                    wps.extend(ind)
                    objValue += subObjValues[idx]
                    fitValue += ind.fitness.values

                # Check whether 'best weighted' route is equal to other best routes
                if np.array_equal(objValue, bestWeightedObjValue):
                    print("'{}' is best weighted route".format(obj))
                    bestWeighted[obj] = True
                    bestWeighted['bestWeighted'] = False
                else:
                    bestWeighted[obj] = False

                routeResponse = self.create_route_response(obj=obj,
                                                           bestWeighted=bestWeighted[obj],
                                                           wps=wps,
                                                           fitValue=fitValue,
                                                           objValue=objValue,
                                                           xCanals=xCanals)

                processedResults['routeResponse'].append(routeResponse)

            # If 'best weighted' route is not equal to other best routes, append its response
            if bestWeighted['bestWeighted']:
                processedResults['routeResponse'].append(bestWeightedResponse)

        try:
            with open(self.procResultsFP, 'wb') as file:
                pickle.dump(processedResults, file)
            with open(self.procResultsFP.as_posix() + '_raw', 'wb') as file:
                pickle.dump(result, file)
        except TypeError:
            print("Save filepath is 'None': Result is not saved")

        return processedResults, result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # import multiprocessing as mp
    # import pprint
    import time

    from case_studies.plot_results import RoutePlotter
    from datetime import datetime
    from scoop import futures
    from support import locations

    startTime = time.time()
    _startEnd = (locations['Caribbean Sea'], locations['North UK'])

    parameters = {'gen': 200,  # Min number of generations
                  'n': 100}    # Population size

    kwargsPlanner = {'inputParameters': parameters, 'tb': _tb, 'criteria': _criteria}
    kwargsCompute = {'startEnd': _startEnd, 'startDate': datetime(2019, 3, 1), 'recompute': True, 'current': False,
                     'weather': True, 'seed': 1}
    multiprocess = True

    if multiprocess:
        # with mp.Pool() as pool:
        _tb.register("map", futures.map)

        planner = RoutePlanner(**kwargsPlanner)
        rawResults = planner.compute(**kwargsCompute)
    else:
        planner = RoutePlanner(**kwargsPlanner)
        rawResults = planner.compute(**kwargsCompute)

    procResults, rawResults = planner.post_process(rawResults)
    routePlotter = RoutePlotter(procResults, rawResults=rawResults, vessel=planner.vessel)
    fig, ax = routePlotter.results(initial=True, ecas=False, nRoutes=5, colorbar=True)

    # pp = pprint.PrettyPrinter(depth=6)
    # pp.pprint(post_processed_results)
    print("--- %s seconds ---" % (time.time() - startTime))

    plt.show()
