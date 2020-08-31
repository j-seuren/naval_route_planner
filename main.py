import functools
import initialization
import math
import numpy as np
import os
import pickle
import random
import support
import time
import uuid

from copy import deepcopy
from datetime import datetime
from data_config.navigable_area import NavigableAreaGenerator
from deap import base, creator, tools, algorithms
from evaluation import route_evaluation, geodesic
from operations import Operators
from os import listdir
from pathlib import Path
from shapely.geometry import Point, Polygon


class RoutePlanner:
    def __init__(self,
                 vesselName='Fairmaster',
                 ecaFactor=1.2,
                 fuelPrice=0.3,
                 inputParameters=None,
                 objectives=None):
        if objectives is None:
            objectives = ['minTime', 'minFuel']
        if 'minTime' in objectives and 'minFuel' in objectives:
            weights = (-5, -1)
        elif 'minTime' in objectives or 'minFuel' in objectives:
            weights = (-1, )
        else:
            raise Exception('Specify at least one objective to minimize')

        self.objectives = objectives

        defaultParameters = {'avoidAntarctic': True,
                             'avoidArctic': True,
                             'splits': 10,         # Threshold for split_polygon
                             'nBar': 50,           # Local archive size (M-PAES, SPEA2)
                             'gen': 300,           # Number of generations
                             'n': 100,             # Population size
                             'cxpb': 0.85,         # Crossover probability (NSGAII, SPEA2)
                             'mutpb': 0.66,        # Mutation probability (NSGAII, SPEA2)
                             'recomb': 5,          # Max recombination trials (M-PAES)
                             'fails': 5,
                             'moves': 10,
                             'widthRatio': 4.8,    # 4.8 obtained from hyp param tuning
                             'radius': 4.6,        # 4.6 obtained from hyp param tuning
                             'shape': 3,
                             'scaleFactor': 0.1,
                             'delFactor': 1.2,
                             'mutationOperators': ['speed', 'insert', 'move', 'delete'],
                             'gauss': False,
                             'stopCriterionRate': 0.1,
                             'res': 'c',
                             'graphDens': 6,
                             'graphVarDens': 4
                             }

        # Set parameters
        if inputParameters:
            self.p = {**defaultParameters, **inputParameters}
        else:
            self.p = defaultParameters

        # Create Fitness and Individual types
        creator.create("FitnessMin", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.tb = base.Toolbox()                # Function toolbox
        self.vessel = support.Vessel(vesselName)  # Vessel class instance
        self.ecaFactor = ecaFactor              # Multiplication factor ECA fuel
        self.fuelPrice = fuelPrice              # Fuel price [1000 dollars / mt]
        self.geod = geodesic.Geodesic()         # Geodesic class instance
        self.resolution = self.p['res']

        # Initialize stopping criterion variables
        self.prioriMBR = 1
        self.posterioriMBR = None
        self.prioriVarMBR = 0.1

        # Load and pre-process shoreline and ECA polygons
        navAreaGenerator = NavigableAreaGenerator(self.p)
        self.treeDict = navAreaGenerator.get_shoreline_tree()
        self.ecaTreeDict = navAreaGenerator.get_eca_tree

        # Initialize "Initializer"
        self.initializer = initialization.Initializer(self.vessel,
                                                      self.treeDict,
                                                      self.ecaTreeDict,
                                                      self.tb,
                                                      self.geod,
                                                      self.p,
                                                      creator.Individual)

        # Load previously calculated initial paths
        self.initPathsDir = Path('output/initialPaths/RES_{}_D{}_VD_{}'.format(self.p['res'],
                                                                               self.p['graphDens'],
                                                                               self.p['graphVarDens']))
        if not os.path.exists(self.initPathsDir):
            os.mkdir(self.initPathsDir)
        files = listdir(self.initPathsDir)
        self.initialPaths = []
        for fp in files:
            with open(self.initPathsDir / fp, 'rb') as file:
                self.initialPaths.append(pickle.load(file))

        # Initialize "Evaluator" and register it's functions
        self.evaluator = route_evaluation.Evaluator(self.vessel,
                                                    self.treeDict,
                                                    self.ecaTreeDict,
                                                    ecaFactor,
                                                    self.fuelPrice,
                                                    self.geod,
                                                    'minTime' not in objectives)
        self.tb.register("e_feasible", self.evaluator.e_feasible)
        self.tb.register("feasible", self.evaluator.feasible)
        self.tb.register("evaluate", self.evaluator.evaluate)
        self.tb.decorate("evaluate", tools.DeltaPenalty(self.tb.feasible,
                                                        [1e+20, 1e+20]))

        # Initialize "Operator" and register it's functions
        self.operators = Operators(self.tb, self.vessel, self.geod, self.p)
        self.tb.register("mutate", self.operators.mutate)
        self.tb.register("mate", self.operators.cx_one_point)

        self.tb.register("population", initialization.init_repeat_list)

        # Initialize Statistics and logbook
        self.mstats = support.statistics()
        self.log = support.logbook()

        # Initialize ParetoFront class and replace its 'update' function
        self.front = tools.ParetoFront()
        self.front.update = functools.partial(support.update, self.front)

        # Initialize algorithm classes
        self.mpaes = self.MPAES(self.tb, self.evaluator, self.mstats, self.log,
                                self.front, self.get_days,
                                self.stopping_criterion, self.p)
        self.spea2 = self.SPEA2(self.tb, self.evaluator, self.mstats, self.log,
                                self.front, self.get_days,
                                self.stopping_criterion, self.p)
        self.nsgaii = self.NSGAII(self.tb, self.evaluator, self.mstats,
                                  self.front, self.get_days,
                                  self.stopping_criterion, self.p)

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
                    # Update global Pareto front with m, as necessary
                    mDominates, mDominated, _, sizeCurr = self.front.update([m])
                    self.nDomPC += mDominates
                    self.nDomCP += mDominated
                moves += 1
            return c

        def compute(self, startEnd, initPaths, startDate, inclCurr, inclWeather, seed=None):
            random.seed(seed)

            result = {'startEnd': startEnd, 'initialRoutes': initPaths, 'logs': {}, 'fronts': {}}
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

                    currGen = 0
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
                        # Step 3: Update global Pareto front
                        self.nDomPC, self.nDomCP, sizePrev, sizeCurr = self.front.update(pop)

                        # Record statistics
                        record = self.mstats.compile(self.front)
                        self.log.record(gen=currGen, evals=self.evals, **record)
                        print('\r', self.log.stream)

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
            self.localFront = []

            self.get_days = get_days
            self.stopping_criterion = stopping_criterion

            # Parameter settings
            self.gen = par['gen']
            self.n = par['n']
            self.cxpb = par['cxpb']
            self.mutpb = par['mutpb']

            # Register NSGA2 selection function
            self.tb.register("select", tools.selNSGA2)

        def compute(self, startEnd, initRoutes, startDate, inclCurr, inclWeather, seed):
            random.seed(seed)

            result = {'startEnd': startEnd, 'initialRoutes': initRoutes, 'logs': {}, 'fronts': {}}
            for rKey, route in initRoutes.items():
                result['logs'][rKey], result['fronts'][rKey] = {}, {}
                print('Path {0}/{1}'.format(rKey + 1, len(initRoutes)))
                for spKey, subPath in route['path'].items():
                    print('Sub path {0}/{1}'.format(spKey+1, len(route['path'])))

                    # Reset functions and caches
                    self.tb.register("individual", initialization.init_individual, self.tb, subPath)
                    self.front.clear()
                    log = support.logbook()

                    # Step 1: Initialization
                    print('Initializing population from shortest path:', end='\n ')
                    pop = self.tb.population(self.tb.individual,
                                             int(self.n / len(subPath.values())))
                    offspring = []
                    currGen = 0
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
                        sizePrev = len(self.localFront)
                        self.localFront, nDomHofers, nDomInds = self.front.update(pop, self.localFront)

                        # Record statistics
                        record = self.mstats.compile(self.front)
                        log.record(gen=currGen, evals=evals, **record)
                        print('\r', log.stream)

                        # Step 4: Termination
                        if self.stopping_criterion(nDomHofers, nDomInds, sizePrev, len(self.localFront), currGen):
                            result['logs'][rKey][spKey] = deepcopy(log)
                            result['fronts'][rKey][spKey] = deepcopy(self.front)
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

        def compute(self, startEnd, initPaths, startDate, inclCurr, inclWeather, seed):
            random.seed(seed)

            result = {'startEnd': startEnd, 'initialRoutes': initPaths, 'logs': {}, 'fronts': {}}
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
                    currGen = 0
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

    def compute(self, startEnd, startDate=None, inclCurr=False,
                inclWeather=False, algorithm='NSGA2', seed=None, avoidArctic=True, avoidAntarctic=True):
        self.evaluator.startDate = startDate  # Update start date

        antarctic_circle = Polygon([(-180, -66), (180, -66), (180, -89), (-180, -89)])
        arctic_circle = Polygon([(-180, 66), (180, 66), (180, 89), (-180, 89)])

        start, end = startEnd
        startPoint, endPoint = Point(start), Point(end)

        if antarctic_circle.contains(startPoint) or arctic_circle.contains(endPoint):
            avoidAntarctic = False
        if arctic_circle.contains(startPoint) or arctic_circle.contains(endPoint):
            avoidArctic = False

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
            initialPaths = self.initializer.get_initial_paths(start, end)

            pathOutput = {'avoidAntarctic': avoidAntarctic,
                          'avoidArctic': avoidArctic,
                          'startEndKey': key,
                          'paths': initialPaths}

            self.initialPaths.append(pathOutput)

            fn = str(uuid.uuid4())
            with open(self.initPathsDir / fn, 'wb') as file:
                pickle.dump(pathOutput, file)

        if algorithm == 'MPAES':
            GA = self.mpaes.compute
        elif algorithm == 'SPEA2':
            GA = self.spea2.compute
        else:
            GA = self.nsgaii.compute

        return GA(startEnd, initialPaths, startDate, inclCurr, inclWeather, seed)

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
            navAreaGenerator = NavigableAreaGenerator(self.p)
            self.treeDict = navAreaGenerator.get_shoreline_tree()
            self.ecaTreeDict = navAreaGenerator.get_eca_tree

            set_attrs(self.evaluator,
                      treeDict=self.treeDict,
                      ecaTreeDict=self.ecaTreeDict)
            self.initializer = initialization.Initializer(self.vessel, self.treeDict, self.ecaTreeDict, self.tb,
                                                          self.geod, self.p, creator.Individual)

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

    def stopping_criterion(self, nDomHofers, nDomInds, sizePrev, sizeCurr, t):
        """The MDR indicator provides different types of information.
        MDR =  1:   New population is completely better than its predecessor.
        MDR =  0:   No substantial progress.
        MDR = -1:   New population does not improve any solution of its predecessor.
        """

        # Calculate MDR indicator of current generation (t)
        if t == 0:
            MDR = 1
        else:
            MDR = nDomHofers / sizePrev - nDomInds / sizeCurr

        # Compute the Kalman with estimated variance of MBR (var=0 for t=0)
        K = self.prioriVarMBR / (self.prioriVarMBR + self.p['stopCriterionRate'])

        # A posteriori estimation of the MBR indicator (a priori estimate=1 for t=0)
        self.posterioriMBR = self.prioriMBR + K * (MDR - self.prioriMBR)

        # Compute a priori estimate of accuracy of MBR of next generation (t+1) as accuracy of current MBR estimate
        # A posteriori error covariance estimate Pt = (I - KtH)Pt-
        posterioriVarMBR = (1 - K) * self.prioriVarMBR
        self.prioriVarMBR = posterioriVarMBR
        # self.prioriVarMBR = (MDR - self.posterioriMBR) ** 2

        # Set next a priori MBR indicator of (t+1) as a posteriori MBR indicator of t
        self.prioriMBR = self.posterioriMBR

        # if self.posterioriMBR < 0:
        #     print('STOPPING: MGBM criterion')
        #     return True

        if t >= self.p['gen']:
            print('STOPPING: Max generations')
            return True

        return False

    def post_process(self, result, ID=None):
        objKeys = self.objectives + ['bestWeighted']
        processedResults = {'paths': [],
                            'units': {'travelTime': 'days',
                                      'fuelCost': 'euros',
                                      'distance': 'nautical miles'}}
        if ID:
            processedResults['id'] = ID

        # Get minimum fuel route and minimum time route for each path
        # Then create output dictionary
        for pathKey, path in result['fronts'].items():
            # Initialize dictionaries
            pathResult = {'crossedCanals': result['initialRoutes'][pathKey]['xCanals']}
            bestInds = {key: [] for key in objKeys}
            bestFitInds = {key: [] for key in objKeys}
            objValue = {obj: np.zeros(2) for obj in objKeys}
            fitValue = {obj: np.zeros(2) for obj in objKeys}
            route = {obj: [] for obj in objKeys}

            # A path is split into sub paths by canals, hence, we merge the sub path results
            for frontKey, subFront in path.items():

                # Evaluate fuel and time for each individual in front
                # NB: Using same parameter settings for Evaluator as in optimization
                indObjVal = np.asarray([self.tb.evaluate(ind) for ind in subFront])
                indFitVal = np.asarray([ind.fitness.values for ind in subFront])
                if self.evaluator.revertOutput:
                    indObjVal = indObjVal[:, [1, 0]]

                for i, obj in enumerate(objKeys):
                    # Get best individual according to objective (obj)
                    if obj == 'bestWeighted':
                        minIdx = minFitIdx = 0
                    else:
                        minIdx, minFitIdx = np.argmin(indObjVal[:, i]), np.argmin(indFitVal[:, i])
                    bestInd, bestFitInd = subFront[minIdx], subFront[minFitIdx]
                    bestInds[obj].append(bestInd)
                    bestFitInds[obj].append(bestFitInd)

                    objValue[obj] += indObjVal[minIdx, :]
                    fitValue[obj] += indFitVal[minIdx, :]

                    route[obj].extend([wp for wp in bestInd])

            substituteBestWeighted = False
            for obj in objKeys:
                # Check whether 'best weighted' route is equal to other best individuals
                if obj != 'bestWeighted' and np.array_equal(objValue[obj], objValue['bestWeighted']):
                    print("Substituted 'best weighted' route with the '{}' route".format(obj))
                    substituteBestWeighted = obj

                # If 'best weighted' route is equal to other best routes, create reference
                if obj == 'bestWeighted' and substituteBestWeighted:
                    routeResponse = obj
                else:
                    print(pathKey, objValue[obj])
                    routeResponse = {'fuelCost': objValue[obj][1],
                                     'travelTime': objValue[obj][0],
                                     'fitValues': fitValue[obj],
                                     'distance': self.geod.total_distance(route[obj]),
                                     'waypoints': [{'lon': wp[0][0], 'lat': wp[0][1], 'speed': wp[1]}
                                                   for i, wp in enumerate(route[obj])]}

                pathResult[obj] = routeResponse

            processedResults['paths'].append(pathResult)

        return processedResults


if __name__ == "__main__":
    import pprint
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

    _startEnd = ((5.077, 54.376), (103.571, 1.298))

    parameters = {'gen': 100,  # Number of generations
                  'graphDens': 8,
                  'graphVarDens': 2,
                  'n': 50}    # Population size

    planner = RoutePlanner(inputParameters=parameters)
    _results = planner.compute(_startEnd,
                               startDate=datetime(2020, 8, 16),
                               inclCurr=False,
                               inclWeather=False,
                               seed=1)

    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(planner.post_process(_results))

    # Save parameters
    timestamp = datetime.now()

    # Save result
    resultFN = '{0:%H_%M_%S}'.format(timestamp)
    with open('output/result/' + resultFN, 'wb') as _file:
        pickle.dump(_results, _file)
    print('Saved result to: ' + 'output/result/' + resultFN)

    print("--- %s seconds ---" % (time.time() - startTime))
