import evaluation
import initialization
import math
import numpy as np
import os
import indicators
import pickle
import pprint
import random
import support
import uuid

from copy import deepcopy
from data_config.navigable_area import NavigableAreaGenerator
from deap import base, creator, tools, algorithms
import geodesic
from operations import Operators
from pathlib import Path
from shapely.geometry import Point


pp = pprint.PrettyPrinter()
_criteria = {'minimalTime': True, 'minimalCost': True}
timeWeight, costWeight = -5, -1

if _criteria['minimalTime'] and _criteria['minimalCost']:
    _criteria = {'minimalTime': timeWeight, 'minimalCost': costWeight}
elif _criteria['minimalCost']:
    _criteria = {'minimalCost': -1}
else:
    _criteria = {'minimalTime': -1}

DIR = Path('D:/')
creator.create("FitnessMin", base.Fitness, weights=tuple(_criteria.values(),))
creator.create("Individual", list, fitness=creator.FitnessMin)
_tb = base.Toolbox()


def crowding_distance(pop):
    return tools.selNSGA2(pop, len(pop))


class RoutePlanner:
    def __init__(self,
                 speedIdx=0,
                 vesselName='Fairmaster_2',
                 shipLoading='normal',
                 ecaFactor=1.5593,
                 fuelPrice=300,  # Fuel price per metric tonne
                 bathymetry=False,
                 inputParameters=None,
                 tb=None,
                 criteria=None):
        if criteria is None:
            criteria = _criteria
        else:
            weights = (-1, -1) if criteria['minimalTime'] and criteria['minimalCost'] else (-1,)

            creator.create("FitnessMin", base.Fitness, weights=weights)
            creator.create("Individual", list, fitness=creator.FitnessMin)

        # Set parameters
        defaultParameters = {
                             # Navigation area parameters
                             'avoidAntarctic': True,
                             'avoidArctic': True,
                             'res': 'i',           # Resolution of shorelines
                             'penaltyValue': 1,    # Penalty value for Bathymetry
                             'graphDens': 4,       # Recursion level graph
                             'graphVarDens': 6,    # Variable recursion level graph
                             'splits': 3,          # Threshold for split_polygon (val 3 yields best performance)

                             # MOEA parameters
                             'n': 336,             # Population size
                             'nBar': 100,          # Local archive size (M-PAES)
                             'cxpb': 0.81,          # Crossover probability (NSGAII, SPEA2)
                             'mutpb': 0.28,        # Mutation probability (NSGAII, SPEA2)
                             'nMutations': 9,      # Max. number of mutations per selected individual
                             'cr_trials': 5,       # Max recombination trials (M-PAES)
                             'l_fails': 3,         # Max fails (M-PAES)
                             'l_opt': 5,           # Max moves (M-PAES)

                             # Stopping parameters
                             'maxEvaluations': None,
                             'gen': 400,           # Minimal number of generations
                             'maxGDs': 33,         # Max length of generational distance list
                             'minVar': 5e-6,       # Minimal variance of generational distance list

                             # Mutation parameters
                             'mutationOperators': ['speed', 'insert', 'move', 'delete'],  # Operators to be included
                             'widthRatio': 1.5,    # 7.5e-4 obtained from hyp param tuning
                             'radius': 0.4,        # 0.39 obtained from hyp param tuning
                             'scaleFactor': 0.1,   # Scale factor for Exponential distribution
                             'delFactor': 1.2,     # Factor of deletions
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
        bathTree = navAreaGenerator.get_bathymetry_rtree()

        # Initialize "Evaluator" and register it's functions
        self.evaluator = evaluation.Evaluator(self.vessel,
                                              landTree,
                                              ecaTree,
                                              bathTree if bathymetry else None,
                                              ecaFactor,
                                              self.geod,
                                              criteria,
                                              self.p,
                                              DIR=DIR)
        self.speedIdx = speedIdx
        # Initialize "Initializer"
        self.initializer = initialization.Initializer(self.evaluator,
                                                      self.vessel,
                                                      landTree,
                                                      ecaTree,
                                                      bathTree,
                                                      self.geod,
                                                      self.p,
                                                      creator.Individual,
                                                      self.speedIdx,
                                                      DIR)

        # Load previously calculated initial paths
        self.initPathsDir = DIR / 'output/initialRoutes/RES_{}_D{}_VD_{}'.format(self.p['res'],
                                                                                 self.p['graphDens'],
                                                                                 self.p['graphVarDens'])
        if not os.path.exists(self.initPathsDir):
            os.mkdir(self.initPathsDir)
        self.initRoutesList = []
        if speedIdx == 0:  # Initial paths are computed for speedIdx = 0
            for fp in os.listdir(self.initPathsDir):
                with open(self.initPathsDir / fp, 'rb') as file:
                    self.initRoutesList.append(pickle.load(file))

        # Initialize "Operator" and register it's functions
        self.operators = Operators(self.evaluator.e_feasible, self.vessel, self.geod, self.p)
        self.tb.register("mutate", self.operators.mutate)
        self.tb.register("mate", self.operators.cx_one_point)
        self.tb.register("population", initialization.init_repeat_list)

    class Terminator:
        def __init__(self, p):
            self.maxEvaluations = p['maxEvaluations']
            self.minGen = p['gen']
            self.minVar = p['minVar']
            self.maxStops = p['maxGDs']

        def termination(self, prevFront, front, gen, gds, evals):
            if self.maxEvaluations:
                if evals > self.maxEvaluations:
                    return True
            else:
                gd = indicators.generational_distance(prevFront, front)
                gds.append(gd)
                if len(gds) > self.maxStops:
                    gds.pop(0)
                if gen >= self.minGen:
                    if np.var(gds) < self.minVar:
                        print('STOPPING: Generational distance')
                        return True
                return False

    class MPAES:
        def __init__(self, tb, evaluator, p, terminator):
            # Set functions / classes
            self.tb = tb
            self.evaluator = evaluator
            self.termination = terminator.termination

            # MOEA parameters
            # M-PAES characteristic parameters
            self.nBar = p['nBar']
            self.l_opt = p['l_opt']
            self.l_fails = p['l_fails']
            self.cr_trials = p['cr_trials']
            self.sizePop = p['n']
            self.evals = 0  # Initialize counter variables

        def test(self, c, m, archive):
            if len(archive) < self.nBar:
                archive.append(m)
                crowding_distance(archive)
            else:
                x = random.choice(archive)
                crowding_distance(archive + [m])
                if m.fitness.crowding_dist < x.fitness.crowding_dist:
                    del archive[np.argmax([ind.fitness.crowding_dist for ind in archive])]
                    archive.append(m)
                    crowding_distance(archive)
            return m if m.fitness.crowding_dist < c.fitness.crowding_dist else c

        def paes(self, c, archive, front):
            fails = moves = 0
            while fails < self.l_fails and moves < self.l_opt:
                m, = self.tb.mutate(self.tb.clone(c))
                m.fitness.values = self.evaluator.evaluate(m)
                self.evals += 1

                if c.fitness.dominates(m.fitness):
                    fails += 1
                    continue
                elif m.fitness.dominates(c.fitness):
                    archive.append(m)
                    c = m
                    fails = 0
                else:
                    # If m is dominated by any member of H, discard m
                    dominated = False
                    for ind in archive:
                        if ind.fitness.dominates(m.fitness):
                            dominated = True
                            break
                    if dominated:
                        continue
                    else:
                        # Determine which becomes the new candidate
                        # and whether to add m to archive
                        c = self.test(c, m, archive)
                front.update([m])
                moves += 1
            return c

        def optimize(self, pop, result, routeIdx, subIdx):
            mstats, log = support.statistics(), support.logbook()  # DEAP statistics and logbook
            front = tools.ParetoFront()  # Initialize ParetoFront class
            self.evals = len(pop)
            gds = []  # Initialize generational distance list
            gen = totalEvals = 0
            while True:
                # Step 3: Update global Pareto front
                prevFront = front.items[:]
                front.update(pop)

                # Record statistics
                record = mstats.compile(front)
                log.record(gen=gen, evals=self.evals, **record)
                print('\r', log.stream)

                totalEvals += self.evals
                self.evals = 0

                # Step 4: Local search
                for i, c in enumerate(pop):
                    # Fill local archive with solutions from Pareto front that do not dominate c
                    archive = [ind for ind in front if not ind.fitness.dominates(c.fitness) and ind is not c]
                    archive.append(c)  # Copy candidate into H
                    pop[i] = self.paes(c, archive, front)  # Replace c with improved version by local search

                # Step 5: Recombination
                popInter = []  # Initialize intermediate population
                while len(popInter) < self.sizePop:
                    r = 0
                    while True:
                        # Randomly choose two parents from P + G
                        mom, dad = tools.selRandom(pop + front.items, 2)
                        mom2, dad2 = self.tb.clone(mom), self.tb.clone(dad)

                        # Recombine to form offspring, evaluate
                        child, _ = self.tb.mate(mom2, dad2)
                        child.fitness.values = self.evaluator.evaluate(child)
                        self.evals += 1
                        childInMoreCrowdedArea = childDominated = False
                        for ind in front:
                            if ind.fitness.dominates(child.fitness):
                                childDominated = True
                                break

                        if not childDominated:
                            crowding_distance(pop + front.items + [child])
                            # Check if c is in more crowded grid location than both parents
                            if child.fitness.crowding_dist < mom.fitness.crowding_dist and\
                                    child.fitness.crowding_dist < dad.fitness.crowding_dist:
                                childInMoreCrowdedArea = True

                            # Update pareto front with c as necessary
                            front.update([child])
                        r += 1
                        if not (
                            (childDominated or childInMoreCrowdedArea) and r < self.cr_trials
                        ):
                            break

                    childAsList = tools.selTournament(front.items, k=1, tournsize=2) if childDominated else [child]

                    popInter.extend(childAsList)

                # Step 4: Termination
                if self.termination(prevFront, front, gen, gds, totalEvals):
                    return front, log

                pop = popInter
                gen += 1

    class NSGAII:
        def __init__(self, tb, evaluator, p, terminator):
            # Set functions / classes
            self.tb = tb
            self.evaluator = evaluator
            self.tb.register("select", tools.selNSGA2)  # Set NSGA2 selection
            self.termination = terminator.termination

            # MOEA parameters
            self.sizePop = p['n']
            self.cxpb = p['cxpb']
            self.mutpb = p['mutpb']

        def optimize(self, pop, result, routeIdx, subIdx):
            mstats, log = support.statistics(), support.logbook()  # DEAP statistics and logbook
            front = tools.ParetoFront()  # Initialize ParetoFront class
            evals = len(pop)
            offspring, gds = [], []  # Initialize offspring and generational distance list
            gen = totalEvals = 0
            while True:
                # Step 3: Environmental selection (and update HoF)
                pop = self.tb.select(pop + offspring, self.sizePop)
                prevFront = deepcopy(front)
                front.update(pop)

                # Record statistics
                record = mstats.compile(front)
                log.record(gen=gen, evals=evals, **record)
                print('\r', log.stream)

                totalEvals += evals
                # Step 4: Termination
                if self.termination(prevFront, front, gen, gds, totalEvals):
                    return front, log

                # Step 5: Variation
                offspring = algorithms.varAnd(pop, self.tb, self.cxpb, self.mutpb)

                # Step 2: Fitness assignment
                invInds = [ind for ind in offspring if not ind.fitness.valid]
                fits = self.tb.map(self.evaluator.evaluate, invInds)
                for ind, fit in zip(invInds, fits):
                    ind.fitness.values = fit

                evals = len(invInds)

                gen += 1

    class SPEA2:
        def __init__(self, tb, evaluator, p, terminator):
            # Set functions / classes
            self.tb = tb
            self.evaluator = evaluator
            self.tb.register("select", tools.selSPEA2)  # Set SPEA2 selection
            self.termination = terminator.termination

            # MOEA Parameters
            self.sizePop = p['n']
            self.cxpb = p['cxpb']
            self.mutpb = p['mutpb']
            self.sizeArchive = self.sizePop  # SPEA2 characteristic parameter

        def optimize(self, pop, result, routeIdx, subIdx):
            mstats, log = support.statistics(), support.logbook()  # DEAP statistics and logbook
            front = tools.ParetoFront()  # Initialize ParetoFront class
            evals = len(pop)
            archive, gds = [], []  # Initialize offspring and generational distance list
            gen = totalEvals = 0
            while True:
                # Step 3: Environmental selection
                archive = self.tb.select(pop + archive, k=self.sizeArchive)
                prevFront = deepcopy(front)
                front.update(archive)

                # Record statistics
                record = mstats.compile(front)
                log.record(gen=gen, evals=evals, **record)
                print('\r', log.stream)

                totalEvals += evals
                # Step 4: Termination
                if self.termination(prevFront, front, gen, gds, totalEvals):
                    return front, log

                # Step 5: Mating Selection
                matingPool = tools.selTournament(archive, k=self.sizePop, tournsize=2)

                # Step 6: Variation
                pop = algorithms.varAnd(matingPool, self.tb, cxpb=1, mutpb=self.mutpb)

                # Step 2: Fitness assignment of both pop and archive
                archive_ex_pop = [ind for ind in archive if ind not in pop]
                invInds = [ind for ind in pop + archive_ex_pop if not ind.fitness.valid]
                fits = self.tb.map(self.evaluator.evaluate, invInds)
                for ind, fit in zip(invInds, fits):
                    ind.fitness.values = fit

                evals = len(invInds)

                gen += 1

    def compute(self, startEnd, recompute=False, startDate=None, current=False,
                weather=False, algorithm='NSGA2', seed=None, avoidArctic=True, avoidAntarctic=True):
        random.seed(seed)
        print('Parameters:')
        pp.pprint(self.p)
        print('\nCompute input:')
        pp.pprint({'startEnd': startEnd, 'recompute': recompute, 'startDate': startDate, 'current': current,
                   'weather': weather, 'algorithm': algorithm, 'seed': seed, 'avoidArctic': avoidArctic,
                   'avoidAntarctic': avoidAntarctic})
        support.clear_caches()  # Clear caches
        start, end = startEnd
        if start == end:
            return 'equal_start_end'

        for point in startEnd:
            if avoidAntarctic and support.antarctic_circle.contains(Point(point)):
                avoidAntarctic = False
            if avoidArctic and support.arctic_circle.contains(Point(point)):
                avoidArctic = False

        dateString = None if startDate is None else startDate.strftime('%Y%m%d')

        fn = "{}_C{}_W{}_d{}_inclS{}_inclN{}_V{}_T{}_C{}_FP{}_ECA{}".format(startEnd, current, weather,
                                                                            dateString, avoidAntarctic,
                                                                            avoidArctic, self.vessel.name,
                                                                            True,
                                                                            True,
                                                                            self.fuelPrice, self.ecaFactor)
        self.procResultsFP = DIR / "output/processedResults/" / fn

        if not recompute and os.path.exists(self.procResultsFP):
            return None

        newParameters, reinitialize = {}, False
        if self.p['avoidAntarctic'] != avoidAntarctic or self.p['avoidArctic'] != avoidArctic:
            reinitialize = True
            newParameters['avoidAntarctic'] = avoidAntarctic
            newParameters['avoidArctic'] = avoidArctic
            self.update_parameters(newParameters, reinitialize=reinitialize)

        key = tuple(sorted(startEnd))
        # Get initial paths
        initRoutes = None
        for initRouteDict in self.initRoutesList:
            if initRouteDict['startEndKey'] == key and initRouteDict['avoidAntarctic'] == avoidAntarctic and \
                    initRouteDict['avoidArctic'] == avoidArctic:
                initRoutes = initRouteDict['paths']
                break
        if initRoutes is None:
            initRoutes = self.initializer.get_initial_routes(start, end)

            pathOutput = {'avoidAntarctic': avoidAntarctic,
                          'avoidArctic': avoidArctic,
                          'startEndKey': key,
                          'paths': initRoutes}

            self.initRoutesList.append(pathOutput)

            fn = str(uuid.uuid4())
            with open(self.initPathsDir / fn, 'wb') as file:
                pickle.dump(pathOutput, file)

        terminator = self.Terminator(self.p)

        if algorithm == 'MPAES':
            MOEA = self.MPAES(self.tb, self.evaluator, self.p, terminator)
        elif algorithm == 'SPEA2':
            MOEA = self.SPEA2(self.tb, self.evaluator, self.p, terminator)
        else:
            MOEA = self.NSGAII(self.tb, self.evaluator, self.p, terminator)

        result = {'startEnd': startEnd, 'initialRoutes': initRoutes, 'indicators': [None] * len(initRoutes),
                  'logs': [None] * len(initRoutes), 'fronts': [None] * len(initRoutes)}
        for routeIdx, route in enumerate(initRoutes):
            result['indicators'][routeIdx] = [None] * len(route['route'])
            result['logs'][routeIdx] = [None] * len(route['route'])
            result['fronts'][routeIdx] = [None] * len(route['route'])
            print('Computing route {0}/{1}'.format(routeIdx + 1, len(initRoutes)))
            for subIdx, subRoute in enumerate(route['route']):
                print('Computing sub route {0}/{1}'.format(subIdx + 1, len(route['route'])))

                # Reset functions and caches
                self.tb.register("individual", initialization.init_individual, self.tb, subRoute)

                # Step 1: Initialization
                print('Initializing population from shortest path:', end='\n ')
                initialPop = self.tb.population(self.tb.individual, int(MOEA.sizePop / len(subRoute.values())))
                print('done')

                # Step 2: Fitness assignment
                print('Fitness assignment:', end='')
                fits = self.tb.map(self.evaluator.evaluate, initialPop)
                for ind, fit in zip(initialPop, fits):
                    ind.fitness.values = fit
                print('\rFitness assigned')
                self.evaluator.set_classes(current, weather, startDate, self.get_days(initialPop))
                self.tb.register("individual", initialization.init_individual, self.tb, subRoute)

                # Begin the generational process
                front, log = MOEA.optimize(initialPop, result, routeIdx, subIdx)
                self.tb.unregister("individual")

                hypervolume = indicators.hypervolume(front)
                print('hypervolume', hypervolume)
                result['indicators'][routeIdx][subIdx] = {'hypervolume': deepcopy(hypervolume)}
                result['logs'][routeIdx][subIdx] = deepcopy(log)
                result['fronts'][routeIdx][subIdx] = deepcopy(front)

        return result

    def update_parameters(self, newParameters, reinitialize=False):
        self.p = {**self.p, **newParameters}
        self.operators = Operators(self.evaluator.e_feasible, self.vessel, self.geod, self.p)

        if reinitialize:
            # Re-populate R-Tree structures
            navAreaGenerator = NavigableAreaGenerator(self.p, DIR=DIR)
            self.evaluator.landRtree = navAreaGenerator.get_shoreline_rtree()
            self.evaluator.ecaRtree = navAreaGenerator.get_eca_rtree()
            self.initializer = initialization.Initializer(self.evaluator, self.vessel, self.evaluator.landRtree,
                                                          self.evaluator.ecaRtree, self.initializer.bathTree,
                                                          self.geod, self.p, creator.Individual, self.speedIdx, DIR)

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

    def post_process(self, result, updateEvaluator=None, ID=None):
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

        if updateEvaluator is not None:
            current = True if 'current' in updateEvaluator else None
            weather = True if 'weather' in updateEvaluator else None
            startDate = updateEvaluator.get('current')
            startDate = updateEvaluator.get('weather') if startDate is None else startDate

            self.evaluator.set_classes(current, weather, startDate, 10)
            self.evaluator.ecaFactor = updateEvaluator.get('eca', self.evaluator.ecaFactor)

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
    import pprint
    import time

    from case_studies.plot_results import RoutePlotter
    from datetime import datetime
    from scoop import futures
    from support import locations

    startTime = time.time()
    # _startEnd = (locations['Veracruz'], locations['Concepcion'])
    _startEnd = (locations['Floro'], locations['Santander'])
    # _startEnd = ((121.75, 25.15), (-123, 37.75))

    # parameters = {'gen': 200,  # Min number of generations
    #               'n': 100}    # Population size
    startDate = datetime(2017, 9, 4)
    kwargsPlanner = {'inputParameters': {}, 'tb': _tb, 'ecaFactor': 1.0,
                     'criteria': _criteria}
    kwargsCompute = {'startEnd': _startEnd, 'startDate': startDate, 'recompute': True, 'current': False,
                     'weather': False, 'seed': 1, 'algorithm': 'NSGA2'}
    multiprocess = False

    if multiprocess:
        # with mp.Pool() as pool:
        _tb.register("map", futures.map)

        planner = RoutePlanner(**kwargsPlanner)
        rawResults = planner.compute(**kwargsCompute)
    else:
        planner = RoutePlanner(**kwargsPlanner)
        rawResults = planner.compute(**kwargsCompute)

    print("--- %s seconds ---" % (time.time() - startTime))
    procResults, rawResults = planner.post_process(rawResults)
    routePlotter = RoutePlotter(DIR, procResults, rawResults=rawResults, vessel=planner.vessel)
    fig, ax = plt.subplots()
    ax = routePlotter.results(ax, bathymetry=True, weatherDate=None, initial=True, ecas=True,
                              nRoutes=5, colorbar=True)
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(procResults)
    plt.show()
