import indicators
import os
import pickle
import main
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from deap import tools
from geodesic import Geodesic

planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)


gulfDir = 'C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/5. Thesis/Current results/Gulf/download_6-10/'
gulfDir = gulfDir + 'var/'
os.chdir(gulfDir)

rawDict = {}
refFiles = [file for file in os.listdir() if 'R' in file]
references = {}
for refFile in refFiles:
    split = refFile.split('_')
    pair = split[-1]
    with open(gulfDir + refFile, 'rb') as fh:
        ref = pickle.load(fh)
    references[pair] = ref

westLocations = ['W1', 'W2', 'W3']
eastLocations = ['E1', 'E2', 'E3']

raws14, raws15 = {}, {}
for file in os.listdir():
    if file not in refFiles:
        continue
    split = file.split('_')
    pair = split[-1]
    with open(gulfDir + file, 'rb') as fh:
        rawList = pickle.load(fh)
    if '2014' in file:
        raws14[pair] = (rawList, references[pair])
    else:
        raws15[pair] = (rawList, references[pair])

totDistance = Geodesic().total_distance
planner.evaluator.set_classes(inclCurr=True, inclWeather=False, nDays=7, startDate=datetime(2014, 11, 15))
evaluate = planner.evaluator.evaluate
for pair, tup in raws14.items():
    binaryHypervolumes, twoSetCoverage = [], []
    rawList, refList = tup
    for raw, ref in zip(rawList, refList):
        front = raw['fronts'][0][0]
        front0 = ref['fronts'][0][0]
        for fr in [front0, front]:
            fits = [evaluate(ind, revert=False, includePenalty=False) for ind in fr]
            for fit, ind in zip(fits, fr.items):
                ind.fitness.values = fit
        newFront = tools.ParetoFront()
        newFront.update(front0.items)

        # fig, ax = plt.subplots()

        # vals = np.array([[ind.fitness.values[0], ind.fitness.values[1]] for ind in front])
        # vals0 = np.array([[ind.fitness.values[0], ind.fitness.values[1]] for ind in newFront])

        # ax.scatter(vals[:, 0], vals[:, 1], label=pair, s=1)
        # ax.scatter(vals0[:, 0], vals0[:, 1], label=pair + ' - ref', s=1)
        # ax.legend()
        # plt.show()

        binaryHypervolumes.append(indicators.binary_hypervolume(front.items, newFront.items))
        twoSetCoverage.append(indicators.two_sets_coverage(front.items, newFront.items))
    print(pair, np.average(binaryHypervolumes))
    print(pair, np.average(twoSetCoverage))


