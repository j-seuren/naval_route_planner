import indicators
import os
import pickle
import main
# import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from deap import tools
from geodesic import Geodesic

planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)


loadDir = 'C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/5. Thesis/MOEA results/'
rawDir = loadDir + 'raws/'
os.chdir(rawDir)

westLocations = ['W1', 'W2', 'W3']
eastLocations = ['E1', 'E2', 'E3']

raw14KT, raw15KT = {}, {}
raw14TK, raw15TK = {}, {}
for file in os.listdir():
    split = file.split('_')
    MOEA = split[0]
    with open(rawDir + file, 'rb') as fh:
        rawList = pickle.load(fh)
    if '2014' in file:
        rDict = raw14TK if 'TK' in file else raw14KT
    else:
        rDict = raw15TK if 'TK' in file else raw15KT
    rDict[MOEA] = rawList

# planner.evaluator.set_classes(inclCurr=True, inclWeather=False, nDays=7, startDate=datetime(2014, 11, 15))
# evaluate = planner.evaluator.evaluate

bestHV = [0, 0, 0]
def compute_triple_hypervolumes(rawDict):
    # 'SPEA2', 'NSGA2', 'MPAES'
    SP, NS, MP = [], [], []
    for rawSP, rawNS, rawMP in zip(rawDict['SPEA2'], rawDict['NSGA2'], rawDict['MPAES']):
        frontSP = rawSP['fronts'][0][0]
        frontNS = rawNS['fronts'][0][0]
        frontMP = rawMP['fronts'][0][0]
        hvSP = indicators.triple_hypervolume(frontSP, frontNS, frontMP)
        hvNS = indicators.triple_hypervolume(frontNS, frontMP, frontSP)
        hvMP = indicators.triple_hypervolume(frontMP, frontSP, frontNS)
        SP.append(hvSP)
        NS.append(hvNS)
        MP.append(hvMP)
        hvList = [hvSP, hvNS, hvMP]
        bestIdx = np.argmax(hvList)
        print(hvList, bestIdx)
        bestHV[bestIdx] += 1

    print('SPEA2', np.average(SP), np.std(SP))
    print('NSGA2', np.average(NS), np.std(NS))
    print('MPAES', np.average(MP), np.std(MP))
    print(bestHV)


for rDict in [raw14TK, raw14KT, raw15KT, raw15TK]:
    compute_triple_hypervolumes(rDict)


