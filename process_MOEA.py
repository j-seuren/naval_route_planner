import indicators
import os
import pandas as pd
import pickle
import main
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tikzplotlib

from itertools import combinations, permutations
# from datetime import datetime
# from deap import tools
# from geodesic import Geodesic
from pathlib import Path

# planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)
loadDir = Path('C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/5. Thesis/MOEA results/CT_MO')
rawDir = loadDir / 'raws'
os.chdir(loadDir)

files = [file for file in os.listdir(rawDir) if 'v2' not in file]

raw14MC, raw15MC = {}, {}
raw14CM, raw15CM = {}, {}
for file in files:
    split = file.split('_')
    MOEA = split[0]
    with open(rawDir / file, 'rb') as fh:
        rawList = pickle.load(fh)
    if '2014' in file:
        rDict = raw14CM if 'CM' in file else raw14MC
    else:
        rDict = raw15CM if 'CM' in file else raw15MC
    rDict[MOEA] = rawList

# planner.evaluator.set_classes(inclCurr=True, inclWeather=False, nDays=7, startDate=datetime(2014, 11, 15))
# evaluate = planner.evaluator.evaluate

writer = pd.ExcelWriter('output.xlsx')


def compute_metrics(name, rawDict):
    dfTernaryHV = pd.DataFrame(columns=[0, 1, 2])
    dfBinaryHV = pd.DataFrame(columns=list(permutations(range(3), r=2)))
    dfCoverage = pd.DataFrame(columns=list(permutations(range(3), r=2)))

    for i, rawTup in enumerate(zip(rawDict['SPEA2'], rawDict['NSGA2'], rawDict['MPAES'])):
        print('\r', i, end='')
        fronts = [rawL['fronts'][0][0] for rawL in rawTup]

        ternaryRow = {}
        for A_idx in range(3):
            f2, f3 = np.delete(fronts, A_idx)
            trHV = indicators.triple_hypervolume(fronts[A_idx], f2, f3)
            ternaryRow[A_idx] = trHV

        dfTernaryHV = dfTernaryHV.append(ternaryRow, ignore_index=True)

        perms = permutations(range(3), r=2)
        coverageRow, binaryHVRow = {}, {}
        for A_idx, B_idx in perms:
            f1, f2 = fronts[A_idx], fronts[B_idx]
            coverageRow[(A_idx, B_idx)] = indicators.two_sets_coverage(f1, f2)
            binaryHVRow[(A_idx, B_idx)] = indicators.binary_hypervolume(f1, f2)
        dfBinaryHV = dfBinaryHV.append(binaryHVRow, ignore_index=True)
        dfCoverage = dfCoverage.append(coverageRow, ignore_index=True)

    axT = dfTernaryHV.boxplot(return_type='axes')
    plt.savefig('{}_ternary.png'.format(name))
    plt.clf()
    axB = dfBinaryHV.boxplot(return_type='axes')
    plt.savefig('{}_binary.png'.format(name))
    plt.clf()
    axC = dfCoverage.boxplot(return_type='axes')
    plt.savefig('{}_coverage.png'.format(name))
    plt.clf()
    for ID, ax in zip(['T', 'B', 'C'], [axT, axB, axC]):
        with open('{}_{}ax'.format(name, ID), 'wb') as fh:
            pickle.dump(ax, fh)

    for df in [dfTernaryHV, dfBinaryHV, dfCoverage]:
        mean, std, min, max, Q1, Q3 = df.mean(), df.std(), df.min(), df.max(), df.quantile(0.25), df.quantile(0.75)
        df.loc['min'] = min
        df.loc['Q1'] = Q1
        df.loc['mean'] = mean
        df.loc['Q3'] = Q3
        df.loc['max'] = max
        df.loc['std'] = std

    dfCoverage.to_excel(writer, sheet_name='{}_C'.format(name))
    dfBinaryHV.to_excel(writer, sheet_name='{}_B'.format(name))
    dfTernaryHV.to_excel(writer, sheet_name='{}_T'.format(name))


for key, rDict in {'14CM': raw14CM}.items():
    print('\r', key, end='\n')
    compute_metrics(key, rDict)


writer.close()
