import indicators
import os
import pandas as pd
import main
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tikzplotlib

from itertools import permutations
from datetime import datetime
from matplotlib import font_manager as fm
from pathlib import Path

fontPropFP = "C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/tex-gyre-pagella.regular.otf"
fontProp = fm.FontProperties(fname=fontPropFP)


def create_raw_dicts():
    loadDir = Path('C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/5. Thesis/MOEA results/WTH')
    rawDir = loadDir / 'raws'
    os.chdir(loadDir)

    files = [file for file in os.listdir(rawDir)]

    frontsDict = {}
    for file in files:
        split = file.split('_')
        MOEA = split[0]
        with open(rawDir / file, 'rb') as fh:
            rawList = pickle.load(fh)
        frontsDict[MOEA] = [raw['fronts'][0][0] for raw in rawList]

    planner = main.RoutePlanner(bathymetry=False, ecaFactor=1.5593)
    planner.evaluator.set_classes(inclCurr=False, inclWeather=True, nDays=30, startDate=datetime(2013, 9, 24, 12))

    return frontsDict, planner


def compute_metrics(name, frontsDict):
    writer = pd.ExcelWriter('output_{}_metrics.xlsx'.format(name))
    columns = ['M', 'N', 'S']
    colPermutations = [(columns[perm[0]], columns[perm[1]]) for perm in permutations(range(3), r=2)]

    dfTernaryHV = pd.DataFrame(columns=columns)
    dfBinaryHV = pd.DataFrame(columns=colPermutations)
    dfCoverage = pd.DataFrame(columns=colPermutations)

    for run, fronts in enumerate(zip(frontsDict['MPAES'], frontsDict['NSGA2'], frontsDict['SPEA2'])):
        print('\r', run, end='')

        ternaryRow = {}
        for A_idx in range(3):
            f2, f3 = np.delete(fronts, A_idx)
            trHV = indicators.triple_hypervolume(fronts[A_idx], f2, f3)
            ternaryRow[columns[A_idx]] = trHV

        dfTernaryHV = dfTernaryHV.append(ternaryRow, ignore_index=True)

        coverageRow, binaryHVRow = {}, {}
        for A, B in colPermutations:
            A_idx, B_idx = columns.index(A), columns.index(B)

            f1, f2 = fronts[A_idx], fronts[B_idx]
            coverageRow[(A, B)] = indicators.two_sets_coverage(f1, f2)
            binaryHVRow[(A, B)] = indicators.binary_hypervolume(f1, f2)
        dfBinaryHV = dfBinaryHV.append(binaryHVRow, ignore_index=True)
        dfCoverage = dfCoverage.append(coverageRow, ignore_index=True)

    for ID, df in zip(['T', 'B', 'C'], [dfTernaryHV, dfBinaryHV, dfCoverage]):
        ax = df.boxplot(return_type='axes')
        if ID == 'T':
            label = 'Ternary hypervolume'
        else:
            label = 'Binary hypervolume' if ID == 'B' else 'Two sets coverage'
        ax.set_ylabel(label, fontproperties=fontProp)
        ax.xticks(fontproperties=fontProp)
        with open('{}_{}ax'.format(name, ID), 'wb') as fh:
            pickle.dump(ax, fh)
        plt.savefig('{}_{}.png'.format(name, ID))
        plt.clf()

        mean, std, _min, _max, Q1, Q3 = df.mean(), df.std(), df.min(), df.max(), df.quantile(0.25), df.quantile(0.75)
        df.loc['min'] = _min
        df.loc['Q1'] = Q1
        df.loc['mean'] = mean
        df.loc['Q3'] = Q3
        df.loc['max'] = _max
        df.loc['std'] = std

        df.to_excel(writer, sheet_name='{}_{}'.format(name, ID))

    writer.close()


rDict, _planner = create_raw_dicts()
compute_metrics('WTH', rDict)
