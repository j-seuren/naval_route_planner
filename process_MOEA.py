import indicators
import os
import pandas as pd
import main
import matplotlib.pyplot as plt
import numpy as np
import pickle
# import tikzplotlib

from itertools import permutations
from datetime import datetime
from matplotlib import font_manager as fm
from pathlib import Path


NAME = 'BLANK'

if NAME == 'BLANK':
    date = None
    weather = False
    current = False
    directory = 'Blank'
elif NAME == 'WTH':
    date = datetime(2013, 9, 24, 12)
    weather = True
    current = False
    directory = 'Weather'
else:
    date = datetime(2014, 11, 25)
    weather = False
    current = True
    directory = 'Current'

fontPropFP = "C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/tex-gyre-pagella.regular.otf"
fontProp = fm.FontProperties(fname=fontPropFP, size=9)

loadDir = Path('D:/output/MOEA/{}'.format(directory))
rawDir = loadDir / 'raws'
os.chdir(loadDir)


def get_fronts_dict():
    # files = [file for file in os.listdir(rawDir) if 'PH' in file and 'H_2' not in file]
    files = [file for file in os.listdir(rawDir)]
    print(files)

    frontsDict = {}
    for file in files:
        split = file.split('_')
        MOEA = split[0]
        with open(rawDir / file, 'rb') as fh:
            rawList = pickle.load(fh)
        frontsDict[MOEA] = [raw['fronts'][0][0] for raw in rawList]

    return frontsDict


def compute_metrics(frontsDict):
    writer = pd.ExcelWriter('output_{}_metrics.xlsx'.format(NAME))
    columns = ['M', 'N', 'S']
    colPermutations = [(columns[perm[0]], columns[perm[1]]) for perm in permutations(range(3), r=2)]

    dfTernaryHV = pd.DataFrame(columns=columns)
    dfBinaryHV = pd.DataFrame(columns=colPermutations)
    dfCoverage = pd.DataFrame(columns=colPermutations)
    tupleStrings = [('M', 'N'), ('M', 'S'), ('N', 'M'), ('N', 'S'), ('S', 'M'), ('S', 'N')]

    for runM, frontM in enumerate(frontsDict['MPAES'][:-3]):
        for runN, frontN in enumerate(frontsDict['NSGA2'][:-3]):
            for runS, frontS in enumerate(frontsDict['SPEA2'][runN:runN+3]):
                print('\r', runM, runN, runS, end='')
                fronts = (frontM, frontN, frontS)

                # [tHV_A, tHV_B, tHV_C], [bHV_AB, bHV_AC, bHV_BA, bHV_BC, bHV_CA, bHV_CB]
                ternary, binary = indicators.binary_ternary_hv_ratio(frontM, frontN, frontS)

                ternaryRow = {}
                for A_idx in range(3):
                    ternaryRow[columns[A_idx]] = ternary[A_idx]

                dfTernaryHV = dfTernaryHV.append(ternaryRow, ignore_index=True)

                binaryHVRow = {}
                for idx, tupleString in enumerate(tupleStrings):
                    binaryHVRow[tupleString] = binary[idx]
                dfBinaryHV = dfBinaryHV.append(binaryHVRow, ignore_index=True)

                coverageRow = {}
                for A, B in colPermutations:
                    A_idx, B_idx = columns.index(A), columns.index(B)
                    f1, f2 = fronts[A_idx], fronts[B_idx]
                    coverageRow[(A, B)] = indicators.two_sets_coverage(f1, f2)
                dfCoverage = dfCoverage.append(coverageRow, ignore_index=True)

    for ID, df in zip(['T', 'B', 'C'], [dfTernaryHV, dfBinaryHV, dfCoverage]):
        ax = df.boxplot(return_type='axes')
        if ID == 'T':
            label = 'Ternary hypervolume ratio'
        else:
            label = 'Binary hypervolume ratio' if ID == 'B' else 'Two sets coverage ratio'
        ax.set_ylabel(label, fontproperties=fontProp)
        fig = plt.gcf()
        fig.set_size_inches(2.7, 2.7)
        plt.yticks(fontproperties=fontProp)
        plt.xticks(fontproperties=fontProp)
        with open('box_plots/{}_{}ax'.format(NAME, ID), 'wb') as fh:
            pickle.dump(ax, fh)
        plt.savefig('box_plots/{}_{}.png'.format(NAME, ID))
        plt.savefig('box_plots/{}_{}.pdf'.format(NAME, ID), bbox_inches='tight', pad_inches=.02)
        plt.clf()

        mean, std, _min, _max, Q1, Q3 = df.mean(), df.std(), df.min(), df.max(), df.quantile(0.25), df.quantile(0.75)
        df.loc['min'] = _min
        df.loc['Q1'] = Q1
        df.loc['mean'] = mean
        df.loc['Q3'] = Q3
        df.loc['max'] = _max
        df.loc['std'] = std

        df.to_excel(writer, sheet_name='{}_{}'.format(NAME, ID))

    writer.close()


def load_axes():
    for i, ID in enumerate(['T', 'B', 'C']):
        with open('box_plots/{}_{}ax'.format(NAME, ID), 'rb') as fh:
            pickle.load(fh)
        fig = plt.gcf()
        fig.set_size_inches(2.7, 2.7)
        plt.yticks(fontproperties=fontProp)
        plt.xticks(fontproperties=fontProp)
        fp = 'box_plots/{}_{}.pdf'.format(NAME, ID)
        print('saved to', fp)
        plt.savefig(fp, bbox_inches='tight', pad_inches=.02)
        plt.close('all')
    # tikzplotlib.save('{}.tex'.format(name))


def plot_fronts(frontsDict, savePlots=True):
    labels = ['M-PAES', 'NSGA-II', 'SPEA2']
    for run, fronts in enumerate(zip(frontsDict['MPAES'], frontsDict['NSGA2'], frontsDict['SPEA2'])):
        fig, ax = plt.subplots()
        ax.set_xlabel('Travel time [d]', fontproperties=fontProp)
        ax.set_ylabel(r'Fuel cost [USD, $\times 1000$]', fontproperties=fontProp)
        # noinspection PyProtectedMember
        cycleFront = ax._get_lines.prop_cycler

        for f, front in enumerate(fronts):
            fits = [fit.values for fit in front.keys]
            days, cost = zip(*fits)
            color = next(cycleFront)['color']

            # Plot front
            marker, s = 'o', 1
            ax.scatter(days, cost, color=color, marker=marker, s=s, label=labels[f])

        ax.legend(prop=fontProp)
        plt.xticks(fontproperties=fontProp)
        plt.yticks(fontproperties=fontProp)

        # plt.show()
        if savePlots:
            fp = 'front_plots/{}_frontM_{}'.format(NAME, run)
            fig.savefig(fp, dpi=300)
            fig.savefig(fp + '.pdf', bbox_inches='tight', pad_inches=.01)
            print('saved to', fp)
            # tikzplotlib.save("{}_frontM_{}.tex".format(pair, run))
        plt.close('all')

import time

t0 = time.time()
compute_metrics(get_fronts_dict())
print(time.time() - t0)
# plot_fronts(get_fronts_dict())

def plot_comp_times():
    csvLoadDir = Path('D:/output/MOEA')
    fp = csvLoadDir / 'Computation_times.xlsx'
    os.chdir(csvLoadDir)
    sheetNames = ['BLANK', 'CURRENT', 'WEATHER']
    for sheetName in sheetNames:
        df = pd.read_excel(fp, sheet_name=sheetName)
        ax = df.boxplot(return_type='axes')
        ax.set_ylabel('Computation time [s]', fontproperties=fontProp)
        fig = plt.gcf()
        fig.set_size_inches(2.7, 2.7)
        plt.xticks(fontproperties=fontProp)
        plt.yticks(fontproperties=fontProp)
        plt.savefig(sheetName + '_CT.pdf', bbox_inches='tight', pad_inches=.01)
        plt.clf()


# plot_comp_times()
