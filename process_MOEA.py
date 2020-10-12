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


NAME = 'WTH'

if NAME == 'BLANK':
    date = None
    weather = False
    current = False
elif NAME == 'WTH':
    date = datetime(2013, 9, 24, 12)
    weather = True
    current = False
else:
    date = datetime(2014, 11, 25)
    weather = False
    current = True

fontPropFP = "C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/tex-gyre-pagella.regular.otf"
fontProp = fm.FontProperties(fname=fontPropFP)

loadDir = Path('D:/output/MOEA/Weather')
rawDir = loadDir / 'raws'
os.chdir(loadDir)


def get_fronts_dict():
    files = [file for file in os.listdir(rawDir)]

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

    for run, fronts in enumerate(zip(frontsDict['MPAES'], frontsDict['NSGA2'], frontsDict['SPEA2'])):
        print('\r', run, end='')

        ternaryRow = {}
        for A_idx in range(3):
            f2, f3 = np.delete(fronts, A_idx)
            trHV = indicators.triple_hypervolume_ratio(fronts[A_idx], f2, f3)
            ternaryRow[columns[A_idx]] = trHV

        dfTernaryHV = dfTernaryHV.append(ternaryRow, ignore_index=True)

        coverageRow, binaryHVRow = {}, {}
        for A, B in colPermutations:
            A_idx, B_idx = columns.index(A), columns.index(B)

            f1, f2 = fronts[A_idx], fronts[B_idx]
            coverageRow[(A, B)] = indicators.two_sets_coverage(f1, f2)
            binaryHVRow[(A, B)] = indicators.binary_hypervolume_ratio(f1, f2)
        dfBinaryHV = dfBinaryHV.append(binaryHVRow, ignore_index=True)
        dfCoverage = dfCoverage.append(coverageRow, ignore_index=True)

    for ID, df in zip(['T', 'B', 'C'], [dfTernaryHV, dfBinaryHV, dfCoverage]):
        ax = df.boxplot(return_type='axes')
        if ID == 'T':
            label = 'Ternary hypervolume ratio'
        else:
            label = 'Binary hypervolume ratio' if ID == 'B' else 'Two sets coverage ratio'
        ax.set_ylabel(label, fontproperties=fontProp)
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
        plt.savefig('box_plots/{}_{}.pdf'.format(NAME, ID), bbox_inches='tight', pad_inches=.02)
        plt.close('all')
    # tikzplotlib.save('{}.tex'.format(name))


def plot_fronts(frontsDict, savePlots=True):
    labels = ['M-PAES', 'NSGA-II', 'SPEA2']
    for run, fronts in enumerate(zip(frontsDict['MPAES'], frontsDict['NSGA2'], frontsDict['SPEA2'])):
        fig, ax = plt.subplots()
        ax.set_xlabel('Travel time [d]', fontproperties=fontProp)
        ax.set_ylabel(r'Fuel cost [$\times 10^3$ USD]', fontproperties=fontProp)
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

        # plt.show()
        if savePlots:
            fig.savefig('front_plots/{}_frontM_{}'.format(NAME, run), dpi=300)
            fig.savefig('front_plots/{}_frontM_{}.pdf'.format(NAME, run),
                        bbox_inches='tight', pad_inches=.01)
            # tikzplotlib.save("{}_frontM_{}.tex".format(pair, run))
        plt.close('all')


compute_metrics(get_fronts_dict())
