import indicators
import os
import pickle
import main
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime
from deap import tools
from matplotlib import font_manager as fm
from pathlib import Path

fontPropFP = "C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/tex-gyre-pagella.regular.otf"
fontProp = fm.FontProperties(fname=fontPropFP)


def create_raw_dicts():
    planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)

    loadDir = Path('D:/output/current/Gulf')
    rawsDir = loadDir / 'raws'
    varDir = rawsDir / 'var'
    maxDir = rawsDir / 'max'
    minDir = rawsDir / 'min'
    os.chdir(loadDir)

    refFilesVar = [file for file in os.listdir(varDir) if 'R' in file]
    refFilesMax = [file for file in os.listdir(maxDir) if 'R' in file]
    refFilesMin = [file for file in os.listdir(minDir) if 'R' in file]
    print('refFiles', '\n', 'var', refFilesVar, '\n', 'max', refFilesMax, '\n', 'min', refFilesMin)
    refFrontsDict = {'2014': {}, '2015': {}}
    for d, date in enumerate([datetime(2014, 11, 15), datetime(2015, 5, 15)]):
        dateKey = '2014' if d == 0 else '2015'

        # Initialize planner
        planner.evaluator.set_classes(inclCurr=True, inclWeather=False, nDays=7, startDate=date)
        evaluate = planner.evaluator.evaluate

        for speed, refFiles in {'var': refFilesVar, 'max': refFilesMax, 'min': refFilesMin}.items():
            refFrontsDict[dateKey][speed] = {}
            for refFile in refFilesVar:
                split = refFile.split('_')
                pair = split[-1]
                with open(rawsDir / speed / refFile, 'rb') as fh:
                    refRawList = pickle.load(fh)
                refFronts = [refRaw['fronts'][0][0] for refRaw in refRawList]
                newRefFronts = []
                for oldFront in refFronts:
                    fits = [evaluate(ind, revert=False, includePenalty=False) for ind in oldFront]
                    for fit, ind in zip(fits, oldFront.items):
                        ind.fitness.values = fit
                    newFront = tools.ParetoFront()
                    newFront.update(oldFront.items)
                    newRefFronts.append(newFront)
                refFrontsDict[dateKey][speed][pair] = newRefFronts

    filesVar = [file for file in os.listdir(varDir) if 'R' not in file]
    filesMax = [file for file in os.listdir(maxDir) if 'R' not in file]
    filesMin = [file for file in os.listdir(minDir) if 'R' not in file]

    frontsDict = {'2014': {}, '2015': {}}
    for speed, files in {'var': filesVar, 'max': filesMax, 'min': filesMin}.items():
        frontsDict['2014'][speed] = {}
        frontsDict['2015'][speed] = {}
        for file in files:
            split = file.split('_')
            pair = split[-1]
            with open(rawsDir / file, 'rb') as fh:
                rawList = pickle.load(fh)
            fronts = [raw['fronts'][0][0] for raw in rawList]
            if '2014' in file:
                frontsDict['2014'][speed][pair] = (fronts, refFrontsDict['2015'][speed][pair])
            else:
                frontsDict['2015'][speed][pair] = (fronts, refFrontsDict['2015'][speed][pair])

    return frontsDict, planner


def metrics_plot(frontsDict, planner, plot=False, savePlot=False):
    years = list(frontsDict.keys())
    for year in years:
        writer = pd.ExcelWriter('output_{}_metrics.xlsx'.format(year))
        speeds = frontsDict[year].keys()
        for speed in speeds:
            pairs = frontsDict[year][speed].keys()

            # Compute Pareto metrics for variable speed only
            if speed == 'var':
                dfBinaryHV = pd.DataFrame(columns=pairs, index=range(5))
                dfCoverage = pd.DataFrame(columns=pairs, index=range(5))

                for pair in pairs:
                    print('\r', pair, end='')
                    fronts, refFronts = frontsDict[year][speed][pair]

                    minFronts, minRefFronts = frontsDict[year]['min'][pair]
                    maxFronts, maxRefFronts = frontsDict[year]['max'][pair]

                    for run, (front, refFront) in enumerate(zip(fronts, refFronts)):
                        biHV = indicators.binary_hypervolume(front, refFront)
                        coverage = indicators.two_sets_coverage(front, refFront)
                        dfBinaryHV.iloc[run][pair] = biHV
                        dfCoverage.iloc[run][pair] = coverage

                        minFrontSpeed = minFronts[run]
                        minFrontSpeedRef = minRefFronts[run]
                        maxFrontSpeed = minFronts[run]
                        maxFrontSpeedRef = minRefFronts[run]
                        speedFronts = [(minFrontSpeed, minFrontSpeedRef), (maxFrontSpeed,maxFrontSpeedRef )]

                        if plot:
                            fig = plot_front(front, refFront, speedFronts, planner)
                            plt.show()
                            if savePlot:
                                fig.savefig('{}_frontM_{}'.format(pair, run), dpi=300)
                                fig.savefig('{}_frontM_{}.pdf'.format(pair, run), bbox_inches='tight', pad_inches=.01)
                                # tikzplotlib.save("{}_frontM_{}.tex".format(pair, run))

                for df in [dfBinaryHV, dfCoverage]:
                    mean, std, minn, maxx = df.mean(), df.std(), df.min(), df.max()
                    df.loc['mean'] = mean
                    df.loc['std'] = std
                    df.loc['min'] = minn
                    df.loc['max'] = maxx

                dfCoverage.to_excel(writer, sheet_name='{}_C'.format(year))
                dfBinaryHV.to_excel(writer, sheet_name='{}_B'.format(year))

        writer.close()


def plot_front(front, refFront, speedFronts, planner):
    evaluate2 = planner.evaluator.evaluate2

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 16))
    ax2.set_xlabel('Travel time [h]', fontproperties=fontProp)
    ax2.set_ylabel('Fuel consumption [t]', fontproperties=fontProp)
    ax1.set_ylabel('Average speed increase [knots]', fontproperties=fontProp)
    cycleFront = ax2._get_lines.prop_cycler
    labels = ['Incl. current', 'Reference']

    # Do for front and ref front
    for idx, f in enumerate([front, refFront]):
        # (Objective) values
        days, cost, dist, _, avgSpeed = zip(*map(evaluate2, f.items))
        hours = np.array(days) * 24.
        currentSpeed = np.array(dist) / np.array(hours) - np.array(avgSpeed)
        color = next(cycleFront)['color']

        # Plot front
        marker, s, zorder = 'o', 1, 2
        ax2.scatter(days, cost, color=color, marker=marker, s=s, label=labels[idx], zorder=zorder)
        ax1.scatter(days, currentSpeed, color=color, marker=marker, s=s, zorder=zorder)

    # Plot constant speeds
    colors = [next(cycleFront)['color'], next(cycleFront)['color']]

    for spdTup in speedFronts:
        for spdIdx, spd in enumerate(spdTup):
            days, cost, dist, _, avgSpeed = zip(*map(evaluate2, spd.items))
            hours = np.array(days) * 24.
            currentSpeed = np.array(dist) / np.array(hours) - np.array(avgSpeed)

            marker, s, zorder = 's', 5, 3
            ax2.scatter(days, cost, color=colors[spdIdx], marker=marker, s=s, label=labels[spdIdx], zorder=zorder)
            ax1.scatter(days, currentSpeed, color=colors[spdIdx], marker=marker, s=s, zorder=zorder)

    ax2.legend(prop=fontProp)
    ax2.grid()
    ax1.grid()
    plt.xticks(fontproperties=fontProp)
    plt.yticks(fontproperties=fontProp)

    return fig


_frontsDict, _planner = create_raw_dicts()

metrics_plot(_frontsDict, _planner, plot=True, savePlot=False)
