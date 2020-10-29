import indicators
import os
import pickle
import main
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np
import pandas as pd
import tikzplotlib

from data_config.current_data import CurrentDataRetriever
from datetime import datetime
from deap import tools
from matplotlib import font_manager as fm
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
from pathlib import Path

fontPropFP = "C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/tex-gyre-pagella.regular.otf"
fontProp = fm.FontProperties(fname=fontPropFP, size=9)

loadDir = Path('D:/output/current_13_10/KC/NSGA2_varSP_BFalse_ECA1.0/5/')
rawDir = loadDir / 'raw'
os.chdir(loadDir)


def get_fronts_dict():
    frontsDictFP = loadDir / 'frontsDict'
    planner = main.RoutePlanner(bathymetry=False, fuelPrice=1., ecaFactor=1, vesselName='Tanaka')
    planner.evaluator.set_classes(inclCurr=True, inclWeather=False, nDays=7, startDate=datetime(2014, 11, 15))
    evaluate = planner.evaluator.evaluate

    if os.path.exists(frontsDictFP):
        with open(frontsDictFP, 'rb') as fh:
            frontsDict = pickle.load(fh)
            print('loaded', frontsDictFP)
        return frontsDict, planner

    refFiles = [file for file in os.listdir(rawDir) if 'R' in file]
    print('refFiles', refFiles)

    refFrontsDict = {}
    for refFile in refFiles:
        split = refFile.split('_')
        pair = split[-1]
        with open(rawDir / refFile, 'rb') as fh:
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
        refFrontsDict[pair] = newRefFronts

    files = [file for file in os.listdir(rawDir) if 'R' not in file]
    print('files', files)

    frontsDict = {}
    for file in files:
        split = file.split('_')
        pair = split[-1]
        with open(rawDir / file, 'rb') as fh:
            rawList = pickle.load(fh)
        fronts = [raw['fronts'][0][0] for raw in rawList]
        frontsDict[pair] = (fronts, refFrontsDict[pair])

    with open(frontsDictFP, 'wb') as fh:
        pickle.dump(frontsDict, fh)

    return frontsDict, planner


def compute_metrics(frontsDict):
    writer = pd.ExcelWriter('output_metrics.xlsx')
    pairs = list(frontsDict.keys())
    dfBinaryHV = pd.DataFrame(columns=pairs)
    dfCoverage = pd.DataFrame(columns=pairs)

    for pair, (fronts, refFronts) in frontsDict.items():
        print('\r', pair, end='')

        for front, refFront in zip(fronts, refFronts):
            biHV = indicators.binary_hypervolume_ratio(front, refFront)
            coverage = indicators.two_sets_coverage(front, refFront)
            dfBinaryHV = dfBinaryHV.append({pair: biHV}, ignore_index=True)
            dfCoverage = dfCoverage.append({pair: coverage}, ignore_index=True)

    for df in [dfBinaryHV, dfCoverage]:
        mean, std, minn, maxx = df.mean(), df.std(), df.min(), df.max()
        df.loc['mean'] = mean
        df.loc['std'] = std
        df.loc['min'] = minn
        df.loc['max'] = maxx

    dfCoverage.to_excel(writer, sheet_name='C')
    dfBinaryHV.to_excel(writer, sheet_name='B')
    writer.close()


def save_fronts(frontsDict, planner):

    evaluate2 = planner.evaluator.evaluate2
    for pair, frontTup in frontsDict.items():
        writer = pd.ExcelWriter('output_{}_fronts.xlsx'.format(pair))
        print('\r', pair, end='')
        fronts, refFronts = frontTup
        dfPairList, refDFPairList = [], []
        for run, (front, refFront) in enumerate(zip(fronts, refFronts)):
            dataFrames = []
            # Do for front and ref front
            for idx, f in enumerate([front, refFront]):
                # (Objective) values
                days, cost, dist, _, avgSpeed = zip(*map(evaluate2, f.items))
                cost = np.array(cost) * 1000.
                hours = np.array(days) * 24.
                currentSpeed = np.array(dist) / np.array(hours) - np.array(avgSpeed)
                df0 = pd.DataFrame(np.transpose(np.stack([hours, cost, dist, avgSpeed, currentSpeed])),
                                   columns=['T', 'C', 'D', 'V', 'S'])

                # Statistics
                dfStat = pd.DataFrame([df0.mean(), df0.std(), df0.min(), df0.max()],
                                      index=['mean', 'std', 'min', 'max'],
                                      columns=['T', 'C', 'D', 'V', 'S'])
                dfStatPairList = dfPairList if idx == 0 else refDFPairList
                dfStatPairList.append(dfStat)

                # Append dataframes
                df0 = dfStat.append(df0, ignore_index=False)
                dataFrames.append(df0)

            # Write to Excel sheet
            dfFronts, dfRefFronts = dataFrames
            dfRefFronts.to_excel(writer, sheet_name='{}_R{}'.format(pair, run))
            dfFronts.to_excel(writer, sheet_name='{}_{}'.format(pair, run))

        dfPair = pd.concat(dfPairList).groupby(level=0).mean()
        dfRefPair = pd.concat(refDFPairList).groupby(level=0).mean()
        dfPair.to_excel(writer, sheet_name='S_{}'.format(pair))
        dfRefPair.to_excel(writer, sheet_name='S_{}_R'.format(pair))
        writer.close()


def plot_fronts(frontsDict, planner, save=False):
    evaluate2 = planner.evaluator.evaluate2
    for pair, frontTup in frontsDict.items():
        print('\r', pair, end='')
        fronts, refFronts = frontTup
        for run, (front, refFront) in enumerate(zip(fronts, refFronts)):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 16))
            ax2.set_xlabel('Travel time [h]', fontproperties=fontProp)
            ax2.set_ylabel('Fuel consumption [t]', fontproperties=fontProp)
            ax1.set_ylabel('Average speed increase [kn]', fontproperties=fontProp)
            # noinspection PyProtectedMember
            cycleFront = ax2._get_lines.prop_cycler
            labels = ['Incl. current', 'Reference']

            # Do for front and ref front
            for idx, f in enumerate([front, refFront]):
                # (Objective) values
                days, cost, dist, _, avgSpeed = zip(*map(evaluate2, f.items))
                cost = np.array(cost) * 1000.
                hours = np.array(days) * 24.
                currentSpeed = np.array(dist) / np.array(hours) - np.array(avgSpeed)
                color = next(cycleFront)['color']
                # Plot front
                marker, s = 'o', 1
                ax2.scatter(hours, cost, color=color, marker=marker, s=s, label=labels[idx])
                ax1.scatter(hours, currentSpeed, color=color, marker=marker, s=s)

            ax2.legend(prop=fontProp)
            ax2.grid()
            ax1.grid()
            plt.xticks(fontproperties=fontProp)
            plt.yticks(fontproperties=fontProp)

            if save:
                fig.savefig('{}_frontM_{}'.format(pair, run), dpi=300)
                fig.savefig('{}_frontM_{}.pdf'.format(pair, run), bbox_inches='tight', pad_inches=.01)
                tikzplotlib.save("{}_frontM_{}.tex".format(pair, run), encoding='utf-8')


def navigation_area(ax, uin, vin, lons, lats):
    extent = (120, 25, 142, 37)
    left, bottom, right, top = extent
    m = Basemap(projection='merc', resolution='l', ax=ax,
                llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left, urcrnrlon=right)
    m.drawmapboundary()
    m.fillcontinents(zorder=2)
    m.drawcoastlines()
    m.drawparallels(np.arange(24., 38, 2.), color='#7f7f7f', linewidth=0.5, labels=[1, 0, 0, 0], fontproperties=fontProp)
    m.drawmeridians(np.arange(120., 144, 2.), color='#7f7f7f', linewidth=0.5, labels=[0, 0, 0, 1], fontproperties=fontProp)

    # Currents
    dLon = extent[2] - extent[0]
    dLat = extent[3] - extent[1]
    vLon = int(dLon * 4)
    vLat = int(dLat * 4)
    uRot, vRot, x, y = m.transform_vector(uin, vin, lons, lats, vLon, vLat, returnxy=True)
    m.quiver(x, y, uRot, vRot, np.hypot(uRot, vRot),
             pivot='mid', width=0.002, headlength=4, cmap='Blues', scale=90, ax=ax)

    lon = 123
    lat = 26

    x, y = m(lon, lat)

    plt.text(x, y, 'K', fontproperties=fontProp, fontweight='bold', fontsize=12, ha='right', va='top', color='k')

    lon = 139
    lat = 34

    x, y = m(lon, lat)

    plt.text(x, y, 'T', fontproperties=fontProp, fontweight='bold', fontsize=12, ha='left', va='bottom', color='k')

    return m


def plot_ind(ind, m, label, cmap, alpha):
    vMin, dV = 8, 6

    waypoints, speeds = zip(*ind)
    for i, leg in enumerate(zip(waypoints[:-1], waypoints[1:])):
        color = cmap((speeds[i] - vMin) / dV) if not isinstance(cmap, str) else cmap
        label = None if i > 0 else label
        m.drawgreatcircle(leg[0][0], leg[0][1], leg[1][0], leg[1][1], label=label, linewidth=1,
                          alpha=alpha, color=color, zorder=3)


def plot_routes(frontsDict, save=False):
    (u, v), lons, lats = CurrentDataRetriever(datetime(2014, 10, 28), nDays=6, DIR=Path('D:/')).get_kc_data()

    for pair, frontTup in frontsDict.items():
        print('\r', pair, end='')
        fronts, refFronts = frontTup
        for run, (front, refFront) in enumerate(zip(fronts, refFronts)):
            fig, ax = plt.subplots()
            m = navigation_area(ax, u, v, lons, lats)
            labels = [None, 'Reference']

            # Do for front and ref front
            for idx, f in enumerate([front, refFront]):
                label = labels[idx]
                cmap = 'k' if idx == 1 else colorbar(m)
                for j, ind in enumerate(f.items):
                    if idx == 1 and j > 0:
                        continue
                    plot_ind(ind, m, label=label, cmap=cmap, alpha=1 if idx == 1 else 0.9)

            ax.legend(loc='upper right', prop=fontProp)
            plt.grid()
            plt.xticks(fontproperties=fontProp)
            plt.yticks(fontproperties=fontProp)

            if save:
                fig.savefig('{}_routeM_{}'.format(pair, run), dpi=300)
                fig.savefig('{}_routeM_{}.pdf'.format(pair, run), bbox_inches='tight', pad_inches=.02)


def plot_subplot_routes(frontsDict, save=False):
    (u, v), lons, lats = CurrentDataRetriever(datetime(2014, 10, 28), nDays=6, DIR=Path('D:/')).get_kc_data()
    run = 2
    fig, ax = plt.subplots()
    m = navigation_area(ax, u, v, lons, lats)
    labels = [['KT', 'Reference'], ['TK', 'Reference']]
    colors = [['#2ca02c', 'k'], ['#d62728', 'k']]

    for plotIdx, (pair, frontTup) in enumerate(frontsDict.items()):
        print('\r', pair, end='')
        fronts, refFronts = frontTup
        front = fronts[run]
        refFront = refFronts[run]
        # Do for front and ref front
        for idx, f in enumerate([front, refFront]):
            if idx == 1 and plotIdx == 1:
                continue
            label = labels[plotIdx][idx]
            color = colors[plotIdx][idx]
            for j, ind in enumerate(f.items):
                if idx == 1 and j > 0:
                    continue
                if j > 0:
                    label = None
                plot_ind(ind, m, label=label, cmap=color, alpha=1 if idx == 1 else 0.9)

    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0], handles[2], handles[1]], [labels[0], labels[2], labels[1]], loc='lower right', prop=fontProp,
              fancybox=False, framealpha=1, edgecolor='k')
    plt.grid()
    plt.xticks(fontproperties=fontProp)
    plt.yticks(fontproperties=fontProp)

    if save:
        fig.savefig('bi_routeM_{}'.format(run), dpi=300)
        fig.savefig('bi_routeM_{}.pdf'.format(run), bbox_inches='tight', pad_inches=.02)


_frontsDict, _planner = get_fronts_dict()


# compute_metrics(_frontsDict)
# save_fronts(_frontsDict, _planner)
# plot_fronts(_frontsDict, _planner, save=True)
# plot_routes(_frontsDict, save=True)
plot_subplot_routes(_frontsDict, save=True)
plt.show()
