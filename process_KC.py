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
fontProp = fm.FontProperties(fname=fontPropFP)


def create_raw_dicts():
    loadDir = Path('D:/output/current/KC/NSGA2_varSP_BFalse_ECA1.0/2/')
    rawDir = loadDir / 'raw'
    os.chdir(loadDir)

    refFiles = [file for file in os.listdir(rawDir) if 'R' in file]
    print('refFiles', refFiles)
    planner = main.RoutePlanner(bathymetry=False, fuelPrice=1., ecaFactor=1, vesselName='Tanaka')
    planner.evaluator.set_classes(inclCurr=True, inclWeather=False, nDays=7, startDate=datetime(2014, 11, 15))
    evaluate = planner.evaluator.evaluate

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

    return frontsDict, planner


def compute_metrics(name, frontsDict):
    writer = pd.ExcelWriter('output_{}_metrics.xlsx'.format(name))
    pairs = list(frontsDict.keys())
    dfBinaryHV = pd.DataFrame(columns=pairs)
    dfCoverage = pd.DataFrame(columns=pairs)

    for pair, (fronts, refFronts) in frontsDict.items():
        print('\r', pair, end='')

        for front, refFront in zip(fronts, refFronts):
            biHV = indicators.binary_hypervolume(front, refFront)
            coverage = indicators.two_sets_coverage(front, refFront)
            dfBinaryHV = dfBinaryHV.append({pair: biHV}, ignore_index=True)
            dfCoverage = dfCoverage.append({pair: coverage}, ignore_index=True)

    for df in [dfBinaryHV, dfCoverage]:
        mean, std, minn, maxx = df.mean(), df.std(), df.min(), df.max()
        df.loc['mean'] = mean
        df.loc['std'] = std
        df.loc['min'] = minn
        df.loc['max'] = maxx

    dfCoverage.to_excel(writer, sheet_name='{}_C'.format(name))
    dfBinaryHV.to_excel(writer, sheet_name='{}_B'.format(name))
    writer.close()


def save_fronts(frontsDict, planner):
    writer = pd.ExcelWriter('output_fronts.xlsx')
    evaluate2 = planner.evaluator.evaluate2
    for pair, frontTup in frontsDict.items():
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
            ax1.set_ylabel('Average speed increase [knots]', fontproperties=fontProp)
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
                # tikzplotlib.save("{}_frontM_{}.tex".format(pair, run))


def navigation_area(ax, uin, vin, lons, lats):
    extent = (120, 25, 142, 37)
    left, bottom, right, top = extent
    m = Basemap(projection='merc', resolution='l', ax=ax,
                llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left, urcrnrlon=right)
    m.drawmapboundary()
    m.fillcontinents(zorder=2)
    m.drawcoastlines()

    # Currents
    dLon = extent[2] - extent[0]
    dLat = extent[3] - extent[1]
    vLon = int(dLon * 4)
    vLat = int(dLat * 4)
    uRot, vRot, x, y = m.transform_vector(uin, vin, lons, lats, vLon, vLat, returnxy=True)
    Q = m.quiver(x, y, uRot, vRot, np.hypot(uRot, vRot),
                 pivot='mid', width=0.002, headlength=4, cmap='Blues', scale=90, ax=ax)
    ax.quiverkey(Q, 0.4, 1.1, 2, r'$2$ knots', labelpos='E')

    return m


def colorbar(m):
    cmap = cm.get_cmap('jet', 12)
    cmapList = [cmap(i) for i in range(cmap.N)][1:-1]
    cmap = cl.LinearSegmentedColormap.from_list('Custom cmap', cmapList, cmap.N - 2)

    vMin, dV = 8, 6

    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = m.colorbar(sm, norm=plt.Normalize(vmin=vMin, vmax=vMin + dV),
                    size=0.2, pad=0.05, location='right')
    nTicks = 6
    cb.ax.set_yticklabels(['%.1f' % round(vMin + i * dV / (nTicks - 1), 1) for i in range(nTicks)],
                          fontproperties=fontProp)
    cb.set_label('Nominal speed [knots]', rotation=270, labelpad=15, fontproperties=fontProp)

    return cmap


def plot_ind(ind, m, label, cmap, alpha):
    vMin, dV = 8, 6

    waypoints, speeds = zip(*ind)
    for i, leg in enumerate(zip(waypoints[:-1], waypoints[1:])):
        color = cmap((speeds[i] - vMin) / dV) if cmap != 'k' else cmap
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


_frontsDict, _planner = create_raw_dicts()


# compute_metrics(key, _fronts)
# save_fronts(*create_raw_dicts())
plot_fronts(_frontsDict, _planner, save=True)
# plot_routes(_frontsDict, save=True)
plt.show()

