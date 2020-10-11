import indicators
import os
import pickle
import main
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np
import pandas as pd

from data_config.wind_data import WindDataRetriever
from datetime import datetime
from deap import tools
from matplotlib import font_manager as fm
from matplotlib import cm, rcParams
from mpl_toolkits.basemap import Basemap
from pathlib import Path

rcParams['text.usetex'] = True
fontPropFP = "C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/tex-gyre-pagella.regular.otf"
fontProp = fm.FontProperties(fname=fontPropFP)


locationsDates = {'NyP': datetime(2017, 9, 4),
                  'NoNy': datetime(2011, 1, 25, 15),
                  'PH': datetime(2013, 9, 24, 12),
                  'KS': datetime(2011, 5, 28),
                  'VMa': datetime(2015, 6, 21)}

loadDir = Path('D:/output/weather/WTH/NSGA2_varSP_BFalse_ECA1.0/5')
rawDir = loadDir / 'raw'
os.chdir(loadDir)
planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)
evaluate = planner.evaluator.evaluate
evaluate2 = planner.evaluator.evaluate2


def evaluate_new_fronts(fronts):
    newFronts = []
    for front in fronts:
        fits = [evaluate(ind, revert=False, includePenalty=False) for ind in front]
        for fit, ind in zip(fits, front.items):
            ind.fitness.values = fit
        newFront = tools.ParetoFront()
        newFront.update(front.items)
        newFronts.append(newFront)
    return newFronts


def get_fronts_dict():
    frontsDictFP = loadDir / 'frontsDict'

    if os.path.exists(frontsDictFP):
        with open(frontsDictFP, 'rb') as fh:
            frontsDict = pickle.load(fh)
            print('loaded', frontsDictFP)
        return frontsDict

    refFiles = [file for file in os.listdir(rawDir) if 'R' in file]
    print('refFiles', refFiles)

    refFrontsDict = {}
    for refFile in refFiles:
        split = refFile.split('_')
        pair = split[-1]
        date = locationsDates[pair]

        planner.evaluator.set_classes(inclCurr=False, inclWeather=True, nDays=30, startDate=date)

        with open(rawDir / refFile, 'rb') as fh:
            refRawList = pickle.load(fh)
        refFronts = [refRaw['fronts'][0][0] for refRaw in refRawList]
        refFrontsDict[pair] = evaluate_new_fronts(refFronts)

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

    return frontsDict


def compute_metrics(frontsDict):
    writer = pd.ExcelWriter('output_metrics.xlsx')

    pairs = list(locationsDates.keys)
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

        dfCoverage.to_excel(writer, sheet_name='{}_C'.format(pair))
        dfBinaryHV.to_excel(writer, sheet_name='{}_B'.format(pair))

    writer.close()


def plot(frontsDict, plotFronts=False, plotRoutes=False, savePlots=True):
    writer = pd.ExcelWriter('output_fronts.xlsx')
    for pair, (fronts, refFronts) in frontsDict.items():
        date = locationsDates[pair]
        planner.evaluator.set_classes(inclCurr=False, inclWeather=True, nDays=30, startDate=date)
        print('\r', pair, end='')

        dfPairList, refDFPairList = [], []
        for run, (front, refFront) in enumerate(zip(fronts, refFronts)):
            dataFrames = []
            # Do for front and ref front
            for idx, f in enumerate([front, refFront]):
                # (Objective) values
                days, cost, dist, _, avgSpeed = zip(*map(evaluate2, f.items))
                cost = np.array(cost) * 1000.
                hours = np.array(days) * 24.
                speedLoss = np.array(dist) / np.array(hours) - np.array(avgSpeed)
                df0 = pd.DataFrame(np.transpose(np.stack([hours, cost, dist, avgSpeed, speedLoss])),
                                   columns=['T', 'C', 'D', 'V', 'V_loss'])

                # Statistics
                dfStat = pd.DataFrame([df0.mean(), df0.std(), df0.min(), df0.max()],
                                      index=['mean', 'std', 'min', 'max'],
                                      columns=['T', 'C', 'D', 'V', 'V_loss'])
                dfStatPairList = dfPairList if idx == 0 else refDFPairList
                dfStatPairList.append(dfStat)

                # Append dataframes
                df0 = dfStat.append(df0, ignore_index=False)
                dataFrames.append(df0)

                # Plot fronts
                if plotFronts:
                    fig = plot_front(front, refFront)
                    # plt.show()
                    if savePlots:
                        fig.savefig('front_plots/{}_frontM_{}'.format(pair, run), dpi=300)
                        fig.savefig('front_plots/{}_frontM_{}.pdf'.format(pair, run),
                                    bbox_inches='tight', pad_inches=.01)
                        # tikzplotlib.save("{}_frontM_{}.tex".format(pair, run))
                    plt.close('all')

                if plotRoutes:
                    lon0, lon1 = f.items[0][0][0][0], f.items[0][-1][0][0]
                    fig = plot_routes(front, refFront, date, np.average(days), lon0, lon1)
                    # plt.show()
                    if savePlots:
                        fig.savefig('route_plots/{}_routeM_{}'.format(pair, run), dpi=300)
                        fig.savefig('route_plots/{}_routeM_{}.pdf'.format(pair, run),
                                    bbox_inches='tight', pad_inches=.02)

            # Write to Excel sheet
            dfFronts, dfRefFronts = dataFrames
            dfRefFronts.to_excel(writer, sheet_name='{}_R{}'.format(pair, run))
            dfFronts.to_excel(writer, sheet_name='{}_{}'.format(pair, run))

        dfPair = pd.concat(dfPairList).groupby(level=0).mean()
        dfRefPair = pd.concat(refDFPairList).groupby(level=0).mean()
        dfPair.to_excel(writer, sheet_name='S_{}'.format(pair))
        dfRefPair.to_excel(writer, sheet_name='S_{}_R'.format(pair))

    writer.close()


def plot_front(front, refFront):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 16))
    ax2.set_xlabel('Travel time [d]', fontproperties=fontProp)
    ax2.set_ylabel(r'Fuel cost [$\times 10^3$  USD]', fontproperties=fontProp)
    ax1.set_ylabel('Average speed loss [knots]', fontproperties=fontProp)
    # noinspection PyProtectedMember
    cycleFront = ax2._get_lines.prop_cycler
    labels = ['Incl. wind', 'Reference']

    # Do for front and ref front
    for idx, f in enumerate([front, refFront]):
        # (Objective) values
        days, cost, dist, _, avgSpeed = zip(*map(evaluate2, f.items))
        hours = np.array(days) * 24.
        speedLoss = (np.array(dist) / np.array(hours) - np.array(avgSpeed)) * -1.
        color = next(cycleFront)['color']

        # Plot front
        marker, s, zorder = 'o', 1, 2
        label = labels[idx]
        ax2.scatter(days, cost, color=color, marker=marker, s=s, label=label, zorder=zorder)
        ax1.scatter(days, speedLoss, color=color, marker=marker, s=s, zorder=zorder)

    ax2.legend(prop=fontProp)
    ax2.grid()
    ax1.grid()
    plt.xticks(fontproperties=fontProp)
    plt.yticks(fontproperties=fontProp)

    return fig


def navigation_area(ax, date, travelDays, lon0, lon1, extent):
    left, bottom, right, top = extent
    m = Basemap(projection='merc', resolution='i', ax=ax,
                llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left, urcrnrlon=right)
    m.drawmapboundary()
    m.fillcontinents(zorder=2, lake_color='lightgray')
    m.drawcoastlines()

    # Weather
    plot_weather(m, date, travelDays, lon0, lon1)

    return m


def colorbar(m):
    cmap = cm.get_cmap('jet', 12)
    cmapList = [cmap(i) for i in range(cmap.N)][1:-1]
    cmap = cl.LinearSegmentedColormap.from_list('Custom cmap', cmapList, cmap.N - 2)

    vMin, dV = 8.8, 15.2 - 8.8

    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = m.colorbar(sm, norm=plt.Normalize(vmin=vMin, vmax=vMin + dV),
                    size=0.2, pad=0.05, location='right')
    nTicks = 6
    cb.ax.set_yticklabels(['%.1f' % round(vMin + i * dV / (nTicks - 1), 1) for i in range(nTicks)],
                          fontproperties=fontProp)
    cb.set_label('Nominal vessel speed [knots]', rotation=270, labelpad=15, fontproperties=fontProp)

    return cmap


def plot_routes(front, refFront, date, travelDays, lon0, lon1):
    minx, miny, maxx, maxy = 180, 90, -180, -90
    for wp in front.items[0]:
        lon, lat = wp[0]
        minx, maxx = min(minx, lon), max(maxx, lon)
        miny, maxy = min(miny, lat), max(maxy, lat)

    margin = 5
    extent = (minx - margin, miny - margin, maxx + margin, maxy + margin)

    fig, ax = plt.subplots()
    m = navigation_area(ax, date, travelDays, lon0, lon1, extent)
    labels = [None, 'Reference']
    vMin, dV = 8.8, 15.2 - 8.8

    # Do for front and ref front
    for idx, f in enumerate([front, refFront]):
        label = labels[idx]
        cmap = 'k' if idx == 1 else colorbar(m)
        for j, ind in enumerate(f.items):
            if idx == 1 and j > 0:
                continue

            alpha = 1 if idx == 1 else 0.9
            waypoints, speeds = zip(*ind)
            for i, leg in enumerate(zip(waypoints[:-1], waypoints[1:])):
                color = cmap((speeds[i] - vMin) / dV) if cmap != 'k' else cmap
                label = None if i > 0 else label
                m.drawgreatcircle(leg[0][0], leg[0][1], leg[1][0], leg[1][1], label=label, linewidth=1,
                                  alpha=alpha, color=color, zorder=3)

    ax.legend(loc='upper right', prop=fontProp)
    plt.grid()
    plt.xticks(fontproperties=fontProp)
    plt.yticks(fontproperties=fontProp)
    return fig


def plot_weather(m, dateTime, travelDays, lon0, lon1):
    hourPeriod = 24 // 6
    maxTravelDays = int(np.ceil(travelDays))
    print('weather travel days', travelDays)

    # Get wind data
    retriever = WindDataRetriever(nDays=maxTravelDays, startDate=dateTime)
    ds = retriever.get_data(forecast=False)

    # Create lon indices
    res = 0.5
    startEndIndex = [int(round((lon + 180) / res)) for lon in [lon0, lon1]]
    (lonS, lonT) = sorted([0 if idx == 720 else idx for idx in startEndIndex])
    lonIndices = np.linspace(lonS, lonT, maxTravelDays * 4 + 1).astype(int)
    print('weather lon indices', len(lonIndices), ':', lonIndices)

    lonPairs = zip(lonIndices[:-1], lonIndices[1:])

    # Create date indices
    dateIndices = np.linspace(0, travelDays * hourPeriod, maxTravelDays * 4).astype(int)
    print('weather date indices', len(dateIndices), ':', dateIndices)

    # Initialize variables for contour
    lons, lats = np.linspace(-180, 179.5, 720), np.linspace(-90, 89.5, 360)
    x, y = m(*np.meshgrid(lons, lats))

    for i, (dateInd, lonPair) in enumerate(zip(dateIndices, lonPairs)):
        lon0, lon1 = min(lonPair), max(lonPair) + 1
        BN = ds[0, dateInd, :-1, lon0:lon1]
        xSlice, ySlice = x[:, lon0:lon1], y[:, lon0:lon1]
        m.contourf(xSlice, ySlice, BN, vmin=0, vmax=12, cmap=cm.get_cmap('jet', 12))

    vmin, vmax = 0, 12
    nColors = 12
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet', nColors))
    cb = m.colorbar(sm, norm=plt.Normalize(vmin=vmin, vmax=vmax), size=0.2, pad=0.2, location='bottom')
    cb.set_label('Wind [BFT]', fontproperties=fontProp)


_frontsDict = get_fronts_dict()

# compute_metrics(key, _fronts)
plot(_frontsDict, plotRoutes=True, plotFronts=True)
