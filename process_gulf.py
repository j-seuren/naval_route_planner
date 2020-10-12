import indicators
import os
import pickle
import main
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np
import pandas as pd
# import tikzplotlib

from data_config.current_data import CurrentDataRetriever
from datetime import datetime
from deap import tools
from matplotlib import font_manager as fm
from matplotlib import cm, rcParams
from mpl_toolkits.basemap import Basemap
from pathlib import Path

rcParams['text.usetex'] = True
fontPropFP = "C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/tex-gyre-pagella.regular.otf"
fontProp = fm.FontProperties(fname=fontPropFP)

dates = [datetime(2014, 11, 25), datetime(2015, 5, 4)]
loadDir = Path('D:/output/current/Gulf')
rawDir = loadDir / 'raws'
os.chdir(loadDir)


def evaluate_new_fronts(fronts, evaluate):
    newFronts = []
    for front in fronts:
        fits = map(evaluate, front.items)
        for fit, ind in zip(fits, front.items):
            ind.fitness.values = fit
        newFront = tools.ParetoFront()
        newFront.update(front.items)
        newFronts.append(newFront)
    return newFronts


def get_front_dicts():
    frontsDictFP = loadDir / 'frontsDict'

    if os.path.exists(frontsDictFP) and False:
        with open(frontsDictFP, 'rb') as fh:
            frontsDict = pickle.load(fh)
            print('loaded', frontsDictFP)
        return frontsDict

    refFrontsDict, frontsDict = {}, {}
    for d, date in enumerate(dates):
        year = date.year
        refFrontsDict[year], frontsDict[year] = {}, {}

        # Initialize planner
        planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)
        planner.evaluator.set_classes(inclCurr=True, inclWeather=False, nDays=7, startDate=date)
        evaluate = planner.evaluator.evaluate

        for speed in ['var', 'min', 'max']:
            speedDir = rawDir / speed
            refFilesSpeed = [file for file in os.listdir(speedDir) if 'R' in file and str(year) in file]
            print('refFiles', year, speed, refFilesSpeed)

            refFrontsDict[year][speed] = {}
            for refFile in refFilesSpeed:
                print('\r', refFile, end='')
                split = refFile.split('_')
                pair = split[-1]
                with open(speedDir / refFile, 'rb') as fh:
                    refRawList = pickle.load(fh)
                fronts = [refRaw['fronts'][0][0] for refRaw in refRawList]
                refFrontsDict[year][speed][pair] = evaluate_new_fronts(fronts, evaluate)
            print('')
            files = [file for file in os.listdir(speedDir) if 'R' not in file and str(year) in file]
            print('files', year, speed, files)

            frontsDict[year][speed] = {}
            for file in files:
                print('\r', file, end='')
                split = file.split('_')
                pair = split[-1]
                with open(speedDir / file, 'rb') as fh:
                    rawList = pickle.load(fh)
                fronts = [raw['fronts'][0][0] for raw in rawList]
                fronts = evaluate_new_fronts(fronts, evaluate)
                frontsDict[year][speed][pair] = (fronts, refFrontsDict[year][speed][pair])
            print('')

    with open(frontsDictFP, 'wb') as fh:
        pickle.dump(frontsDict, fh)

    return frontsDict


def metrics_plot(frontsDict, plotFront=False, plotRoute=False, savePlot=False):
    years = list(frontsDict.keys())
    for y, year in enumerate(years):
        date = dates[y]
        planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)
        planner.evaluator.set_classes(inclCurr=True, inclWeather=False, nDays=7, startDate=date)

        if plotRoute:
            u, v = CurrentDataRetriever(date, nDays=7, DIR=Path('D:/')).get_data()
            lons = np.linspace(-179.875, 179.875, 1440)
            lats = np.linspace(-89.875, 89.875, 72)
            print('current shapes', u.shape, lons.shape, lats.shape)
        else:
            u = v = lons = lats = None

        writer = pd.ExcelWriter('output_{}_metrics.xlsx'.format(year))
        speeds = frontsDict[year].keys()
        for speed in speeds:
            pairs = frontsDict[year][speed].keys()

            # Compute Pareto metrics for variable speed only
            if speed == 'var':
                dfBinaryHV = pd.DataFrame(columns=pairs, index=range(5))
                dfCoverage = pd.DataFrame(columns=pairs, index=range(5))

                cSpeeds = ['max', 'min']

                for pair in pairs:
                    print('\r', pair, end='')
                    fronts, refFronts = frontsDict[year][speed][pair]

                    for run, (front, refFront) in enumerate(zip(fronts, refFronts)):
                        biHV = indicators.binary_hypervolume_ratio(front, refFront)
                        coverage = indicators.two_sets_coverage(front, refFront)
                        dfBinaryHV.iloc[run][pair] = biHV
                        dfCoverage.iloc[run][pair] = coverage

                        if plotFront:
                            # Get 'fronts' of constant speed simulations for plotting
                            cSpeedFrontsList = [
                                [frontsDict[year][cSpeed][pair][refIdx][run] for refIdx in range(2)]
                                for cSpeed in cSpeeds]
                            fig = plot_front(front, refFront, cSpeedFrontsList)
                            # plt.show()
                            if savePlot:
                                fig.savefig('front_plots/{}_frontM_{}_{}'.format(pair, year, run), dpi=300)
                                fig.savefig('front_plots/{}_frontM_{}_{}.pdf'.format(pair, year, run),
                                            bbox_inches='tight', pad_inches=.01)
                                # tikzplotlib.save("{}_frontM_{}.tex".format(pair, run))
                            plt.close('all')

                        if plotRoute:
                            fig = plot_routes(front, refFront, u, v, lons, lats)
                            # plt.show()
                            if savePlot:
                                fig.savefig('route_plots/{}_routeM_{}_{}'.format(pair, year, run), dpi=300)
                                fig.savefig('route_plots/{}_routeM_{}_{}.pdf'.format(pair, year, run),
                                            bbox_inches='tight', pad_inches=.02)

                for df in [dfBinaryHV, dfCoverage]:
                    mean, std, minn, maxx = df.mean(), df.std(), df.min(), df.max()
                    df.loc['mean'] = mean
                    df.loc['std'] = std
                    df.loc['min'] = minn
                    df.loc['max'] = maxx

                dfCoverage.to_excel(writer, sheet_name='{}_C'.format(year))
                dfBinaryHV.to_excel(writer, sheet_name='{}_B'.format(year))

        writer.close()


def plot_front(front, refFront, cSpeedsFrontsList):
    planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)
    evaluate2 = planner.evaluator.evaluate2

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 16))
    ax2.set_xlabel('Travel time [d]', fontproperties=fontProp)
    ax2.set_ylabel(r'Fuel cost [$\times 10^3$  USD]', fontproperties=fontProp)
    ax1.set_ylabel('Average speed increase [kn]', fontproperties=fontProp)
    # noinspection PyProtectedMember
    cycleFront = ax2._get_lines.prop_cycler
    labels = ['', 'R']

    # Do for front and ref front
    for idx, f in enumerate([front, refFront]):
        # (Objective) values
        days, cost, dist, _, avgSpeed = zip(*map(evaluate2, f.items))
        hours = np.array(days) * 24.
        currentSpeed = np.array(dist) / np.array(hours) - np.array(avgSpeed)
        color = next(cycleFront)['color']

        # Plot front
        marker, s, zorder = 'o', 1, 2
        label = 'V' + labels[idx]
        ax2.scatter(days, cost, color=color, marker=marker, s=s, label=label, zorder=zorder)
        ax1.scatter(days, currentSpeed, color=color, marker=marker, s=s, zorder=zorder)

    # Plot constant speeds
    colors = [next(cycleFront)['color'], next(cycleFront)['color']]
    for cSpeedIdx, cSpeedFrontTup in enumerate(cSpeedsFrontsList):
        for refIdx, front in enumerate(cSpeedFrontTup):
            days, cost, dist, _, avgSpeed = zip(*map(evaluate2, front.items))
            currentSpeed = np.array(dist) / (np.array(days) * 24.) - np.array(avgSpeed)

            zorder = 3
            (marker, s) = ('+', 50) if refIdx == 1 else ('x', 35)
            label = 'C' + labels[refIdx] if cSpeedIdx == 0 else None
            ax2.scatter(days, cost, color=colors[refIdx], marker=marker, s=s, label=label, zorder=zorder)
            ax1.scatter(days, currentSpeed, color=colors[refIdx], marker=marker, s=s, zorder=zorder)

    ax2.legend(prop=fontProp)
    ax2.grid()
    ax1.grid()
    plt.xticks(fontproperties=fontProp)
    plt.yticks(fontproperties=fontProp)

    return fig


def navigation_area(ax, u, v, lons, lats):
    dateIdx = len(u) // 2

    mrg = 1
    extent = (-74 - mrg, 32 - mrg, -50 + mrg, 46 + mrg)

    lonSlice = slice(find_nearest_idx(lons, extent[0]), find_nearest_idx(lons, extent[2]))
    latSlice = slice(find_nearest_idx(lats, extent[1]), find_nearest_idx(lats, extent[3]))
    lons, lats = lons[lonSlice], lats[latSlice]
    u, v = u[dateIdx, latSlice, lonSlice], v[dateIdx, latSlice, lonSlice]

    left, bottom, right, top = extent
    m = Basemap(projection='merc', resolution='i', ax=ax,
                llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left, urcrnrlon=right)
    m.drawmapboundary()
    m.fillcontinents(zorder=2)
    m.drawcoastlines()

    # Currents
    dLon = extent[2] - extent[0]
    dLat = extent[3] - extent[1]
    vLon = int(dLon * 4)
    vLat = int(dLat * 4)
    print(np.max(np.hypot(u, v)))
    uRot, vRot, x, y = m.transform_vector(u, v, lons, lats, vLon, vLat, returnxy=True)
    Q = m.quiver(x, y, uRot, vRot, np.hypot(uRot, vRot),
                 pivot='mid', width=0.002, headlength=4, cmap='Blues', scale=90, ax=ax)
    ax.quiverkey(Q, 0.4, 1.1, 2, r'$2$ knots', labelpos='E')

    return m


def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def colorbar(m):
    cmap = cm.get_cmap('viridis', 12)
    # cmapList = [cmap(i) for i in range(cmap.N)][1:-1]
    # cmap = cl.LinearSegmentedColormap.from_list('Custom cmap', cmapList, cmap.N - 2)

    vMin, dV = 8.8, 15.2 - 8.8

    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = m.colorbar(sm, norm=plt.Normalize(vmin=vMin, vmax=vMin + dV),
                    size=0.2, pad=0.05, location='right')
    nTicks = 6
    cb.ax.set_yticklabels(['%.1f' % round(vMin + i * dV / (nTicks - 1), 1) for i in range(nTicks)],
                          fontproperties=fontProp)
    cb.set_label('Nominal vessel speed [kn]', rotation=270, labelpad=15, fontproperties=fontProp)

    return cmap


def plot_routes(front, refFront, u, v, lons, lats):
    fig, ax = plt.subplots()
    m = navigation_area(ax, u, v, lons, lats)
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


metrics_plot(get_front_dicts(), plotFront=True, plotRoute=True, savePlot=True)
