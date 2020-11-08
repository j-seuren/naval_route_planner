import indicators
import os
import pickle
import main
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np
import pandas as pd
import tikzplotlib

from data_config.wind_data import WindDataRetriever
from datetime import datetime
from deap import tools
from matplotlib import font_manager as fm
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from scipy import spatial

fontPropFP = "C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/tex-gyre-pagella.regular.otf"
fontProp = fm.FontProperties(fname=fontPropFP, size=9)

locationsDates = {
    'NyP': datetime(2017, 9, 4),
    'NoNy': datetime(2011, 1, 25, 15),
    'PH': datetime(2013, 9, 24, 12),
    'KS': datetime(2011, 5, 28),
    'VMa': datetime(2015, 6, 21)
}

loadDir = Path('D:/output/weather_16_10/WTH')
rawDir = loadDir / 'raw'
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


def get_fronts_dict():
    frontsDictFP = loadDir / 'frontsDict'

    if os.path.exists(frontsDictFP):
        with open(frontsDictFP, 'rb') as fh:
            frontsDict = pickle.load(fh)
            print('loaded', frontsDictFP)
        return frontsDict

    refFrontsDict, frontsDict = {}, {}
    for pair, date in locationsDates.items():
        planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)
        planner.evaluator.set_classes(inclCurr=False, inclWeather=True, nDays=30, startDate=date)
        evaluate = planner.evaluator.evaluate

        refFrontsDict[pair], frontsDict[pair] = {}, {}
        for speed in ['var']:
            speedDir = rawDir / speed
            refFiles = [file for file in os.listdir(speedDir) if 'R' in file and pair in file]
            print('refFiles', refFiles)

            for refFile in refFiles:
                with open(speedDir / refFile, 'rb') as fh:
                    refRawList = pickle.load(fh)
                refFronts = [refRaw['fronts'][0][0] for refRaw in refRawList]
                newRefFronts = evaluate_new_fronts(refFronts, evaluate)
                refFrontsDict[pair][speed] = newRefFronts * 5
                print('1', len(refFrontsDict[pair][speed]))

            files = [file for file in os.listdir(speedDir) if 'R' not in file and pair in file]
            print('files', files)

            for file in files:
                with open(speedDir / file, 'rb') as fh:
                    rawList = pickle.load(fh)
                fronts = [raw['fronts'][0][0] for raw in rawList]
                if len(fronts) == 1:
                    fronts = evaluate_new_fronts(fronts, evaluate) * 5
                else:
                    fronts = evaluate_new_fronts(fronts, evaluate)
                frontsDict[pair][speed] = (fronts, refFrontsDict[pair][speed])

    with open(frontsDictFP, 'wb') as fh:
        pickle.dump(frontsDict, fh)

    return frontsDict


def compute_metrics(frontsDict):
    writer = pd.ExcelWriter('output_metrics.xlsx')

    pairs = list(locationsDates.keys())
    dfBinaryHV = pd.DataFrame(columns=pairs)
    dfCoverage = pd.DataFrame(columns=pairs)

    for pair in locationsDates:
        (fronts, refFronts) = frontsDict[pair]['var']

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

        dfCoverage.to_excel(writer, sheet_name='{}_C'.format(pair))
        dfBinaryHV.to_excel(writer, sheet_name='{}_B'.format(pair))

    writer.close()


def save_fronts(frontsDict):
    speeds = ['var', 'min', 'max']

    # dfBinaryHV = pd.DataFrame(columns=locationsDates.keys(), index=range(5))
    # dfCoverage = pd.DataFrame(columns=locationsDates.keys(), index=range(5))

    writer = pd.ExcelWriter('output_front_stats.xlsx')
    for pair, date in locationsDates.items():
        planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)
        planner.evaluator.set_classes(inclCurr=False, inclWeather=True, nDays=30, startDate=date)
        evaluate2 = planner.evaluator.evaluate2

        for speed in speeds:
            fronts, refFronts = frontsDict[pair][speed]

            dfPairList, refDFPairList = [], []
            # Compute Pareto metrics for variable speed only
            for run, (front, refFront) in enumerate(zip(fronts, refFronts)):
                # dfBinaryHV.iloc[run][pair] = indicators.binary_hypervolume_ratio(front, refFront)
                # dfCoverage.iloc[run][pair] = indicators.two_sets_coverage(front, refFront)

                for idx, f in enumerate([front, refFront]):
                    # (Objective) values
                    days, cost, dist, _, avgSpeed = zip(*map(evaluate2, f.items))
                    hours = np.array(days) * 24.
                    speedLoss = np.array(dist) / np.array(hours) - np.array(avgSpeed)
                    df0 = pd.DataFrame(np.transpose(np.stack([days, cost, dist, avgSpeed, speedLoss])),
                                       columns=['T', 'C', 'D', 'V', 'V_loss'])

                    # Statistics
                    dfStat = pd.DataFrame([df0.mean(), df0.std(), df0.min(), df0.max()],
                                          index=['mean', 'std', 'min', 'max'],
                                          columns=['T', 'C', 'D', 'V', 'V_loss'])
                    dfStatPairList = dfPairList if idx == 0 else refDFPairList
                    dfStatPairList.append(dfStat)

                    # # Write to Excel sheet
                    # df0 = dfStat.append(df0, ignore_index=False)
                    # rStr = '' if idx == 0 else 'R'
                    # df0.to_excel(writer, sheet_name='{}_{}{}'.format(pair, rStr, run))

            for dLIdx, dfList in enumerate([dfPairList, refDFPairList]):
                rStr = '' if dLIdx == 0 else '_R'
                df = pd.concat(dfList).groupby(level=0).mean()
                df.to_excel(writer, sheet_name='S_{}_{}_{}'.format(pair, speed, rStr))

    writer.close()


def plot_all_fronts(frontsDict, save=False, convexHull=True):
    speed = 'var'
    # cSpeeds = ['max', 'min']

    for pair, date in locationsDates.items():
        planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)
        planner.evaluator.set_classes(inclCurr=False, inclWeather=True, nDays=30, startDate=date)
        evaluate2 = planner.evaluator.evaluate2

        fronts, refFronts = frontsDict[pair][speed]

        # Compute Pareto metrics for variable speed only
        for run, (front, refFront) in enumerate(zip(fronts, refFronts)):
            # Get 'fronts' of constant speed simulations for plotting
            # cSpeedFrontsList = [
            #     [frontsDict[pair][cSpeed][refIdx][run] for refIdx in range(2)] for cSpeed in cSpeeds]

            fig = plot_front(front, refFront, evaluate2, [], convexHull)
            # plt.show()
            if save:
                fn = 'front_plots{}/{}_frontM_{}'.format('/convexHull' if convexHull else '', pair, run)
                print('saved front plot', fn)
                fig.savefig(fn, dpi=300)
                fig.savefig(fn + '.pdf', bbox_inches='tight', pad_inches=.01)
                tikzplotlib.save(fn + '.tex')
            plt.clf()


def plot_all_routes(frontsDict, save=False, convexHull=True):
    speed = 'var'
    for pair, date in locationsDates.items():
        fronts, refFronts = frontsDict[pair][speed]

        # Compute Pareto metrics for variable speed only
        for run, (front, refFront) in enumerate(zip(fronts, refFronts)):
            lon0, lon1 = front.items[0][0][0][0], front.items[0][-1][0][0]
            avgDays = np.average([fit.values[0] for fit in front.keys])
            minDays = np.min([fit.values[0] for fit in front.keys])
            maxDays = np.max([fit.values[0] for fit in front.keys])
            for daysKey, travelDays in {'avgDays': avgDays, 'minDays': minDays, 'maxDays': maxDays}.items():
                fig = plot_front_routes(front, refFront, date, travelDays, lon0, lon1, convexHull)
                # plt.show()
                if save:
                    fn = 'route_plots{}/{}_routeM_{}_{}'.format('/convexHull' if convexHull else '', pair, run, daysKey)
                    print('saved route plot', fn)
                    fig.savefig(fn, dpi=300)
                    fig.savefig(fn + '.pdf', bbox_inches='tight', pad_inches=.02)
                plt.clf()


def plot_front(front, refFront, evaluate2, cSpeedFrontsList, convexHull):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 16))
    ax2.set_xlabel('Travel time [d]', fontproperties=fontProp)
    ax2.set_ylabel(r'Fuel cost [USD, $\times 1000$]', fontproperties=fontProp)
    ax1.set_ylabel('Average speed loss [kn]', fontproperties=fontProp)
    # noinspection PyProtectedMember
    cycleFront = ax2._get_lines.prop_cycler
    labels = ['Weather-optimized', 'Reference']

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

        if convexHull:
            fits = [[day, cost] for day, cost in zip(days, cost)]
            hull = spatial.ConvexHull(fits)
            hullPoints = hull.points[hull.vertices]
            travelTimes, fuelCosts = zip(*hullPoints)
            ax2.scatter(travelTimes, fuelCosts, color=color, marker='x', zorder=zorder)

    # # Plot constant speeds
    colors = [next(cycleFront)['color'], next(cycleFront)['color']]
    for cSpeedIdx, cSpeedFrontTup in enumerate(cSpeedFrontsList):
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


def navigation_area(fig, ax, date, travelDays, lon0, lon1, extent):
    left, bottom, right, top = extent
    if date.year == 2011 and date.month == 5:
        dLon2 = (180 - lon0 + lon1 + 180) / 2
        lon_0 = lon0 + dLon2 if lon0 + dLon2 < 180 else lon0 + dLon2 - 360
        width = 13000000
        height = 5500000
        m = Basemap(projection='stere', resolution='i', ax=ax, lon_0=lon_0 - 4, lat_0=46, width=width, height=height)
    else:
        m = Basemap(projection='merc', resolution='i', ax=ax,
                    llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left, urcrnrlon=right)
    m.drawmapboundary()
    m.fillcontinents(zorder=2, lake_color='lightgray')
    m.drawcoastlines()

    # Weather
    cmap = plot_weather(fig, ax, m, date, travelDays, lon0, lon1)

    return m, cmap


def navigation_area2(ax, date, travelDays, startEnd, extent):
    (lon0, lat0), (lon1, lat1) = startEnd

    left, bottom, right, top = extent
    if date.year == 2011 and date.month == 5:
        dLon2 = (180 - lon0 + lon1 + 180) / 2
        lon_0 = lon0 + dLon2 if lon0 + dLon2 < 180 else lon0 + dLon2 - 360
        width = 13000000
        height = 5500000
        m = Basemap(projection='stere', resolution='i', ax=ax, lon_0=lon_0 - 4, lat_0=46, width=width, height=height)
    else:
        m = Basemap(projection='merc', resolution='i', ax=ax,
                    llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left, urcrnrlon=right)
    m.drawmapboundary()
    m.fillcontinents(zorder=2, lake_color='lightgray')
    m.drawcoastlines()

    # Weather
    plot_weather2(m, date, travelDays, lon0, lon1)

    lon, lat = startEnd[0]
    x, y = m(lon, lat)

    m.scatter(x, y, marker='D', s=5, color='c',  zorder=4)

    lon, lat = startEnd[1]
    x, y = m(lon, lat)

    m.scatter(x, y, marker='D', s=5, color='m', zorder=4)

    return m


def plot_front_routes(front, refFront, date, travelDays, lon0, lon1, convexHull):
    minx, miny, maxx, maxy = 180, 90, -180, -90
    for wp in front.items[0]:
        lon, lat = wp[0]
        minx, maxx = min(minx, lon), max(maxx, lon)
        miny, maxy = min(miny, lat), max(maxy, lat)

    margin = 5
    extent = (minx - margin, miny - margin, maxx + margin, maxy + margin)

    fig, ax = plt.subplots()
    m, cmap0 = navigation_area(fig, ax, date, travelDays, lon0, lon1, extent)
    labels = [None, 'Reference']
    vMin, dV = 8.8, 15.2 - 8.8

    # Do for front and ref front
    for idx, f in enumerate([front, refFront]):
        inds = [ind for ind in f]
        if convexHull:
            fits = [fit.values for fit in f.keys]
            hull = spatial.ConvexHull(fits)
            inds = np.take(inds, hull.vertices, 0)

        label = labels[idx]
        cmap = 'k' if idx == 1 else cmap0
        for j, ind in enumerate(inds):
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


def plot_weather(fig, ax, m, dateTime, travelDays, lonStart, lonDest):
    hourPeriod = 24 // 6
    maxTravelDays = int(np.ceil(travelDays))
    print('weather travel days', travelDays)

    # Get wind data
    retriever = WindDataRetriever(nDays=maxTravelDays, startDate=dateTime)
    ds = retriever.get_data(forecast=False)

    # Create lon indices
    res = 0.5
    startEndIndex = [int(round((lon + 180) / res)) for lon in [lonStart, lonDest]]
    lon0, lon1 = [0 if idx == 720 else idx for idx in startEndIndex]
    if dateTime.year == 2011 and dateTime.month == 5:
        nSlices = maxTravelDays * hourPeriod + 1
        slicesLeft = int(round(nSlices * (180 - lonStart) / ((180 - lonStart) + abs(lonDest + 180))))
        slicesRight = nSlices - slicesLeft
        lonIndicesLeft = np.round(np.linspace(lon0, 719, slicesLeft)).astype(int)
        lonIndicesRight = np.round(np.linspace(0, lon1, slicesRight)).astype(int)
        lonPairs = list(zip(lonIndicesLeft[:-1], lonIndicesLeft[1:])) + list(zip(lonIndicesRight[:-1], lonIndicesRight[1:]))
        print(lonPairs)
    else:
        lonIndices = np.linspace(lon0, lon1, maxTravelDays * hourPeriod + 1).astype(int)
        lonPairs = list(zip(lonIndices[:-1], lonIndices[1:]))

    # Create date indices
    dateIndices = np.round(np.linspace(0, travelDays * hourPeriod - 1, maxTravelDays * hourPeriod)).astype(int)

    # Initialize variables for contour
    lons, lats = np.linspace(-180, 179.5, 720), np.linspace(-90, 90, 361)
    x, y = m(*np.meshgrid(lons, lats))

    for i, (dateInd, lonPair) in enumerate(zip(dateIndices, lonPairs)):
        lonMin, lonMax = min(lonPair), max(lonPair) + 1
        if lonPairs[0][0] < lonPairs[1][1]:
            if i == 0:
                lonMin = 0
            elif i == len(dateIndices) - 1:
                lonMax = -1
        elif lonPair[0] > lonPair[1]:
            if i == len(dateIndices) - 1:
                lonMin = 0
            elif i == 0:
                lonMax = -1
        lonSlice = slice(lonMin, lonMax)
        BN = ds[0, dateInd, :, lonSlice]
        xSlice, ySlice = x[:, lonSlice], y[:, lonSlice]
        try:
            m.contourf(xSlice, ySlice, BN, vmin=0, vmax=12, cmap=cm.get_cmap('purples', 13))
        except ValueError:
            print('something wrong')

    vmin, vmax = 0, 12
    nColors = 13
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('purples', nColors))
    cb = fig.colorbar(sm, norm=plt.Normalize(vmin=vmin, vmax=vmax), cax=cax)
    cb.set_label('Wind [BFT]', rotation=270, labelpad=15, fontproperties=fontProp)

    # Speed colorbar
    cmap = cm.get_cmap('jet', 12)
    # cmapList = [cmap(i) for i in range(cmap.N)][1:-1]
    # cmap = cl.LinearSegmentedColormap.from_list('Custom cmap', cmapList, cmap.N - 2)

    vMin, dV = 8.8, 15.2 - 8.8

    smS = plt.cm.ScalarMappable(cmap=cmap)
    cax = divider.append_axes("right", size="5%", pad=0.75)
    cbS = fig.colorbar(smS, norm=plt.Normalize(vmin=vMin, vmax=vMin + dV), cax=cax)
    nTicks = 6
    cbS.ax.set_yticklabels(['%.1f' % round(vMin + i * dV / (nTicks - 1), 1) for i in range(nTicks)],
                          fontproperties=fontProp)
    cbS.set_label('Nominal vessel speed [kn]', rotation=270, labelpad=15, fontproperties=fontProp)

    return cmap


def plot_group_routes(frontsDict, save=False, convexHull=True):
    fig, axs = plt.subplots(3, constrained_layout=True, figsize=(5, 9.2),
                            gridspec_kw=dict(height_ratios=[1, 1, 1],
                                             width_ratios=[1],
                                             hspace=0.))

    # Wind cmap
    windCmap = cm.get_cmap('Purples', 13)

    # Speed cmap
    speedCmap = cm.get_cmap('jet', 12)

    fns = ['KS_route.pdf', 'NoNy_route.pdf', 'PH_route.pdf']

    pairRun = {'KS': 3, 'NoNy': 0, 'PH': 1}
    titles = ['Keelung - San Francisco (KEE-SF) - stereographic projection',
              'English Channel - New York (EC-NY)',
              'Plymouth - Havana (PLH-HAV)']
    speed = 'var'
    i = -1
    for idx, (pair, date) in enumerate(locationsDates.items()):
        if pair not in pairRun:
            continue
        i += 1
        ax = axs[i]
        fronts, refFronts = frontsDict[pair][speed]

        run = pairRun[pair]
        front, refFront = fronts[run], refFronts[run]

        startEnd = (front.items[0][0][0], front.items[0][-1][0])
        avgDays = np.average([fit.values[0] for fit in front.keys])
        # minDays = np.min([fit.values[0] for fit in front.keys])
        # maxDays = np.max([fit.values[0] for fit in front.keys])
        plot_front_routes2(ax, speedCmap, front, refFront, date, avgDays, startEnd, convexHull)

    # axs[0].legend(loc=(0.45, 1.1), prop=fontProp, fancybox=False, framealpha=1, edgecolor='k')

    # Wind colorbar
    aspect = 50
    vmin, vmax = 0, 12
    dV = vmax - vmin
    windMappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=windCmap)
    cbW = fig.colorbar(windMappable, ax=axs[2], location='bottom', pad=0.01, aspect=aspect, shrink=1)
    cbW.set_label('Wind [BFT]', fontproperties=fontProp)
    nTicks = 7
    cbW.ax.set_xticklabels(['%.0f' % round(vmin + i * dV / (nTicks - 1), 1) for i in range(nTicks)],
                           fontproperties=fontProp)

    # Speed colorbar
    vMin, dV = 8.8, 15.2 - 8.8
    speedShrink = 0.6
    speedMappable = plt.cm.ScalarMappable(cmap=speedCmap)
    cbS = fig.colorbar(speedMappable, norm=plt.Normalize(vmin=vMin, vmax=vMin + dV), ax=axs[:], location='right',
                       pad=0.003, shrink=speedShrink, aspect=aspect * .3)  # , rotation=270)
    nTicks = 6
    cbS.ax.set_yticklabels(['%.1f' % round(vMin + i * dV / (nTicks - 1), 1) for i in range(nTicks)],
                           fontproperties=fontProp)
    # cbS.set_label('Nominal vessel speed [kn]', rotation=270, labelpad=15, fontproperties=fontProp)
    cbS.set_label('Nominal vessel speed [kn]', rotation=270, labelpad=15, fontproperties=fontProp)

    # for i, ax in enumerate(axs):
    #     ax.set_title(titles[i], fontproperties=fontProp, size=9)

    if save:
        fn = 'route_plots{}/comb_routeM'.format('/convexHull' if convexHull else '')
        fig.savefig(fn, dpi=300)
        fig.savefig(fn + '.pdf', bbox_inches='tight', pad_inches=.02)
        # tikzplotlib.save(fn + '.tex', encoding='utf-8')
        print('saved route plot', fn)

        for i, ax in enumerate(axs):
            # Save just the portion _inside_ the second axis's boundaries
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

            # Expanded
            fig.savefig(fns[i], bbox_inches=extent.expanded(1.01, 1.01), pad_inches=.0)


def plot_front_routes2(ax, cmap0, front, refFront, date, travelDays, startEnd, convexHull):
    minx, miny, maxx, maxy = 180, 90, -180, -90
    for wp in front.items[0]:
        lon, lat = wp[0]
        minx, maxx = min(minx, lon), max(maxx, lon)
        miny, maxy = min(miny, lat), max(maxy, lat)

    margin = 5
    extent = (minx - margin, miny - margin, maxx + margin, maxy + margin)

    m = navigation_area2(ax, date, travelDays, startEnd, extent)
    labels = [None, 'Reference']
    vMin, dV = 8.8, 15.2 - 8.8

    # Do for front and ref front
    for idx, f in enumerate([front, refFront]):
        inds = [ind for ind in f]
        if convexHull:
            fits = [fit.values for fit in f.keys]
            hull = spatial.ConvexHull(fits)
            inds = np.take(inds, hull.vertices, 0)

        label = labels[idx]
        cmap = 'k' if idx == 1 else cmap0
        for j, ind in enumerate(inds):
            if idx == 1 and j > 0:
                continue

            alpha = 1 if idx == 1 else 0.9
            waypoints, speeds = zip(*ind)
            for i, leg in enumerate(zip(waypoints[:-1], waypoints[1:])):
                color = cmap((speeds[i] - vMin) / dV) if cmap != 'k' else cmap
                label = None if i > 0 else label
                m.drawgreatcircle(leg[0][0], leg[0][1], leg[1][0], leg[1][1], label=label, linewidth=1,
                                  alpha=alpha, color=color, zorder=3)


def plot_weather2(m, dateTime, travelDays, lonStart, lonDest):
    hourPeriod = 24 // 6
    maxTravelDays = int(np.ceil(travelDays))
    print('weather travel days', travelDays)

    # Get wind data
    retriever = WindDataRetriever(nDays=maxTravelDays, startDate=dateTime)
    ds = retriever.get_data(forecast=False)

    # Create lon indices
    res = 0.5
    startEndIndex = [int(round((lon + 180) / res)) for lon in [lonStart, lonDest]]
    lon0, lon1 = [0 if idx == 720 else idx for idx in startEndIndex]
    if dateTime.year == 2011 and dateTime.month == 5:
        nSlices = maxTravelDays * hourPeriod + 1
        slicesLeft = int(round(nSlices * (180 - lonStart) / ((180 - lonStart) + abs(lonDest + 180))))
        slicesRight = nSlices - slicesLeft
        lonIndicesLeft = np.round(np.linspace(lon0, 719, slicesLeft)).astype(int)
        lonIndicesRight = np.round(np.linspace(0, lon1, slicesRight)).astype(int)
        lonPairs = list(zip(lonIndicesLeft[:-1], lonIndicesLeft[1:])) + list(zip(lonIndicesRight[:-1], lonIndicesRight[1:]))
        print(lonPairs)
    else:
        lonIndices = np.linspace(lon0, lon1, maxTravelDays * hourPeriod + 1).astype(int)
        lonPairs = list(zip(lonIndices[:-1], lonIndices[1:]))

    # Create date indices
    dateIndices = np.round(np.linspace(0, travelDays * hourPeriod - 1, maxTravelDays * hourPeriod)).astype(int)

    # Initialize variables for contour
    lons, lats = np.linspace(-180, 179.5, 720), np.linspace(-90, 90, 361)
    x, y = m(*np.meshgrid(lons, lats))

    for i, (dateInd, lonPair) in enumerate(zip(dateIndices, lonPairs)):
        lonMin, lonMax = min(lonPair), max(lonPair) + 1
        if lonPairs[0][0] < lonPairs[1][1]:
            if i == 0:
                lonMin = 0
            elif i == len(dateIndices) - 1:
                lonMax = -1
        elif lonPair[0] > lonPair[1]:
            if i == len(dateIndices) - 1:
                lonMin = 0
            elif i == 0:
                lonMax = -1
        lonSlice = slice(lonMin, lonMax)
        BN = ds[0, dateInd, :, lonSlice]
        xSlice, ySlice = x[:, lonSlice], y[:, lonSlice]
        try:
            m.contourf(xSlice, ySlice, BN, vmin=0, vmax=12, cmap=cm.get_cmap('Purples', 13))
        except ValueError:
            print('something wrong')


_frontsDict = get_fronts_dict()

# compute_metrics(_frontsDict)
# plot_all_routes(_frontsDict, save=True, convexHull=True)
plot_group_routes(_frontsDict, save=True, convexHull=True)
# plot_all_fronts(_frontsDict, save=True, convexHull=True)
# save_fronts(_frontsDict)
plt.show()