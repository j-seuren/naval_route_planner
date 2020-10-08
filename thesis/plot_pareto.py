# import case_studies.plot_results as plot_results
import itertools
import main
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np
import os
import pickle
import pprint
import tikzplotlib

# from copy import deepcopy
from data_config.wind_data import WindDataRetriever
from datetime import datetime
from deap import tools
from pathlib import Path
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import font_manager as fm
from matplotlib import cm, patches
from matplotlib.collections import PatchCollection
from scipy import spatial
fontPropFP = "C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/tex-gyre-pagella.regular.otf"
fontProp = fm.FontProperties(fname=fontPropFP)

DIR = Path('D:/')

pp = pprint.PrettyPrinter()


class MergedPlots:
    def __init__(self, directory, date, experiment, contains):
        os.chdir(directory)
        self.files = [file for file in os.listdir() if contains in file]
        pp.pprint(self.files)

        self.planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)
        self.experiment = experiment
        self.fn = '{}'.format(datetime.now().strftime('%m%d%H%M'))

        self.vMin, vMax = min(self.planner.vessel.speeds), max(self.planner.vessel.speeds)
        self.dV = vMax - self.vMin

        self.initialLabel = 'not set'

        self.date = date

        # Fronts and routes
        self.outFiles = []
        for i, rawFN in enumerate(self.files):
            with open(rawFN, 'rb') as f:
                rawList = pickle.load(f)
            updateDict = {experiment: 1.5593} if experiment == 'eca' else {experiment: date}
            updateDict = None if experiment == 'bathymetry' else updateDict
            proc, raw = self.planner.post_process(rawList[0], updateEvaluator=updateDict)
            fronts, hulls = zip(*[get_front(_front, self.planner, experiment, date) for _front in raw['fronts']])

            self.outFiles.append({'fronts': fronts, 'hulls': hulls, 'proc': proc, 'raw': raw, 'filename': rawFN})

    def merged_pareto(self, save=False):
        frontFig, frontAx = plt.subplots()
        frontAx.set_xlabel('Travel time [d]', fontproperties=fontProp)
        frontAx.set_ylabel('Fuel costs [x1000 USD]', fontproperties=fontProp)
        cycleFront = frontAx._get_lines.prop_cycler

        if self.experiment == 'eca':
            labels = ['Incl. ECA (E)', 'Excl. ECA (R)']
            C, V = '', ''
            R0 = 'E'
        elif self.experiment == 'bathymetry':
            labels = ['Incl. bathymetry (B)', 'Excl. bathymetry (R)']
            C, V = '', ''
            R0 = 'B'
        else:
            labels = ['Constant speed - ref. (CR)', 'Constant speed (C)',
                      'Variable speed - ref. (VR)', 'Variable speed (V)']
            C, V = 'C', 'V'
            R0 = ''
        labelString, next_color = '', 'black'
        for file in self.outFiles:
            S = C if C in file['filename'] else V
            R = 'R' if 'R' in file['filename'] else R0
            newLabelString = '({}{})'.format(S, R)
            noLabel = True if labelString == newLabelString else False
            labelString = newLabelString
            label = None if noLabel else [label for label in labels if labelString in label][0]
            print(label)

            next_color = next_color if noLabel else next(cycleFront)['color']
            # Plot front
            (marker, s, zorder) = ('s', 5, 2) if 'C' in labelString else ('o', 1, 1)
            for front in file['fronts']:
                travelTimes, fuelCosts = zip(*list(front.keys()))
                frontAx.scatter(travelTimes, fuelCosts,
                                color=next_color, marker=marker, s=s, zorder=zorder, label=label)
            for hull in file['hulls']:
                travelTimes, fuelCosts = zip(*list(hull.keys()))
                frontAx.scatter(travelTimes, fuelCosts, color=next_color, marker='x', zorder=zorder)

        frontAx.legend(prop=fontProp)
        plt.xticks(fontproperties=fontProp)
        plt.yticks(fontproperties=fontProp)

        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)

        if save:
            frontFig.savefig('{}_front_merged'.format(self.fn), dpi=300)
            frontFig.savefig('{}_front_merged.pdf'.format(self.fn), bbox_inches='tight', pad_inches=0)
            tikzplotlib.save("{}_front_merged.tex".format(self.fn))

    def colorbar(self, ax):
        cmap = cm.get_cmap('jet', 12)
        cmapList = [cmap(i) for i in range(cmap.N)][1:-1]
        cmap = cl.LinearSegmentedColormap.from_list('Custom cmap', cmapList, cmap.N - 2)
        # Create color bar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)  # Colorbar axis
        cb = plt.colorbar(sm, norm=plt.Normalize(vmin=self.vMin, vmax=self.vMin + self.dV), cax=cax)
        nTicks = 6
        cb.ax.set_yticklabels(['%.1f' % round(self.vMin + i * self.dV / (nTicks - 1), 1) for i in range(nTicks)],
                              fontproperties=fontProp)
        cb.set_label('Nominal speed [knots]', rotation=270, labelpad=15, fontproperties=fontProp)

        return cmap

    def plot_ind(self, ind, m, label=None, width=1, line='solid', color='gray', alpha=0.5, cmap=None):
        waypoints, speeds = zip(*ind)
        for i, leg in enumerate(zip(waypoints[:-1], waypoints[1:])):
            color = cmap((speeds[i] - self.vMin) / self.dV) if cmap and speeds[i] is not None else color
            label = None if i > 0 else label
            m.drawgreatcircle(leg[0][0], leg[0][1], leg[1][0], leg[1][1], label=label, linestyle=line, linewidth=width,
                              alpha=alpha, color=color, zorder=3)

    def merged_routes(self, zoom=1, it=0, initial=False, intervalRoutes=None, colorbar=False, alpha=0.5, save=False, hull=True):
        routeFig, routeAx = plt.subplots()
        w, h = routeFig.get_size_inches()
        routeFig.set_size_inches(w * zoom, h * zoom)
        cycleRoute = routeAx._get_lines.prop_cycler
        cmap = self.colorbar(routeAx) if colorbar else None

        # Plot navigation area
        if self.experiment == 'current':
            cData = self.planner.evaluator.currentOp.data
            lons0 = np.linspace(-179.875, 179.875, 1440)
            lats0 = np.linspace(-89.875, 89.875, 720)
            currentDict = {'u': cData[0, 0], 'v': cData[1, 0], 'lons': lons0, 'lats': lats0}
            m = navigation_area(routeAx, self.outFiles[it]['proc'], initial, current=currentDict)
        elif self.experiment == 'weather':
            m = navigation_area(routeAx, self.outFiles[it]['proc'], initial, weather=self.date)
        else:
            m = navigation_area(routeAx, self.outFiles[it]['proc'], initial,
                                eca=self.experiment == 'eca',
                                bathymetry=self.experiment == 'bathymetry')

        if self.experiment == 'eca':
            labels = ['Incl. ECA (E)', 'Excl. ECA (R)']
            C, V = '', ''
            R0 = 'E'
        elif self.experiment == 'bathymetry':
            labels = ['L', 'S',
                      'RL', 'RS']
            C, V = '', ''
            R0 = ''
        else:
            labels = ['Constant speed - ref. (CR)', 'Constant speed (C)',
                      'Variable speed - ref. (VR)', 'Variable speed (V)']
            C, V = 'C', 'V'
            R0 = ''

        labelString, appLabels = '', [' - time', ' - cost']
        # Plot initial routes
        if initial:
            for initRoute in self.outFiles[-1]['raw']['initialRoutes']:
                for subInitRoute in initRoute['route']:
                    for objRoute in subInitRoute.values():
                        self.initialLabel = 'Initial' if self.initialLabel == 'not set' else None
                        self.plot_ind(objRoute, m,  alpha=1, label=self.initialLabel)
        label = None
        for file in self.outFiles:
            S = C if 'C' in file['filename'] else V
            R = 'R' if 'R' in file['filename'] else R0
            if labelString == '{}{}'.format(S, R):  # Plot constant speed profile only once
                continue
            labelString = '{}{}'.format(S, R)

            # if not intervalRoutes:  # or R == 'R':  # Plot route responses
            #     color = next(cycleRoute)['color']
            #     for r, route in enumerate(file['proc']['routeResponse']):
            #         if self.experiment != 'bathymetry' and r > 1:
            #             break
            #         elif r % 2 != 0:
            #             continue
            #         label0 = '{}{}'.format(rLabel, appLabels[r]) if self.experiment != 'bathymetry' and R != 'R' else rLabel
            #         label = label0 if label != label0 else None
            #         ind = [((leg['lon'], leg['lat']), leg['speed']) for leg in route['waypoints']]
            #         line = 'dashed' if r > 0 and self.experiment != 'bathymetry' else 'solid'
            #         self.plot_ind(ind, m, label=label, color=color, line=line)

            fronts = file['hulls'] if hull else file['fronts']
            shortLong = iter(['S', 'L'])
            for front in fronts:
                color = next(cycleRoute)['color'] if cmap is None else 'k'
                if self.experiment == 'bathymetry':
                    bLabel = '{}{}'.format(labelString, next(shortLong))
                    try:
                        rLabel = [la for la in labels if bLabel in la][0]
                    except IndexError:
                        rLabel = bLabel
                    ind = list(front.values())[0] if 'R' in rLabel else list(front.values())[-1]
                    label = rLabel if label != rLabel and cmap is None else None
                    self.plot_ind(ind, m, label=label, color=color, width=1, alpha=alpha, cmap=cmap)
                else:
                    for fit, ind in front.items():
                        rLabel = labelString
                        if intervalRoutes and not intervalRoutes[0] < fit[0] < intervalRoutes[1]:
                            continue
                        label = rLabel if label != rLabel and cmap is None else None
                        self.plot_ind(ind, m, label=label, color=color, alpha=alpha, cmap=cmap)
        if cmap is None or self.initialLabel is None:
            routeAx.legend(loc='upper right', prop=fontProp)

        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)

        if save:
            routeFig.savefig('{}_route_merged'.format(self.fn), dpi=300)
            routeFig.savefig('{}_route_merged.pdf'.format(self.fn), bbox_inches='tight', pad_inches=.01)
            # tikzplotlib.save("{}_route_merged.tex".format(self.fn))


def update(population):
    front = {}

    for ID, ind in population:
        nonDominated = True
        for _, otherInd in population:
            dominates = True
            for otherVal, indVal in zip(otherInd, ind):
                if otherVal >= indVal:
                    dominates = False
            if dominates:
                nonDominated = False
                break
        if nonDominated:
            ID = (tuple(ID[0]), tuple(ID[1]))
            front[ID] = np.array([ind])
    return front

#
# def get_front_multiple_routes(frontIn, planner, experiment, date, getConvexHull):
#     current = True if experiment == 'current' else False
#     weather = True if experiment == 'weather' else False
#     planner.evaluator.set_classes(current, weather, date, 10)
#
#     subObjVals, deapFronts = [], []
#     for front in frontIn:
#         newFits = [planner.evaluator.evaluate(ind, revert=False, includePenalty=False) for ind in front]
#         for ind, fit in zip(front.items, newFits):
#             ind.fitness.values = fit
#         subObjVals.append(np.array(newFits))
#
#     # First concatenate sub path objective values
#     indDict = {x: sum(x) for x in itertools.product(*subObjVals)}
#     indDict = update(indDict)
#     if getConvexHull:
#         indValues = list(indDict.keys())
#         indDict = indDict[spatial.ConvexHull(indDict).vertices]
#
#     return indDict, frontIn


def get_front(frontIn, planner, experiment, date):
    current = True if experiment == 'current' else False
    weather = True if experiment == 'weather' else False
    planner.evaluator.set_classes(current, weather, date, 10)

    objVals, fronts = [], []
    for front in frontIn:
        # newFits = [planner.evaluator.evaluate(ind, revert=False, includePenalty=False) for ind in front]
        newFits = [ind.fitness.values for ind in front]
        print(newFits)
        fronts.append({newFits[f]: ind for f, ind in enumerate(front.items)})
        objVals.append(np.array(newFits))

    concatObjVals = [(x, sum(x)) for x in itertools.product(*objVals)]
    # concatFront0 = update(concatObjVals)
    inds = []
    for ID, fit in concatObjVals:
        ind = []
        for i, subFit in enumerate(ID):
            ind.extend(fronts[i][tuple(subFit)])
        ind = main.creator.Individual(ind)
        ind.fitness.values = fit
        inds.append(ind)


    # front = frontIn[0]
    # newFits = [planner.evaluator.evaluate(ind, revert=False, includePenalty=False) for ind in front]
    # for ind, fit in zip(front.items, newFits):
    #     ind.fitness.values = fit

    newFront = tools.ParetoFront()
    newFront.update(inds)

    frontOut = {ind.fitness.values: ind for ind in newFront}
    print('frontsize', len(frontOut))

    if len(frontOut) > 1:
        hull = spatial.ConvexHull(list(frontOut.keys()))
        vertices = hull.vertices
        hullPoints = hull.points[vertices]
        convexHull = {tuple(point): frontOut[tuple(point)] for point in hullPoints}
        print('hullsize', len(convexHull))
    else:
        convexHull = frontOut

    return frontOut, convexHull


def set_extent(proc, initial):
    minx, miny, maxx, maxy = 180, 85, -180, -85
    if proc:
        for route in proc['routeResponse']:
            lons, lats = zip(*[(leg['lon'], leg['lat']) for leg in route['waypoints']])
            for x, y in zip(lons, lats):
                minx, miny = min(minx, x), min(miny, y)
                maxx, maxy = max(maxx, x), max(maxy, y)
        if initial:
            for initRoute in proc['initialRoutes']:
                for subInitRoute in initRoute['route']:
                    for objRoute in subInitRoute.values():
                        lons, lats = zip(*[leg[0] for leg in objRoute])
                        for x, y in zip(lons, lats):
                            minx, miny = min(minx, x), min(miny, y)
                            maxx, maxy = max(maxx, x), max(maxy, y)
        margin = 0.1 * max((maxx - minx), (maxy - miny)) / 2
        return max(minx - margin, -180), max(miny - margin, -90), min(maxx + margin, 180), min(maxy + margin, 90)
    else:
        return -180, -80, 180, 80


def currents(ax, m, uin, vin, lons, lats, extent):
    dLon = extent[2] - extent[0]
    dLat = extent[3] - extent[1]

    vLon = int(dLon * 4)
    vLat = int(dLat * 4)
    uRot, vRot, x, y = m.transform_vector(uin, vin, lons, lats, vLon, vLat, returnxy=True)
    Q = m.quiver(x, y, uRot, vRot, np.hypot(uRot, vRot), pivot='mid', width=0.002, headlength=4, cmap='PuBu', scale=90,
                 ax=ax)
    ax.quiverkey(Q, 0.4, 1.1, 2, r'$2$ knots', labelpos='E')


def weather_contour(m, dateTime, travelDays, lonStart, lonEnd):
    hourPeriod = 24//6
    travelDaysCeil = int(np.ceil(travelDays))
    print(travelDays)
    print(travelDays * 4)

    # Get wind data
    retriever = WindDataRetriever(nDays=travelDaysCeil, startDate=dateTime)
    ds = retriever.get_data(forecast=False)

    # Create lon indices
    resolution = 0.5
    lonStart_idx = int(round((lonStart + 180) / resolution))
    lonStart_idx = 0 if lonStart_idx == 720 else lonStart_idx
    lonEnd_idx = int(round((lonEnd + 180) / resolution))
    lonEnd_idx = 0 if lonEnd_idx == 720 else lonEnd_idx
    (lonS, lonT) = (lonStart_idx, lonEnd_idx) if lonStart_idx < lonEnd_idx else (lonEnd_idx, lonStart_idx)
    lonIndices = np.linspace(lonS, lonT, travelDaysCeil * 4 + 1).astype(int)
    print(len(lonIndices), '\n', lonIndices)
    lonPairs = zip(lonIndices[:-1], lonIndices[1:])

    # Create date indices
    dateIndices = np.linspace(0, travelDays * hourPeriod, travelDaysCeil * 4).astype(int)
    print(len(dateIndices), '\n', dateIndices)

    # Initialize variables for contourf
    lons, lats = np.linspace(-180, 179.5, 720), np.linspace(-90, 89.5, 360)
    x, y = m(*np.meshgrid(lons, lats))
    maxWind, C1 = 0, None
    Cs = [None for _ in range(len(dateIndices))]

    for i, (dateInd, lonPair) in enumerate(zip(dateIndices, lonPairs)):
        lon0, lon1 = min(lonPair), max(lonPair) + 1
        BNarr = ds[0, dateInd, :-1, lon0:lon1]
        xx, yy = x[:, lon0:lon1], y[:, lon0:lon1]
        Cs[i] = m.contourf(xx, yy, BNarr, vmin=0, vmax=12, cmap=cm.get_cmap('jet', 12))
        if np.max(BNarr) > maxWind:
            C1 = Cs[i]

    cb = m.colorbar(C1, norm=plt.Normalize(vmin=0, vmax=12), size=0.2, pad=0.2, location='bottom')
    vDif = 12
    nTicks = 13
    cb.ax.set_xticklabels(['%.f' % round(i * vDif / (nTicks - 1), 1) for i in range(nTicks)], fontsize=8)
    cb.set_label('Wind [BFT]', fontproperties=fontProp)


def navigation_area(ax, proc, initial, eca=False, current=None, weather=None, bathymetry=False):
    extent = set_extent(proc, initial)
    left, bottom, right, top = extent
    m = Basemap(projection='merc', resolution='i', llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left,
                urcrnrlon=right, ax=ax)
    m.drawmapboundary(color='black')
    m.fillcontinents(color='lightgray', lake_color='lightgray', zorder=2)
    m.drawcoastlines()

    if current:
        uin, vin = current['u'], current['v']
        lons, lats = current['lons'], current['lats']
        currents(ax, m, uin, vin, lons, lats, extent)

    if eca:
        m.readshapefile(Path(DIR / "data/eca_reg14_sox_pm/eca_reg14_sox_pm").as_posix(), 'eca_reg14_sox_pm',
                        drawbounds=False)
        ps = [patches.Polygon(np.array(shape), True) for shape in m.eca_reg14_sox_pm]
        ax.add_collection(PatchCollection(ps, facecolor='green', alpha=0.5, zorder=3))

    if bathymetry:
        # m = Basemap(projection='lcc', resolution=None, llcrnrlat=bottom, urcrnrlat=top,
        #         llcrnrlon=left, urcrnrlon=right, lat_0=(top+bottom)/2, lon_0=(right+left)/2, ax=ax)
        # m.etopo()
        m.readshapefile(Path(DIR / "data/bathymetry_200m/ne_10m_bathymetry_K_200").as_posix(),
                        'ne_10m_bathymetry_K_200', drawbounds=False)
        ps = [patches.Polygon(np.array(shape), True) for shape in m.ne_10m_bathymetry_K_200]
        ax.add_collection(PatchCollection(ps, facecolor='white', zorder=2))
        m.drawmapboundary(color='black', fill_color='khaki')

    if weather:
        minTravelDays = proc['routeResponse'][0]['travelTime']
        waypoints = proc['routeResponse'][0]['waypoints']
        lon0, lon1 = waypoints[0]['lon'], waypoints[-1]['lon']
        try:
            maxTravelDays = proc['routeResponse'][1]['travelTime']
        except IndexError:
            maxTravelDays = minTravelDays
        avgTravelDays = (minTravelDays + maxTravelDays) / 2
        weather_contour(m, weather, avgTravelDays, lon0, lon1)

    return m


if __name__ == '__main__':
    #  BATHYMETRY
    # _directory = 'C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/5. Thesis/bathymetry results'
    # mergedPlots = MergedPlots(_directory, datetime(2014, 11, 25), experiment='bathymetry', contains='VC')
    # mergedPlots.merged_routes(zoom=1.5, initial=True, colorbar=False, alpha=1, save=True, hull=False)

    #  ECA
    # _directory = 'C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/5. Thesis/eca results/Flo'
    # contains='FloSa'

    _directory = 'D:/output/weather/WTH/NSGA2_varSP_BFalse_ECA1.0/5/raw'
    mergedPlots = MergedPlots(_directory, datetime(2011, 5, 28), experiment='weather', contains='PH')

    # mergedPlots.merged_pareto(save=False)
    mergedPlots.merged_routes(zoom=1.2, initial=False, colorbar=True, alpha=0.5, save=False, hull=True)

    plt.show()