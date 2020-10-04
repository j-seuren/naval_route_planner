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
from datetime import datetime
from pathlib import Path
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import font_manager as fm
from matplotlib import cm, patches
from matplotlib.collections import PatchCollection

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


        # Fronts and routes
        self.outFiles = []
        for i, rawFN in enumerate(self.files):
            with open(rawFN, 'rb') as f:
                rawList = pickle.load(f)
            updateDict = {experiment: 1.5593} if experiment == 'eca' else {experiment: date}
            proc, raw = self.planner.post_process(rawList[0], updateEvaluator=updateDict)
            fronts = [get_front(_front, self.planner, experiment, date) for _front in raw['fronts']]

            self.outFiles.append({'fronts': fronts, 'proc': proc, 'raw': raw, 'filename': rawFN})

    def merged_pareto(self, save=False):
        frontFig, frontAx = plt.subplots()
        frontAx.set_xlabel('Travel time [d]', fontproperties=fontProp)
        frontAx.set_ylabel('Fuel costs [x1000 USD/t]', fontproperties=fontProp)
        cycleFront = frontAx._get_lines.prop_cycler

        labels = ['Constant speed - ref. (CR)', 'Constant speed (C)',
                  'Variable speed - ref. (VR)', 'Variable speed (V)']
        labelString, next_color = '', 'black'
        for file in self.outFiles:
            S = 'C' if 'C' in file['filename'] else 'V'
            R = 'R' if 'R' in file['filename'] else ''
            newLabelString = '({}{})'.format(S, R)
            noLabel = True if labelString == newLabelString else False
            labelString = newLabelString
            label = None if noLabel else [label for label in labels if labelString in label][0]
            print(label)

            next_color = next_color if noLabel else next(cycleFront)['color']
            # Plot front
            (marker, s, zorder) = ('s', 5, 2) if 'C' in labelString else ('o', 1, 1)
            for front in file['fronts']:
                frontAx.scatter(front[:, 0], front[:, 1],
                                color=next_color, marker=marker, s=s, zorder=zorder, label=label)

        frontAx.legend(prop=fontProp)

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
                              fontsize=8)
        cb.set_label('Nominal speed [knots]', rotation=270, labelpad=15)

        return cmap

    def merged_routes(self, allRoutes=False, colorbar=False, save=False):
        routeFig, routeAx = plt.subplots()
        cycleRoute = routeAx._get_lines.prop_cycler
        cmap = self.colorbar(routeAx) if colorbar else None

        # Plot navigation area
        if self.experiment == 'current':
            cData = self.planner.evaluator.currentOperator.data
            lons0 = np.linspace(-179.875, 179.875, 1440)
            lats0 = np.linspace(-89.875, 89.875, 720)
            currentDict = {'u': cData[0, 0], 'v': cData[1, 0], 'lons': lons0, 'lats': lats0}
        else:
            currentDict = None
        m = navigation_area(routeAx, self.outFiles[0]['proc'], eca=self.experiment == 'eca', current=currentDict)
        labelString, labels = '', [' - time', ' - cost']
        for file in self.outFiles:
            S = 'C' if 'C' in file['filename'] else 'V'
            R = 'R' if 'R' in file['filename'] else ''
            if labelString == '{}{}'.format(S, R):  # Plot constant speed profile only once
                continue
            labelString = '{}{}'.format(S, R)

            if not allRoutes:  # Plot route responses
                color = next(cycleRoute)['color']
                for r, route in enumerate(file['proc']['routeResponse']):
                    if r > 1:
                        break
                    label = '{}{}'.format(labelString, labels[r])
                    waypoints = [(leg['lon'], leg['lat']) for leg in route['waypoints']]
                    legs = zip(waypoints[:-1], waypoints[1:])
                    line = 'dashed' if r > 0 else 'solid'
                    for i, leg in enumerate(legs):
                        label = None if i > 0 else label
                        m.drawgreatcircle(leg[0][0], leg[0][1], leg[1][0], leg[1][1],
                                          label=label, linestyle=line, linewidth=1, alpha=0.5, color=color, zorder=3)
            else:
                for fronts in file['raw']['fronts']:
                    for front in fronts:
                        for ind in front:
                            waypoints, speeds = zip(*ind)
                            for i, leg in enumerate(zip(waypoints[:-1], waypoints[1:])):
                                color = cmap((speeds[i] - self.vMin) / self.dV) if cmap else 'k'
                                m.drawgreatcircle(leg[0][0], leg[0][1], leg[1][0], leg[1][1],
                                                  linewidth=0.5, color=color, zorder=3)

        routeAx.legend(loc='lower right', prop=fontProp)

        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)

        if save:
            routeFig.savefig('{}_route_merged'.format(self.fn), dpi=300)
            routeFig.savefig('{}_route_merged.pdf'.format(self.fn), bbox_inches='tight', pad_inches=.5)
            # tikzplotlib.save("{}_route_merged.tex".format(self.fn))


def update(population):
    front = np.empty((0, 2))

    for ind in population:
        nonDominated = True
        for otherInd in population:
            dominates = True
            for otherVal, indVal in zip(otherInd, ind):
                if otherVal >= indVal:
                    dominates = False
            if dominates:
                nonDominated = False
                break
        if nonDominated:
            front = np.append(front, np.array([ind]), axis=0)
    return front


def get_front(frontIn, planner, experiment, date):
    current = True if experiment == 'current' else False
    weather = True if experiment == 'weather' else False
    planner.evaluator.set_classes(current, weather, date, 10)

    subObjVals = []
    for front in frontIn:
        objValues = [planner.evaluator.evaluate(ind, revert=False, includePenalty=False) for ind in front]
        subObjVals.append(np.array(objValues))

    # First concatenate sub path objective values
    objVals = [sum(x) for x in itertools.product(*subObjVals)]
    return update(objVals)


def set_extent(proc):
    minx, miny, maxx, maxy = 180, 85, -180, -85
    if proc:
        for route in proc['routeResponse']:
            lons, lats = zip(*[(leg['lon'], leg['lat']) for leg in route['waypoints']])
            for x, y in zip(lons, lats):
                minx, miny = min(minx, x), min(miny, y)
                maxx, maxy = max(maxx, x), max(maxy, y)
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


def navigation_area(ax, proc, eca=False, current=None):
    extent = set_extent(proc)
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

    return m


# def merged_plots(directory, experiment, contains=''):
#
#     planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)
#
#     os.chdir(directory)
#     files = [file for file in os.listdir() if contains in file]
#
#     pp.pprint(files)
#
#     routeFig, routeAx = plt.subplots()
#     cycleRoute = routeAx._get_lines.prop_cycler
#
#     frontFig, frontAx = plt.subplots()
#     frontAx.set_xlabel('Travel time [d]', fontproperties=fp)
#     frontAx.set_ylabel('Fuel costs [x1000 USD/t]', fontproperties=fp)
#     cycleFront = frontAx._get_lines.prop_cycler
#
#     raws = []
#     fronts = []
#
#     frontLabels = ['Constant speed - ref. (CR)', 'Constant speed (C)', 'Variable speed - ref. (VR)', 'Variable speed (V)']
#     labels = ['CR', 'C', 'VR', 'V']
#     ii = 0
#     for i, rawFN in enumerate(files):
#         with open(rawFN, 'rb') as f:
#             rawList = pickle.load(f)
#         raw = rawList[0]
#         raws.append(rawList[0])
#         fronts.append(raw['fronts'])
#         date = datetime(2015, 3, 15) if '2015' in rawFN else datetime(2014, 2, 15)
#
#         # Plot front
#         (marker, s, zorder) = ('s', 5, 2) if i < 4 else ('o', 1, 1)
#         if i != 1 and i != 3:
#             next_color = next(cycleFront)['color']
#         else:
#             ii += 1
#         for _front in raw['fronts']:
#             front = get_front(_front, planner, experiment, date)
#             if i == 1 or i == 3:
#                 frontAx.scatter(front[:, 0], front[:, 1], color=next_color, marker=marker, s=s, zorder=zorder)
#             else:
#                 frontAx.scatter(front[:, 0], front[:, 1],
#                                 color=next_color,
#                                 marker=marker,
#                                 s=s,
#                                 zorder=zorder,
#                                 label=frontLabels[ii])
#
#         proc, raw = planner.post_process(raw, inclEnvironment={experiment: date})
#         for pr in proc['routeResponse']:
#             pr = deepcopy(pr)
#             pr['waypoints'] = None
#             pp.pprint(pr)
#
#         # Plot navigation area
#         if i == 0:
#             if experiment == 'current':
#                 cData = planner.evaluator.currentOperator.data
#                 lons0 = np.linspace(-179.875, 179.875, 1440)
#                 lats0 = np.linspace(-89.875, 89.875, 720)
#                 currentDict = {'u': cData[0, 0], 'v': cData[1, 0], 'lons': lons0, 'lats': lats0}
#             else:
#                 currentDict = None
#             m = navigation_area(routeAx, proc, current=currentDict)
#
#         # Plot route responses
#         color = next(cycleRoute)['color']
#         routeLabels = [' - time', ' - cost']
#         for r, route in enumerate(proc['routeResponse']):
#             if r > 1:
#                 continue
#             label = '{}{}'.format(labels[ii], routeLabels[r])
#             route = [((leg['lon'], leg['lat']), leg['speed']) for leg in route['waypoints']]
#             waypoints = [leg[0] for leg in route]
#             arcs = zip(waypoints[:-1], waypoints[1:])
#             line = 'dashed' if r > 0 else 'solid'
#             for aIdx, a in enumerate(arcs):
#                 if aIdx == 0 and (i != 1 and i != 3):
#                     m.drawgreatcircle(a[0][0], a[0][1], a[1][0], a[1][1], label=label, linestyle=line, linewidth=1,
#                                       color=color, zorder=3)
#                 else:
#                     m.drawgreatcircle(a[0][0], a[0][1], a[1][0], a[1][1], linestyle=line, linewidth=1, color=color,
#                                       zorder=3)
#
#     frontAx.legend(prop=fp)
#     routeAx.legend(loc='lower right', prop=fp)
#
#     # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     # plt.margins(0, 0)
#
#     fn = '{}'.format(datetime.now().strftime('%m%d%H%M'))
#     frontFig.savefig('{}_front_merged'.format(fn), dpi=300)
#     routeFig.savefig('{}_route_merged'.format(fn), dpi=300)
#     frontFig.savefig('{}_front_merged.pdf'.format(fn), pad_inches=0.1)
#     routeFig.savefig('{}_route_merged.pdf'.format(fn), bbox_inches='tight', pad_inches=.5)


_directory = 'C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/5. Thesis/Current results/Gulf/Gulf0'
_experiment = 'current'
mergedPlots = MergedPlots(_directory, datetime(2014, 11, 25), _experiment, contains='31')

mergedPlots.merged_pareto(save=True)
mergedPlots.merged_routes(colorbar=True, save=True)

plt.show()

#
#
#
#
# SPEED = 'constant'  # 'constant' or 'var'
# # currentDir = Path('C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/5. Thesis/Current results/{}_NSGA'.format(SPEED))
# currentDir = Path('C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/5. Thesis/Current results/raws'.format(SPEED))
# os.chdir(currentDir)
#
# fileList = os.listdir()
# fileList = [file for file in fileList if 'KT' in file]
#
# pp = pprint.PrettyPrinter()
# pp.pprint(fileList)
#
# DIR = Path('D:/')
# speedOps = ['insert', 'move', 'delete'] if SPEED == 'constant' else ['insert', 'move', 'speed', 'delete']
# par = {'mutationOperators': speedOps}
# ECA_F = 1
# DEPTH = False
# PLANNER = main.RoutePlanner(inputParameters=par, bathymetry=DEPTH, ecaFactor=ECA_F,
#                             criteria={'minimalTime': True, 'minimalCost': True})
#
# exp = 'current'
#
# for i, rawFile in enumerate(fileList):
#     print(rawFile)
#     depDate = datetime(2015, 3, 15) if '2015' in rawFile else datetime(2014, 9, 17)
#
#     with open(rawFile, 'rb') as f:
#         rawList = pickle.load(f)
#
#     for j, raw in enumerate(rawList):
#         proc, raw = PLANNER.post_process(raw, inclEnvironment={exp: depDate})
#         for response in proc['routeResponse']:
#             print('distance', response['distance'],
#                   'fuelCost', response['fuelCost'],
#                   'travelTime', response['travelTime'],
#                   'fitValues', response['fitValues'], )
#         statisticsPlotter = plot_results.StatisticsPlotter(raw, DIR=DIR)
#         frontFig, _ = statisticsPlotter.plot_fronts()
#         statsFig, _ = statisticsPlotter.plot_stats()
#
#         if exp == 'current':
#             cData = PLANNER.evaluator.currentOperator.data
#             lons0 = np.linspace(-179.875, 179.875, 1440)
#             lats0 = np.linspace(-89.875, 89.875, 720)
#             currentDict = {'u': cData[0, 0], 'v': cData[1, 0], 'lons': lons0, 'lats': lats0}
#         else:
#             currentDict = None
#
#         weatherDate = depDate if exp == 'weather' else None
#         routePlotter = plot_results.RoutePlotter(DIR, proc, rawResults=raw, vessel=PLANNER.vessel)
#
#         routeFig, ax = plt.subplots()
#         ax = routePlotter.results(ax, initial=False, ecas=False, bathymetry=DEPTH, nRoutes=None,
#                                   weatherDate=weatherDate, current=currentDict, colorbar=True, wps=False)
#
#         fn = '{}_v{}.png'.format(i, j)
#         frontFig.savefig('front/' + fn, dpi=300)
#         statsFig.savefig('stats/' + fn, dpi=300)
#         routeFig.savefig('route/' + fn, dpi=300)
#
#         plt.close('all')

