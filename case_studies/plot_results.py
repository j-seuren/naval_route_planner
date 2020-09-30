import evaluation
import itertools
import matplotlib.colors as cl
import matplotlib.pyplot as plt
import numpy as np

from data_config.wind_data import WindDataRetriever
from matplotlib import cm, patches
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# plt.rcParams.update({
#     # "text.usetex": True,
#     "font.size": 10,
#     "font.family": "serif",
#     "font.serif": ["TeX Gyre Pagella"],
# })


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


class StatisticsPlotter:
    def __init__(self, rawResults, DIR):
        self.rawResults = rawResults
        self.DIR = DIR

    def plot_stats(self):
        logs = self.rawResults['logs']
        fig, axs = plt.subplots(1, len(logs))
        for i, log in enumerate(logs):
            if len(logs) == 1:
                statistics(log, axs)
            else:
                statistics(log, axs[i])
        return fig, axs

    def plot_fronts(self):
        fronts = self.rawResults['fronts']
        fig, axs = plt.subplots(1, len(fronts))
        for i, front in enumerate(fronts):
            concatFront = concatenated_front(front)
            ax = axs if len(fronts) == 1 else axs[i]

            # Plot front
            ax.scatter(concatFront[:, 0], concatFront[:, 1], c="b", s=1)
            ax.axis("tight")
            ax.grid()
            ax.set_xlabel('Travel time [days]')
            ax.set_ylabel('Fuel costs [x1000 EUR per tonne]')

        return fig, axs


def statistics(log, ax, title=None):
    for subLog in log:
        genNumber = subLog.select("gen")
        fitMin = subLog.chapters["fitness"].select("min")
        avgSize = subLog.chapters["size"].select("avg")

        ax.title.set_text(title)
        # Plot minimum Fitness
        line1 = ax.plot(genNumber, fitMin, "b-", label="Minimum fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness", color="b")
        for tl in ax.get_yticklabels():
            tl.set_color("b")

        # Plot average size
        ax2 = ax.twinx()
        line2 = ax2.plot(genNumber, avgSize, "r-", label="Average nr. waypoints")
        ax2.set_ylabel("Size", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        lines = line1 + line2
        labs = [line.get_label() for line in lines]
        ax.legend(lines, labs, loc="center right")


def concatenated_front(front):
    # First concatenate sub path objective values
    subObjVals = []
    for subFront in front:
        subObjVals.append(np.array([ind.fitness.values for ind in subFront]))
    objVals = [sum(x) for x in itertools.product(*subObjVals)]
    return update(objVals)


class RoutePlotter:
    def __init__(self, DIR, inputResults=None, rawResults=None, vessel=None):
        self.processedResults = inputResults
        self.rawResults = rawResults
        self.DIR = DIR

        # Colorbar settings
        if vessel is None:
            vessel = evaluation.Vessel(fuelPrice=0.3)
        self.vMin, self.vMax = min(vessel.speeds), max(vessel.speeds)

        cmap = cm.get_cmap('jet', 12)
        cmapList = [cmap(i) for i in range(cmap.N)][1:-1]
        self.cmap = cl.LinearSegmentedColormap.from_list('Custom cmap', cmapList, cmap.N-2)
        # bounds = np.linspace(self.vMin, self.vMax, 9)
        # self.norm = cl.BoundaryNorm(bounds, self.cmap.N)

        # Set extent
        minx, miny, maxx, maxy = 180, 85, -180, -85
        if inputResults:
            for route in inputResults['routeResponse']:
                lons, lats = zip(*[(leg['lon'], leg['lat']) for leg in route['waypoints']])
                for x, y in zip(lons, lats):
                    minx, miny = min(minx, x), min(miny, y)
                    maxx, maxy = max(maxx, x), max(maxy, y)
            for initRoute in inputResults['initialRoutes']:
                for subInitRoute in initRoute['route']:
                    for objRoute in subInitRoute.values():
                        lons, lats = zip(*[leg[0] for leg in objRoute])
                        for x, y in zip(lons, lats):
                            minx, miny = min(minx, x), min(miny, y)
                            maxx, maxy = max(maxx, x), max(maxy, y)
            margin = 0.1 * max((maxx - minx), (maxy - miny)) / 2
            self.extent = (max(minx - margin, -180), max(miny - margin, -90),
                           min(maxx + margin, 180), min(maxy + margin, 90))

        else:
            self.extent = (-180, -80, 180, 80)

    def results(self, resolution='i',
                weatherDate=None,
                current=None,
                initial=False,
                nRoutes=None,
                bathymetry=False,
                ecas=False,
                colorbar=True
                ):
        fig, ax = plt.subplots()

        if weatherDate:
            bathymetry = False

        m = self.navigation_area(ax, resolution, current=current, weather=weatherDate, bathymetry=bathymetry,
                                 eca=ecas)

        if initial:
            for initRoute in self.processedResults['initialRoutes']:
                for subInitRoute in initRoute['route']:
                    for objRoute in subInitRoute.values():
                        self.route(objRoute, m)
        if colorbar:
            self.colorbar(ax, self.cmap)
        # Plot route responses
        if nRoutes is None:
            for route in self.processedResults['routeResponse']:
                route = [((leg['lon'], leg['lat']), leg['speed']) for leg in route['waypoints']]
                self.route(route, m, colors=self.cmap)
        else:
            for front in self.rawResults['fronts']:
                for subFront in front:
                    n = len(subFront)
                    if nRoutes == 'all':
                        ii = range(n)
                    else:
                        midIndexes = np.unique(np.linspace(1, n-2, nRoutes-2).astype(int)).tolist()
                        ii = [0] + midIndexes + [n-1] if n > 2 else [0, 1] if n == 2 else [0]
                    for i in ii:
                        line = 'dashed' if 0 < i < n-1 else 'solid'
                        self.route(subFront[i], m, line=line, colors=self.cmap)
        return fig, ax

    def navigation_area(self, ax, resolution='c', current=None, bathymetry=False, eca=False, weather=None):
        left, bottom, right, top = self.extent
        m = Basemap(projection='merc', resolution=resolution,
                    llcrnrlat=bottom, urcrnrlat=top,
                    llcrnrlon=left, urcrnrlon=right,
                    ax=ax)
        # m.drawparallels(np.arange(-90., 90., 30.), labels=[1, 0, 0, 0], fontsize=8)
        # m.drawmeridians(np.arange(-180., 180., 60.), labels=[0, 0, 0, 1], fontsize=8)
        m.drawmapboundary(color='black')
        m.fillcontinents(color='lightgray', lake_color='lightgray', zorder=2)
        m.drawcoastlines()

        # Bathymetry
        if weather:
            plot_weather(m, weather)
            bathymetry = False

        if bathymetry:
            # m = Basemap(projection='lcc', resolution=None, llcrnrlat=bottom, urcrnrlat=top,
            #         llcrnrlon=left, urcrnrlon=right, lat_0=(top+bottom)/2, lon_0=(right+left)/2, ax=ax)
            # m.etopo()
            m.readshapefile(Path(self.DIR / "data/bathymetry_200m/ne_10m_bathymetry_K_200").as_posix(), 'ne_10m_bathymetry_K_200',
                            drawbounds=False)
            ps = [patches.Polygon(np.array(shape), True) for shape in m.ne_10m_bathymetry_K_200]
            ax.add_collection(PatchCollection(ps, facecolor='white', zorder=2))
            m.drawmapboundary(color='black', fill_color='khaki')

        if eca:
            m.readshapefile(Path(self.DIR / "data/eca_reg14_sox_pm/eca_reg14_sox_pm").as_posix(), 'eca_reg14_sox_pm', drawbounds=False)
            ps = [patches.Polygon(np.array(shape), True) for shape in m.eca_reg14_sox_pm]
            ax.add_collection(PatchCollection(ps, facecolor='green', alpha=0.5, zorder=3))

        if current:
            uin, vin = current['u'], current['v']
            lons, lats = current['lons'], current['lats']
            self.currents(ax, m, uin, vin, lons, lats)

        return m

    def currents(self, ax, m, uin, vin, lons, lats):
        # Transform vector and coordinate data
        dLon = self.extent[2] - self.extent[0]
        dLat = self.extent[3] - self.extent[1]

        vLon = int(dLon * 4)
        vLat = int(dLat * 4)
        uRot, vRot, x, y = m.transform_vector(uin, vin, lons, lats, vLon, vLat, returnxy=True)
        Q = m.quiver(x, y, uRot, vRot, np.hypot(uRot, vRot), pivot='mid', width=0.002, headlength=4, cmap='PuBu',
                     scale=90, ax=ax)
        ax.quiverkey(Q, 0.5, -0.1, 2, r'$2$ knots', labelpos='E')

    def colorbar(self, ax, colors):
        # Create color bar
        sm = plt.cm.ScalarMappable(cmap=colors)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.05)  # Colorbar axis
        cb = plt.colorbar(sm, norm=plt.Normalize(vmin=self.vMin, vmax=self.vMax), cax=cax)
        vDif = self.vMax - self.vMin
        nTicks = 6
        cb.ax.set_yticklabels(['%.1f' % round(self.vMin + i * vDif / (nTicks - 1), 1) for i in range(nTicks)], fontsize=8)
        cb.set_label('Calm water speed [knots]', rotation=270, labelpad=15)

    def route(self, route, m, line='solid', colors=None):
        waypoints = [leg[0] for leg in route]
        speeds = [leg[1] for leg in route]
        for i, speed in enumerate(speeds):
            if speed is None:
                speeds[i] = 0.0
        arcs = zip(waypoints[:-1], waypoints[1:])

        # Normalized speeds for colors
        normalized_speeds = [(speed - self.vMin) / (self.vMax - self.vMin) for speed in speeds] + [0]

        # Plot edges
        for i, a in enumerate(arcs):
            if colors is None:
                color = 'k'
            else:
                color = colors(1-normalized_speeds[i])
            m.drawgreatcircle(a[0][0], a[0][1], a[1][0], a[1][1], linewidth=1, linestyle=line,
                              color=color, zorder=3)
        for i, (x, y) in enumerate(waypoints):
            m.scatter(x, y, latlon=True, color='black', marker='o', s=1, zorder=4)


def plot_weather(m, dateTime):
    retriever = WindDataRetriever(nDays=1, startDate=dateTime)
    ds = retriever.get_data(forecast=False)

    lons, lats = np.linspace(-180, 179.5, 720), np.linspace(-90, 89.5, 360)
    x, y = m(*np.meshgrid(lons, lats))

    BNarr = ds[0, 0, :-1]
    contourf = m.contourf(x, y, BNarr, cmap=cm.jet)
    m.colorbar(mappable=contourf, location='bottom')


if __name__ == '__main__':
    import os

    os.chdir('..')
    routePlotter = RoutePlotter(DIR=Path('D:/'))

    _fig, _ax = plt.subplots()
    routePlotter.navigation_area(_ax, resolution='l', eca=True, bathymetry=True)

    _fig.suptitle('This is a somewhat long figure title', fontsize=16)
    plt.show()

