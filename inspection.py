import datetime
import main
import math
import matplotlib.pyplot as plt
import numpy as np
import ocean_current
import pickle

from matplotlib import cm
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_stats(path_logs, name):
    for p, path_log in enumerate(path_logs.values()):
        for sp, sub_path_log in enumerate(path_log.values()):
            _gen = sub_path_log.select("gen")
            fit_mins = sub_path_log.chapters["fitness"].select("min")
            size_avgs = sub_path_log.chapters["size"].select("avg")

            fig, ax1 = plt.subplots()
            fig.suptitle(name)
            line1 = ax1.plot(_gen, fit_mins, "b-", label="Minimum Fitness")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Fitness", color="b")
            for tl in ax1.get_yticklabels():
                tl.set_color("b")

            ax2 = ax1.twinx()
            line2 = ax2.plot(_gen, size_avgs, "r-", label="Average Size")
            ax2.set_ylabel("Size", color="r")
            for tl in ax2.get_yticklabels():
                tl.set_color("r")

            lines = line1 + line2
            labs = [line.get_label() for line in lines]
            ax1.legend(lines, labs, loc="center right")
            print('{} - final avg size'.format(name), size_avgs[-1])


class RoutePlotter:
    def __init__(self,
                 paths,
                 n_plots,
                 vessel=None,
                 titles=None,
                 ecas=None,
                 show_colorbar=True,
                 uin=None,
                 vin=None,
                 lons=None,
                 lats=None):
        self.col_map = cm.rainbow
        if titles:
            self.titles = titles
        else:
            self.titles = ['no_title'] * n_plots

        if vessel:
            self.vessel = vessel
        else:
            self.vessel = main.Vessel()

        if uin is not None:
            self.uin = uin
            self.vin = vin
            self.lons = lons
            self.lats = lats
            plot_currents = True
        else:
            self.uin = self.vin = self.lons = self.lats = None
            plot_currents = False

        # Set extent
        all_wps = [wp[0] for sp in paths.values()
                   for pop in sp.values()
                   for ind in pop
                   for wp in ind]
        minx = int(min(all_wps, key=lambda t: t[0])[0])
        miny = int(min(all_wps, key=lambda t: t[1])[1])
        maxx = int(max(all_wps, key=lambda t: t[0])[0])
        maxy = int(max(all_wps, key=lambda t: t[1])[1])
        margin = 5
        lef = max(minx - margin, -180)
        rig = min(maxx + margin, 180)
        bot = max(miny - margin, -90)
        top = min(maxy + margin, 90)

        rows = round(math.sqrt(n_plots))
        columns = math.ceil(math.sqrt(n_plots))

        r_iter = iter([r for ro in range(rows) for r in [ro] * columns])
        c_iter = iter(list(range(columns)) * rows)

        plt.figure()

        self.ax = [None] * n_plots
        self.m = [None] * n_plots

        for i in range(n_plots):
            r, c = next(r_iter), next(c_iter)
            self.ax[i] = plt.subplot2grid((rows, columns), (r, c))

            self.ax[i].set_title("{}".format(self.titles[i]))
            self.m[i] = Basemap(projection='merc', resolution='c',
                                llcrnrlat=bot, urcrnrlat=top,
                                llcrnrlon=lef, urcrnrlon=rig, ax=self.ax[i])
            self.m[i].drawparallels(np.arange(-90., 90., 10.),
                                    labels=[1, 0, 0, 0], fontsize=10)
            self.m[i].drawmeridians(np.arange(-180., 180., 10.),
                                    labels=[0, 0, 0, 1], fontsize=10)
            self.m[i].drawcoastlines()
            self.m[i].fillcontinents()

            if plot_currents:
                # Transform vector and coordinate data
                vec_lon = int(uin.shape[1] / (2 * 360 / (rig - lef)))
                vec_lat = int(uin.shape[0] / (2 * 180 / (top - bot)))
                u_rot, v_rot, x, y = self.m[i].transform_vector(uin, vin,
                                                                lons, lats,
                                                                vec_lon, vec_lat,
                                                                returnxy=True)

                vec_plot = self.m[i].quiver(x, y, u_rot, v_rot,
                                            color='gray', scale=50, width=.002)
                plt.quiverkey(vec_plot, 0.2, -0.2, 1, '1 knot', labelpos='W')  # Position and reference label

            if ecas:
                for eca in ecas:
                    lon, lat = eca.exterior.xy
                    x, y = self.m[i](lon, lat)
                    self.m[i].plot(x, y, 'o-', markersize=2, linewidth=1)

            if show_colorbar:
                # Create color bar
                sm = plt.cm.ScalarMappable(cmap=self.col_map)
                divider = make_axes_locatable(self.ax[i])  # Width of cax is x% of ax, padding cax <-> ax: 0.05inch
                cax = divider.append_axes("right", size="1%", pad=0.05)
                col_bar = plt.colorbar(sm, norm=plt.Normalize(vmin=min(vessel.speeds), vmax=max(vessel.speeds)),
                                       cax=cax)
                min_s, max_s = min(vessel.speeds), max(vessel.speeds)
                col_bar.ax.set_yticklabels(['%.1f' % round(min_s, 1),
                                            '%.1f' % round((1 / 5) * (max_s - min_s) + min_s, 1),
                                            '%.1f' % round((2 / 5) * (max_s - min_s) + min_s, 1),
                                            '%.1f' % round((3 / 5) * (max_s - min_s) + min_s, 1),
                                            '%.1f' % round((4 / 5) * (max_s - min_s) + min_s, 1),
                                            '%.1f' % round(max_s, 1)])

    def plot_shortest_paths(self, shortest_paths, i):
        for shortest_path in shortest_paths.values():
            for sub_shortest_path in shortest_path.values():
                for ind in sub_shortest_path.values():
                    # Plot edges
                    waypoints = [item[0] for item in ind]
                    edges = zip(waypoints[:-1], waypoints[1:])
                    for e in edges:
                        self.m[i].drawgreatcircle(e[0][0], e[0][1],
                                                  e[1][0], e[1][1],
                                                  linewidth=2, color='black', zorder=1)

    def plot_individuals(self, inds, i):
        for ind in inds:
            # Create colors
            true_speeds = [item[1] for item in ind[:-1]]
            normalized_speeds = [(speed - min(self.vessel.speeds)) /
                                 (max(self.vessel.speeds) - min(self.vessel.speeds))
                                 for speed in true_speeds] + [0]

            # Plot edges
            waypoints = [item[0] for item in ind]
            edges = zip(waypoints[:-1], waypoints[1:])
            for j, e in enumerate(edges):
                self.m[i].drawgreatcircle(e[0][0], e[0][1], e[1][0], e[1][1], linewidth=2,
                                          color=self.col_map(normalized_speeds[j]), zorder=1)
            for j, (x, y) in enumerate(waypoints):
                self.m[i].scatter(x, y, latlon=True, color='dimgray', marker='o', s=5, zorder=2)


if __name__ == "__main__":
    route_planner = main.RoutePlanner(eca_f=1.2,
                                      res='c',
                                      spl_th=4,
                                      vessel_name='Fairmaster',
                                      incl_curr=True)

    _ID_dict = {'hof test': '19_44_34'}

    # # Brazil -> Caribbean Sea
    # _ID_dict = {'   Currents -    ECA': '',
    #             '   Currents - no ECA': '',
    #             'no Currents -    ECA': '11_40_38',
    #             'no Currents - no ECA': '11_42_36'}

    # # Scandinavia -> Caribbean Sea
    # _ID_dict = {'   Currents -    ECA': '16_55_20',
    #             '   Currents - no ECA': '17_06_23',
    #             'no Currents -    ECA': '17_10_02',
    #             'no Currents - no ECA': '17_54_32'}

    # _ID_dict = {'   Currents -    ECA': '15_13_57',  # Currents and eca
    #             '   Currents - no ECA': '15_21_43',  # Currents and no eca
    #             'no Currents -    ECA': '15_25_58',  # No currents and eca
    #             'no Currents - no ECA': '15_24_44'}  # No currents and no eca

    with open('C:/dev/data/seca_areas_csv', 'rb') as file:
        _ecas = pickle.load(file)

    with open('output/paths/{}_paths'.format(next(iter(_ID_dict.values()))), 'rb') as f:
        first_paths = pickle.load(f)

    plotter = RoutePlotter(paths=first_paths,
                           n_plots=len(_ID_dict),
                           vessel=route_planner.vessel,
                           titles=list(_ID_dict.keys()),
                           ecas=_ecas)

    for _i, key in enumerate(_ID_dict.keys()):
        with open('output/paths/{}_paths'.format(_ID_dict[key]), 'rb') as f:
            _paths = pickle.load(f)

        with open('output/logs/{}_logs'.format(_ID_dict[key]), 'rb') as f:
            _path_logs = pickle.load(f)

        with open('output/hofs/{}_hofs'.format(_ID_dict[key]), 'rb') as f:
            _hofs = pickle.load(f)

        with open('output/glob_routes/{}_init_routes'.format(_ID_dict[key]), 'rb') as f:
            _global_routes = pickle.load(f)

        plot_stats(_path_logs, key)

        best_inds = {}
        for _path in _hofs.values():
            for _hof in _path.values():
                min_tt, min_fc = math.inf, math.inf
                tt_ind, fc_ind = None, None
                for i, _fit in enumerate(_hof.keys):
                    tt, fc = _fit.values
                    if tt < min_tt:
                        tt_ind = _hof.items[i]
                        min_tt = tt
                    if fc < min_fc:
                        fc_ind = _hof.items[i]
                        min_fc = fc

                best_inds['Minimal fuel: '] = fc_ind
                best_inds['Minimal time: '] = tt_ind

        plotter.plot_individuals(best_inds.values(), _i)
        plotter.plot_shortest_paths(_global_routes, _i)

        _n_days = math.ceil(max([ind.fitness.values[0] for ind in best_inds.values()]))
        print('n_days:', _n_days)
        _start_date = datetime.datetime(2016, 1, 1)

        if route_planner.incl_curr:
            route_planner.evaluator.current_data = ocean_current.CurrentOperator(
                _start_date, _n_days)

        for k, _ind in best_inds.items():
            fit = route_planner.tb.evaluate(_ind)
            print('{0:>20}, {1}'.format(list(_ID_dict.keys())[_i], k), fit)
            print('{0:>20}, {1} ORIGINAL'.format(list(_ID_dict.keys())[_i], k), _ind.fitness.values)
            # print('DIFF ORIGINAL: {}'.format(np.subtract(_ind.fitness.values, fit)))

    plt.show()
