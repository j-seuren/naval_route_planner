import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import ocean_current
import pickle

from matplotlib import cm
from mpl_toolkits.basemap import Basemap


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
                 vessel,
                 secas,
                 n_plots,
                 id_dict,
                 plot_secas=True,
                 show_colorbar=True,
                 uin=None,
                 vin=None,
                 lons=None,
                 lats=None):
        self.vessel = vessel
        self.col_map = cm.rainbow
        self.ID_dict = id_dict

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

            self.ax[i].set_title("{}".format(list(self.ID_dict.keys())[i]))
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
                u_rot, v_rot, x, y = self.m[i].transform_vector(uin, vin, lons, lats, vec_lon, vec_lat, returnxy=True)

                vec_plot = self.m[i].quiver(x, y, u_rot, v_rot, color='darkgray', scale=50, width=.002)
                plt.quiverkey(vec_plot, 0.2, -0.2, 1, '1 knot', labelpos='W')  # Position and reference label

            if plot_secas:
                for poly in secas:
                    lon, lat = poly.exterior.xy
                    x, y = self.m[i](lon, lat)
                    self.m[i].plot(x, y, 'o-', markersize=2, linewidth=1)

            if show_colorbar:
                # Create color bar
                sm = plt.cm.ScalarMappable(cmap=self.col_map)
                col_bar = plt.colorbar(sm, norm=plt.Normalize(vmin=min(vessel.speeds), vmax=max(vessel.speeds)))
                max_s = max(vessel.speeds)
                min_s = min(vessel.speeds)
                col_bar.ax.set_yticklabels(['{}'.format(min_s),
                                            '{}'.format(round((1 / 5) * (max_s - min_s) + min_s, 1)),
                                            '{}'.format(round((2 / 5) * (max_s - min_s) + min_s, 1)),
                                            '{}'.format(round((3 / 5) * (max_s - min_s) + min_s, 1)),
                                            '{}'.format(round((4 / 5) * (max_s - min_s) + min_s, 1)),
                                            '{}'.format(round(max_s, 1))])

    def plot_global_routes(self, global_routes, i):
        for pop in global_routes:
            for ind in pop:
                # Plot edges
                waypoints = [item[0] for item in ind]
                edges = zip(waypoints[:-1], waypoints[1:])
                for e in edges:
                    self.m[i].drawgreatcircle(e[0][0], e[0][1], e[1][0], e[1][1], linewidth=2, color='black', zorder=1)

    def plot_individuals(self, inds, i):
        for ind in inds:
            # Create colors
            true_speeds = [item[1] for item in ind[:-1]]
            normalized_speeds = [(speed - min(self.vessel.speeds)) /
                                 (max(self.vessel.speeds) - min(self.vessel.speeds)) for speed in true_speeds] + [0]

            # Plot edges
            waypoints = [item[0] for item in ind]
            edges = zip(waypoints[:-1], waypoints[1:])
            for j, e in enumerate(edges):
                self.m[i].drawgreatcircle(e[0][0], e[0][1], e[1][0], e[1][1], linewidth=2,
                                          color=self.col_map(normalized_speeds[j]), zorder=1)
            for j, (x, y) in enumerate(waypoints):
                self.m[i].scatter(x, y, latlon=True, color='dimgray', marker='o', s=5, zorder=2)


if __name__ == "__main__":
    import main

    route_planner = main.RoutePlanner(seca_factor=1.2,
                                      res='c',
                                      spl_th=4,
                                      vessel_name='Fairmaster',
                                      incl_curr=True)

    _ID_dict = {'1    current': '16_57_02',
                '1 no current': '17_46_15',
                '2 1,200    current': '',
                '2 1,200 no current': '21_31_13'}

    # # Brazil -> Caribbean Sea
    # _ID_dict = {'   Currents -    SECA': '',
    #             '   Currents - no SECA': '',
    #             'no Currents -    SECA': '11_40_38',
    #             'no Currents - no SECA': '11_42_36'}

    # # Scandinavia -> Caribbean Sea
    # _ID_dict = {'   Currents -    SECA': '16_55_20',
    #             '   Currents - no SECA': '17_06_23',
    #             'no Currents -    SECA': '17_10_02',
    #             'no Currents - no SECA': '17_54_32'}

    # _ID_dict = {'   Currents -    SECA': '15_13_57',  # Currents and seca
    #             '   Currents - no SECA': '15_21_43',  # Currents and no seca
    #             'no Currents -    SECA': '15_25_58',  # No currents and seca
    #             'no Currents - no SECA': '15_24_44'}  # No currents and no seca

    with open('C:/dev/data/seca_areas_csv', 'rb') as file:
        _secas = pickle.load(file)

    with open('output/paths/{}_paths'.format(next(iter(_ID_dict.values()))), 'rb') as f:
        first_paths = pickle.load(f)

    _n_plots = len(_ID_dict)
    plotter = RoutePlotter(first_paths, route_planner.vessel, _secas, _n_plots, _ID_dict)

    for _i, key in enumerate(_ID_dict.keys()):
        with open('output/paths/{}_paths'.format(_ID_dict[key]), 'rb') as f:
            _paths = pickle.load(f)

        with open('output/logs/{}_logs'.format(_ID_dict[key]), 'rb') as f:
            _path_logs = pickle.load(f)

        with open('output/glob_routes/{}_init_routes'.format(_ID_dict[key]), 'rb') as f:
            _global_routes = pickle.load(f)

        plot_stats(_path_logs, key)

        best_inds = {}
        for _sub_paths in _paths.values():
            for _pop in _sub_paths.values():
                min_tt, min_fc = math.inf, math.inf
                tt_ind, fc_ind = None, None
                for min_ind in _pop:
                    tt, fc = min_ind.fitness.values
                    if tt < min_tt:
                        tt_ind = min_ind
                        min_tt = tt
                    if fc < min_fc:
                        fc_ind = min_ind
                        min_fc = fc

                best_inds['Minimal fuel: '] = fc_ind
                best_inds['Minimal time: '] = tt_ind

        plotter.plot_individuals(best_inds.values(), _i)
        plotter.plot_global_routes(_global_routes, _i)

        _n_days = math.ceil(max([ind.fitness.values[0] for ind in best_inds.values()]))
        print('n_days:', _n_days)
        _start_date = datetime.datetime(2016, 1, 1)
        route_planner.evaluator.current_data = ocean_current.CurrentData(
            _start_date, _n_days)

        for k, _ind in best_inds.items():
            fit = route_planner.tb.evaluate(_ind)
            print('{0:>20}, {1}'.format(list(_ID_dict.keys())[_i], k), fit)
            print('{0:>20}, {1} ORIGINAL'.format(list(_ID_dict.keys())[_i], k), _ind.fitness.values)
            # print('DIFF ORIGINAL: {}'.format(np.subtract(_ind.fitness.values, fit)))

    plt.show()
