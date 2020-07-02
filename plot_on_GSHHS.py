import math
import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib import cm
from mpl_toolkits.basemap import Basemap


class Plotter:
    def __init__(self,
                 paths,
                 vessel,
                 secas,
                 n_plots,
                 ID_dict,
                 plot_secas=True,
                 show_colorbar=True,
                 uin=None,
                 vin=None,
                 lons=None,
                 lats=None):
        self.vessel = vessel
        self.col_map = cm.rainbow
        self.ID_dict = ID_dict

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
        all_wps = []
        for sub_paths in paths.values():
            for pop in sub_paths.values():
                for ind in pop:
                    all_wps.extend([item[0] for item in ind])
        min_x, min_y = min(all_wps, key=lambda t: t[0])[0], min(all_wps, key=lambda t: t[1])[1]
        max_x, max_y = max(all_wps, key=lambda t: t[0])[0], max(all_wps, key=lambda t: t[1])[1]
        margin = 5
        left, right = max(math.floor(min_x) - margin, -180), min(math.ceil(max_x) + margin, 180)
        bottom, top = max(math.floor(min_y) - margin, -90), min(math.ceil(max_y) + margin, 90)

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
                                llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left, urcrnrlon=right, ax=self.ax[i])
            self.m[i].drawparallels(np.arange(-90., 90., 10.), labels=[1, 0, 0, 0], fontsize=10)
            self.m[i].drawmeridians(np.arange(-180., 180., 10.), labels=[0, 0, 0, 1], fontsize=10)
            self.m[i].drawcoastlines()
            self.m[i].fillcontinents()

            if plot_currents:
                # Transform vector and coordinate data
                vec_lon = int(uin.shape[1] / (2 * 360 / (right - left)))
                vec_lat = int(uin.shape[0] / (2 * 180 / (top - bottom)))
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
                                      resolution='c',
                                      max_poly_size=4,
                                      n_gen=100,
                                      mu=4 * 20,
                                      vessel_name='Fairmaster',
                                      cx_prob=0.9,
                                      include_currents=True)

    ID_dict = {'NSGA2': '22_43_00',
               'SPEA2': '19_12_34'}

    # ID_dict = {'Currents Seca': '16_55_20',
    #            'Currents     ': '17_06_23',
    #            '         Seca': '17_10_02',
    #            '             ': '17_54_32'}

    # ID_dict = {'Currents Seca': '15_13_57',  # Currents and seca
    #            'Currents     ': '15_21_43',  # Currents and no seca
    #            '         Seca': '15_25_58',  # No currents and seca
    #            '             ': '15_24_44'}  # No currents and no seca
    initialized = False
    for _i, key in enumerate(ID_dict.keys()):
        with open('C:/dev/data/seca_areas_csv', 'rb') as file:
            _secas = pickle.load(file)

        with open('output/paths/{}_paths'.format(ID_dict[key]), 'rb') as f:
            _paths = pickle.load(f)

        with open('output/glob_routes/{}_init_routes'.format(ID_dict[key]), 'rb') as f:
            _global_routes = pickle.load(f)

        if not initialized:
            n_plots = len(ID_dict)
            plotter = Plotter(_paths, route_planner.vessel, _secas, n_plots, ID_dict)
            initialized = True

        best_inds = {}
        for sub_paths in _paths.values():
            for pop in sub_paths.values():
                min_tt, min_fc = math.inf, math.inf
                tt_ind, fc_ind = None, None
                for min_ind in pop:
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

        for k, ind in best_inds.items():
            fit = route_planner.toolbox.evaluate(ind)
            print('{0}, {1}'.format(list(ID_dict.keys())[_i], k), fit)

    plt.show()
