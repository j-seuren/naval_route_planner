import evaluation
import folium
import main
import math
import matplotlib.pyplot as plt
import numpy as np
import weather
import os
import pickle

from matplotlib import cm, patches
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path


def plot_stats(path_logs, name):
    for path_log in path_logs.values():
        for sub_path_log in path_log.values():
            _gen = sub_path_log.select("gen")
            fit_mins = sub_path_log.chapters["fitness"].select("min")
            size_avgs = sub_path_log.chapters["size"].select("avg")

            fig, ax1 = plt.subplots()
            fig.suptitle(name)
            line1 = ax1.save_fronts(_gen, fit_mins, "b-", label="Minimum Fitness")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Fitness", color="b")
            for tl in ax1.get_yticklabels():
                tl.set_color("b")

            ax2 = ax1.twinx()
            line2 = ax2.save_fronts(_gen, size_avgs, "r-", label="Average Size")
            ax2.set_ylabel("Size", color="r")
            for tl in ax2.get_yticklabels():
                tl.set_color("r")

            lines = line1 + line2
            labs = [line.get_label() for line in lines]
            ax1.legend(lines, labs, loc="center right")
            print('{} - final avg size'.format(name), size_avgs[-1])


class RoutePlotter:
    def __init__(self, bounds, n_plots, vessel=None, titles=None, ecas=None, show_colorbar=True, uin=None, vin=None,
                 lons=None, lats=None, DIR=Path('D:/')):
        self.DIR = DIR
        self.col_map = cm.rainbow
        if titles:
            self.titles = titles
        else:
            self.titles = ['no_title'] * n_plots

        self.ecas = ecas
        self.show_colorbar = show_colorbar
        self.n_plots = n_plots

        if vessel:
            self.vessel = vessel
        else:
            self.vessel = evaluation.Vessel()

        if uin is not None:
            self.uin = uin
            self.vin = vin
            self.lons = lons
            self.lats = lats
            self.plot_currents = True
        else:
            self.uin = self.vin = self.lons = self.lats = None
            self.plot_currents = False

        # Set extent
        minx, miny, maxx, maxy = bounds
        margin = 2
        self.lef = max(minx - margin, -180)
        self.rig = min(maxx + margin, 180)
        self.bot = max(miny - margin, -90)
        self.top = min(maxy + margin, 90)

        self.rows = int(round(math.sqrt(n_plots)))
        self.columns = int(math.ceil(math.sqrt(n_plots)))

    def plot_routes(self, results_files, best_inds_files):
        fig = plt.figure()
        fig.suptitle('Routes', fontsize=12)

        r_iter = iter([r for ro in range(self.rows) for r in [ro] * self.columns])
        c_iter = iter(list(range(self.columns)) * self.rows)
        for i, result in enumerate(results_files):
            r, c = next(r_iter), next(c_iter)

            shortest_paths = result['initialRoutes']
            individuals = best_inds_files[i]

            ax = plt.subplot2grid((self.rows, self.columns), (r, c))
            ax.set_title("{}".format(self.titles[i]), fontsize=10)
            m = Basemap(projection='merc', resolution='c', llcrnrlat=self.bot, urcrnrlat=self.top, llcrnrlon=self.lef,
                        urcrnrlon=self.rig, ax=ax)
            m.drawparallels(np.arange(-90., 90., 10.), labels=[1, 0, 0, 0], fontsize=8)
            m.drawmeridians(np.arange(-180., 180., 10.), labels=[0, 0, 0, 1], fontsize=8)
            m.drawmapboundary(color='black', fill_color='aqua')
            m.fillcontinents(color='lightgray', lake_color='lightgray', zorder=2)
            m.readshapefile(self.DIR / 'data/bathymetry_200m/ne_10m_bathymetry_K_200', 'ne_10m_bathymetry_K_200',
                            drawbounds=False)

            ps = [patches.Polygon(np.array(shape), True) for shape in m.ne_10m_bathymetry_K_200]
            ax.add_collection(PatchCollection(ps, facecolor='white', zorder=2))

            # Plot initial routes
            for p in shortest_paths.values():
                for sp in p['path'].values():
                    for shortest_path in sp.values():
                        waypoints = [item[0] for item in shortest_path]
                        edges = zip(waypoints[:-1], waypoints[1:])
                        for e in edges:
                            m.drawgreatcircle(e[0][0], e[0][1], e[1][0], e[1][1], linewidth=2, color='black', zorder=3)

            # Plot individuals
            for p in individuals.values():
                for sp in p.values():
                    for ind in sp:
                        # Create colors
                        true_speeds = [item[1] for item in ind[:-1]]
                        normalized_speeds = [(speed - min(self.vessel.speeds)) / (
                            max(self.vessel.speeds) - min(self.vessel.speeds)) for speed in true_speeds] + [0]

                        # Plot edges
                        waypoints = [item[0] for item in ind]
                        edges = zip(waypoints[:-1], waypoints[1:])
                        for j, e in enumerate(edges):
                            m.drawgreatcircle(e[0][0], e[0][1], e[1][0], e[1][1], linewidth=2,
                                              color=self.col_map(normalized_speeds[j]), zorder=3)
                        for j, (x, y) in enumerate(waypoints):
                            m.scatter(x, y, latlon=True, color='dimgray', marker='o', s=5, zorder=4)

            if self.plot_currents:
                # Transform vector and coordinate data
                vec_lon = int(self.uin.shape[1] / (2 * 360 / (self.rig - self.lef)))
                vec_lat = int(self.uin.shape[0] / (2 * 180 / (self.top - self.bot)))
                u_rot, v_rot, x, y = m.transform_vector(self.uin, self.vin, self.lons, self.lats, vec_lon, vec_lat,
                                                        returnxy=True)

                vec_plot = m.quiver(x, y, u_rot, v_rot, color='gray', scale=50, width=.002)
                plt.quiverkey(vec_plot, 0.2, -0.2, 1, '1 knot', labelpos='W')  # Position and reference label

            if self.ecas:
                for eca in self.ecas:
                    lon, lat = eca.exterior.xy
                    x, y = m(lon, lat)
                    m.plot(x, y, 'o-', markersize=2, linewidth=1)

            if self.show_colorbar:
                # Create color bar
                sm = plt.cm.ScalarMappable(cmap=self.col_map)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                col_bar = plt.colorbar(sm,
                                       norm=plt.Normalize(vmin=min(self.vessel.speeds), vmax=max(self.vessel.speeds)),
                                       cax=cax)
                min_s, max_s = min(self.vessel.speeds), max(self.vessel.speeds)
                # noinspection PyProtectedMember
                col_bar._ax.set_yticklabels(
                    ['%.1f' % round(min_s, 1), '%.1f' % round((1 / 5) * (max_s - min_s) + min_s, 1),
                     '%.1f' % round((2 / 5) * (max_s - min_s) + min_s, 1),
                     '%.1f' % round((3 / 5) * (max_s - min_s) + min_s, 1),
                     '%.1f' % round((4 / 5) * (max_s - min_s) + min_s, 1), '%.1f' % round(max_s, 1)], fontsize=8)
                col_bar.set_label('Calm water speed [kn]', rotation=270, labelpad=15, fontsize=10)


def plot_fronts(front):
    # Plot fronts
    front = np.array([_ind.fitness.values for _ind in front])

    ax = plt.subplot2grid((max_sp, n_p), (sp_key, _p))
    ax.scatter(front[:, 0], front[:, 1], c="b", s=1)
    ax.set_title('P{} - SP{}'.format(_p, sp_key))
    ax.axis("tight")
    ax.grid()
    ax.set_xlabel('Travel time [d]')
    ax.set_ylabel(r'Fuel cost [USD, $\times 1000$]')


def plot_interactive_route(path, path_key, obj_key):
    wps = [el[0] for el in path]
    start_point, end_point = wps[0], wps[-1]

    # Plot of the path using folium
    geo_path = np.asarray(wps)
    geo_path[:, [0, 1]] = geo_path[:, [1, 0]]  # swap columns
    geo_map = folium.Map([0, 0], zoom_start=2)
    for point in geo_path:
        folium.Marker(point, popup=str(point)).add_to(geo_map)
    folium.PolyLine(geo_path).add_to(geo_map)

    # Add a Mark on the start and positions in a different color
    folium.Marker(geo_path[0], popup=str(start_point), icon=folium.Icon(color='red')).add_to(geo_map)
    folium.Marker(geo_path[-1], popup=str(end_point), icon=folium.Icon(color='red')).add_to(geo_map)

    # Save the interactive plot as a map
    output_name = 'output/example_path_{}_obj_{}_plot.html'.format(path_key, obj_key)
    geo_map.save(output_name)
    print('Output saved to: {}'.format(output_name))


if __name__ == "__main__":
    os.chdir('..')
    _ID_dict = {'Cruisje naar Sydney': '22_09_26'}

    # _start_date = datetime.datetime(2016, 1, 1)
    _start_date = None
    planner = main.RoutePlanner()

    DIR = Path('D:/')

    with open(DIR / 'data/seca_areas_csv', 'rb') as file:
        _ecas = pickle.load(file)

    # Get outer bounds of all paths
    _minx, _miny, _maxx, _maxy = 180, 90, -180, -90
    for file_id in _ID_dict.values():
        with open('output/result/{}'.format(file_id), 'rb') as f:
            _result = pickle.load(f)

        for _p in _result['fronts'].values():
            for _front in _p.values():
                # Get outer bounds of paths
                _wps = np.asarray([row[0] for ind in _front for row in ind])
                minx_i, miny_i = np.amin(_wps, axis=0)
                maxx_i, maxy_i = np.amax(_wps, axis=0)
                _minx, _miny = min(minx_i, _minx), min(miny_i, _miny)
                _maxx, _maxy = max(maxx_i, _maxx), max(maxy_i, _maxy)

    _bounds = (_minx, _miny, _maxx, _maxy)

    plotter = RoutePlotter(bounds=_bounds, n_plots=len(_ID_dict), titles=list(_ID_dict.keys()), ecas=_ecas,
                           vessel=planner.vessel)

    all_results, all_best_inds = [], []
    # Inspect best individuals for every ID dict key
    for f_idx, f_title in enumerate(_ID_dict.keys()):
        with open('output/result/{}'.format(_ID_dict[f_title]), 'rb') as f:
            _result = pickle.load(f)

        # Initialize pareto_fig
        pareto_fig, axs = plt.subplots(squeeze=False)
        pareto_fig.suptitle(f_title)

        n_p = len(_result['fronts'])
        max_sp = max([len(_result['fronts'][p]) for p in _result['fronts']])

        # Get best individuals in front
        best_inds = {}
        max_days = 0
        for _p, p_val in _result['fronts'].items():
            best_inds[_p] = {'fuel': [], 'time': [], 'fit': []}
            for sp_key, _front in p_val.items():
                # Get min time, min fuel and min fitness individuals
                fit_values = np.asarray([_ind.fitness.values for _ind in _front])

                time_ind = _front[np.argmin(fit_values[:, 0])]
                fuel_ind = _front[np.argmin(fit_values[:, 1])]
                fit_ind = _front[0]

                best_inds[_p]['fuel'].append(fuel_ind)
                best_inds[_p]['time'].append(time_ind)
                best_inds[_p]['fit'].append(fit_ind)

                # Get max days
                max_days = math.ceil(max(max_days, max(
                    [fuel_ind.fitness.values[0], time_ind.fitness.values[0], fit_ind.fitness.values[0]])))

                # Plot fronts
                plot_fronts(_front)

        print('n_days:', max_days)

        if _start_date:
            planner.evaluator.currentOp = weather.CurrentOperator(_start_date, max_days)

        for _p, _path in best_inds.items():
            for _k, _ind_list in _path.items():
                total_fit = total_original_fit = np.zeros(2)
                for _ind in _ind_list:
                    plot_interactive_route(_ind, _p, _k)
                    fit = np.array(planner.tb.evaluate(_ind))
                    original_fit = np.array(_ind.fitness.values)
                    total_fit += fit
                    total_original_fit += original_fit

                file_key = list(_ID_dict.keys())[f_idx]
                print('File {} Path {} Obj {:>4} real'.format(file_key, _p, _k), np.round(total_fit,
                                                                                          2))  # print('File {} Path
                # {} Obj {:>4} orig'.format(file_key, _p, _k), np.round(total_original_fit, 2))  # print('DIFF
                # ORIGINAL: {}'.format(np.subtract(_ind.fitness.values, fit)))

        all_results.append(_result)
        all_best_inds.append(best_inds)

    plotter.plot_routes(all_results, all_best_inds)

    for f_idx, f_title in enumerate(_ID_dict.keys()):
        with open('output/result/{}'.format(_ID_dict[f_title]), 'rb') as f:
            _result = pickle.load(f)

        plot_stats(_result['logs'], f_title)
    plt.show()
