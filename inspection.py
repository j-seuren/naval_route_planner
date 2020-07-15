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
                 bounds,
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

        self.ecas = ecas
        self.show_colorbar = show_colorbar
        self.n_plots = n_plots

        if vessel:
            self.vessel = vessel
        else:
            self.vessel = main.Vessel()

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
        margin = 5
        self.lef = max(minx - margin, -180)
        self.rig = min(maxx + margin, 180)
        self.bot = max(miny - margin, -90)
        self.top = min(maxy + margin, 90)

        self.rows = int(round(math.sqrt(n_plots)))
        self.columns = int(math.ceil(math.sqrt(n_plots)))

    def plot_routes(self, results_files, best_inds_files):
        fig = plt.figure()
        fig.suptitle('Routes')

        r_iter = iter([r for ro in range(self.rows) for r in [ro] * self.columns])
        c_iter = iter(list(range(self.columns)) * self.rows)
        for i, result in enumerate(results_files):
            r, c = next(r_iter), next(c_iter)

            shortest_paths = result['shortest_paths']
            individuals = best_inds_files[i]

            ax = plt.subplot2grid((self.rows, self.columns), (r, c))
            ax.set_title("{}".format(self.titles[i]))
            m = Basemap(projection='merc', resolution='c',
                        llcrnrlat=self.bot, urcrnrlat=self.top,
                        llcrnrlon=self.lef, urcrnrlon=self.rig,
                        ax=ax)
            m.drawparallels(np.arange(-90., 90., 10.),
                            labels=[1, 0, 0, 0], fontsize=10)
            m.drawmeridians(np.arange(-180., 180., 10.),
                            labels=[0, 0, 0, 1], fontsize=10)
            m.drawcoastlines()
            m.fillcontinents()

            # Plot initial routes
            for p in shortest_paths.values():
                for sp in p.values():
                    for shortest_path in sp.values():
                        waypoints = [item[0] for item in shortest_path]
                        edges = zip(waypoints[:-1], waypoints[1:])
                        for e in edges:
                            m.drawgreatcircle(e[0][0], e[0][1], e[1][0], e[1][1],
                                              linewidth=2, color='black', zorder=1)

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
                                              color=self.col_map(normalized_speeds[j]), zorder=1)
                        for j, (x, y) in enumerate(waypoints):
                            m.scatter(x, y, latlon=True, color='dimgray', marker='o', s=5, zorder=2)

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
                cax = divider.append_axes("right", size="1%", pad=0.05)
                col_bar = plt.colorbar(sm, norm=plt.Normalize(vmin=min(self.vessel.speeds),
                                                              vmax=max(self.vessel.speeds)),
                                       cax=cax)
                min_s, max_s = min(self.vessel.speeds), max(self.vessel.speeds)
                col_bar.ax.set_yticklabels(
                    ['%.1f' % round(min_s, 1), '%.1f' % round((1 / 5) * (max_s - min_s) + min_s, 1),
                     '%.1f' % round((2 / 5) * (max_s - min_s) + min_s, 1),
                     '%.1f' % round((3 / 5) * (max_s - min_s) + min_s, 1),
                     '%.1f' % round((4 / 5) * (max_s - min_s) + min_s, 1), '%.1f' % round(max_s, 1)])


def plot_fronts(front):
    # Plot fronts
    front = np.array([_ind.fitness.values for _ind in front])

    ax = plt.subplot2grid((max_sp, n_p), (sp_key, p_key))
    ax.scatter(front[:, 0], front[:, 1], c="b", s=1)
    ax.set_title('P{} - SP{}'.format(p_key, sp_key))
    ax.axis("tight")
    ax.grid()
    ax.set_xlabel('Travel time (days)')
    ax.set_ylabel('Fuel costs (1000 USD per tonne)')


if __name__ == "__main__":
    _ID_dict = {'spea2': '13_35_43',
                'nsga2': '13_35_12',
                'm-paes': '17_49_42'}
    incl_curr = False
    _start_date = datetime.datetime(2016, 1, 1)
    planner = main.RoutePlanner(incl_curr=incl_curr)

    with open('C:/dev/data/seca_areas_csv', 'rb') as file:
        _ecas = pickle.load(file)

    # Get outer bounds of all paths
    _minx, _miny, _maxx, _maxy = 180, 90, -180, -90
    for file_id in _ID_dict.values():
        with open('output/result/{}'.format(file_id), 'rb') as f:
            _result = pickle.load(f)

        for p_key in _result['fronts'].values():
            for _front in p_key.values():
                # Get outer bounds of paths
                wps = np.asarray([row[0] for ind in _front for row in ind])
                minx_i, miny_i = np.amin(wps, axis=0)
                maxx_i, maxy_i = np.amax(wps, axis=0)
                _minx, _miny = min(minx_i, _minx), min(miny_i, _miny)
                _maxx, _maxy = max(maxx_i, _maxx), max(maxy_i, _maxy)

    _bounds = (_minx, _miny, _maxx, _maxy)

    plotter = RoutePlotter(bounds=_bounds,
                           n_plots=len(_ID_dict),
                           titles=list(_ID_dict.keys()),
                           ecas=_ecas,
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
        for p_key, p_val in _result['fronts'].items():
            best_inds[p_key] = {}
            for sp_key, _front in p_val.items():
                # Get min time, min fuel and min fitness individuals
                fit_values = np.asarray([_ind.fitness.values for _ind in _front])

                time_ind = _front[np.argmin(fit_values[:, 0])]
                fuel_ind = _front[np.argmin(fit_values[:, 1])]
                fit_ind = _front[0]

                # Append best individuals to list
                if sp_key == 0:
                    best_inds[p_key]['fuel'] = [fuel_ind]
                    best_inds[p_key]['time'] = [time_ind]
                    best_inds[p_key]['fit'] = [fit_ind]
                else:
                    best_inds[p_key]['fuel'].append(fuel_ind)
                    best_inds[p_key]['time'].append(time_ind)
                    best_inds[p_key]['fit'].append(fit_ind)

                # Get max days
                max_days = math.ceil(max(max_days, max([fuel_ind.fitness.values[0],
                                                        time_ind.fitness.values[0],
                                                        fit_ind.fitness.values[0]])
                                         )
                                     )

                # Plot fronts
                plot_fronts(_front)

        print('n_days:', max_days)

        if incl_curr:
            planner.evaluator.current_data = ocean_current.CurrentOperator(
                _start_date, max_days)

        for p_key in best_inds:
            for _k, _ind_list in best_inds[p_key].items():
                total_fit = total_original_fit = np.zeros(2)
                for _ind in _ind_list:
                    fit = np.array(planner.tb.evaluate(_ind))
                    original_fit = np.array(_ind.fitness.values)
                    total_fit += fit
                    total_original_fit += original_fit

                file_key = list(_ID_dict.keys())[f_idx]
                print('File {} Path {} Obj {:>4} real'.format(file_key, p_key, _k), np.round(total_fit, 2))
                # print('File {} Path {} Obj {:>4} orig'.format(file_key, _p, _k), np.round(total_original_fit, 2))
                # print('DIFF ORIGINAL: {}'.format(np.subtract(_ind.fitness.values, fit)))

        all_results.append(_result)
        all_best_inds.append(best_inds)

    plotter.plot_routes(all_results, all_best_inds)

    plt.show()
