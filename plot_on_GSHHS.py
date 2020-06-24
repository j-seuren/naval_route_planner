from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import pickle
import math
import numpy as np
import ocean_current


def draw_basemap(individuals, vessel, m=None, uin=None, vin=None, lons=None, lats=None):
    col_map = cm.rainbow

    if not m:
        wps = []
        for ind in individuals:
            wps.extend([item[0] for item in ind])

        min_x, min_y = min(wps, key=lambda t: t[0])[0], min(wps, key=lambda t: t[1])[1]
        max_x, max_y = max(wps, key=lambda t: t[0])[0], max(wps, key=lambda t: t[1])[1]
        margin = 5
        # left, right, bottom, top = max(math.floor(min_x) - margin, -180), min(math.ceil(max_x) + margin, 180), \
        #                            max(math.floor(min_y) - margin, -90), min(math.ceil(max_y) + margin, 90)

        left, right, bottom, top = -179, 179, -75, 75

        m = Basemap(projection='merc', resolution='c', llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left, urcrnrlon=right)
        m.drawparallels(np.arange(-90., 90., 10.), labels=[1, 0, 0, 0], fontsize=10)
        m.drawmeridians(np.arange(-180., 180., 10.), labels=[0, 0, 0, 1], fontsize=10)
        m.drawcoastlines()
        m.fillcontinents()

        # # Transform vector and coordinate data
        # vec_lon = int(uin.shape[1] / (2 * 360 / (right - left)))
        # vec_lat = int(uin.shape[0] / (2 * 180 / (top - bottom)))
        # u_rot, v_rot, x, y = m.transform_vector(uin, vin, lons, lats, vec_lon, vec_lat, returnxy=True)
        #
        # vec_plot = m.quiver(x, y, u_rot, v_rot, color='darkgray', scale=50, width=.002)
        # plt.quiverkey(vec_plot, 0.2, -0.2, 1, '1 knot', labelpos='W')  # Position and reference label

        # Create color bar
        sm = plt.cm.ScalarMappable(cmap=col_map)
        col_bar = plt.colorbar(sm, norm=plt.Normalize(vmin=min(vessel.speeds), vmax=max(vessel.speeds)))
        max_s = max(vessel.speeds)
        min_s = min(vessel.speeds)
        col_bar.ax.set_yticklabels(['{}'.format(min_s),
                                    '{}'.format(round((1 / 5) * (max_s - min_s) + min_s, 1)),
                                    '{}'.format(round((2 / 5) * (max_s - min_s) + min_s, 1)),
                                    '{}'.format(round((3 / 5) * (max_s - min_s) + min_s, 1)),
                                    '{}'.format(round((4 / 5) * (max_s - min_s) + min_s, 1)),
                                    '{}'.format(round(max_s, 1))])

    for ind in individuals:
        # Create colors
        true_speeds = [item[1] for item in ind[:-1]]
        normalized_speeds = [(speed - min(vessel.speeds)) / (max(vessel.speeds) - min(vessel.speeds))
                             for speed in true_speeds] + [0]

        # Plot edges
        waypoints = [item[0] for item in ind]
        edges = zip(waypoints[:-1], waypoints[1:])
        for i, e in enumerate(edges):
            m.drawgreatcircle(e[0][0], e[0][1], e[1][0], e[1][1], linewidth=2, color=col_map(normalized_speeds[i]), zorder=1)

        for i, (x, y) in enumerate(waypoints):
            m.scatter(x, y, latlon=True, color='dimgray', marker='o', s=5, zorder=2)

    return m


if __name__ == "__main__":
    from deap import base, creator

    # Create Fitness and Individual types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)


    class Vessel:
        def __init__(self, name, _speeds, _fuel_rates):
            self.name = name
            self.speeds = _speeds
            self.fuel_rates = _fuel_rates


    # Vessel characteristics
    vessel_name = 'Fairmaster'
    table = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name=vessel_name)
    speeds = [round(speed, 1) for speed in table['Speed']]
    fuel_rates = {speed: round(table['Fuel'][i], 1) for i, speed in enumerate(speeds)}
    vessel = Vessel(vessel_name, speeds, fuel_rates)

    populations = []
    individuals = []
    for i in range(2):
        with open('output/path_0/14_43_17_population_sub_path_{}'.format(i), 'rb') as f:
            population = pickle.load(f)

        min_tt, min_fc = 9999999999, 99999999999
        tt_ind, fc_ind = None, None
        for ind in population:
            travel_time, fuel_consumption = ind.fitness.values
            if travel_time < min_tt:
                tt_ind = ind
                min_tt = travel_time
            if fuel_consumption < min_fc:
                fc_ind = ind
                min_fc = fuel_consumption

        # individuals.append(fc_ind)
        individuals.append(tt_ind)
        print('Fuel individual:', fc_ind.fitness.values)
        print('Travel time individual: ', tt_ind.fitness.values)

        populations.append(population)
    # u_out, v_out, lons, lats = ocean_current.read_netcdf('20160101')
    m = draw_basemap(individuals, vessel)

    populations = []
    individuals = []
    for i in range(1):
        with open('output/path_1/14_43_17_population_sub_path_0', 'rb') as f:
            population = pickle.load(f)

        min_tt, min_fc = 9999999999, 99999999999
        tt_ind, fc_ind = None, None
        for ind in population:
            travel_time, fuel_consumption = ind.fitness.values
            if travel_time < min_tt:
                tt_ind = ind
                min_tt = travel_time
            if fuel_consumption < min_fc:
                fc_ind = ind
                min_fc = fuel_consumption

        # individuals.append(fc_ind)
        individuals.append(tt_ind)
        print('Fuel individual:', fc_ind.fitness.values)
        print('Travel time individual: ', tt_ind.fitness.values)

        populations.append(population)
    draw_basemap(individuals, vessel, m=m)

    plt.show()
