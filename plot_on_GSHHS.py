from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
import pandas as pd
import pickle
from math import floor, ceil
import numpy as np


def draw_basemap(individuals, vessel):
    wps = []
    for ind in individuals:
        wps.extend([item[0] for item in ind])

    min_x, min_y = min(wps, key=lambda t: t[0])[0], min(wps, key=lambda t: t[1])[1]
    max_x, max_y = max(wps, key=lambda t: t[0])[0], max(wps, key=lambda t: t[1])[1]
    m = 5
    left, right, bottom, top = max(floor(min_x) - m, -180), min(ceil(max_x) + m, 180), \
                               max(floor(min_y) - m, -90), min(ceil(max_y) + m, 90)

    m = Basemap(projection='merc', resolution='c', llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left, urcrnrlon=right)
    m.drawparallels(np.arange(-90., 90., 10.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 180., 10.), labels=[0, 0, 0, 1], fontsize=10)
    m.drawcoastlines()
    m.fillcontinents()
    # m.bluemarble()

    # Create colorbar
    col_map = cm.jet
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
            m.drawgreatcircle(e[0][0], e[0][1], e[1][0], e[1][1], linewidth=1, color=col_map(normalized_speeds[i]),)

        # Plot waypoints
        for i, (x, y) in enumerate(waypoints):
            m.scatter(x, y, latlon=True, color=col_map(normalized_speeds[i]), marker='o', s=2)


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
    idx = 0
    for i in range(2):
        with open('C:/dev/projects/naval_route_planner/output/path_0/ 22_13_27_population_sub_path_{}'.format(i), 'rb') as f:
            population = pickle.load(f)
        individual = population[idx]

        populations.append(population)
        individuals.append(individual)

        print('Speeds: ', [item[1] for item in individual])
        print(individual.fitness.values)
    draw_basemap(individuals, vessel)
    # draw_basemap([[[(-5.352121, 48.021295), 10], [(-72.984139, 40.235248), None]]], vessel)
    plt.show()
