import os
import numpy as np
import pandas
import pickle

from data_config import katana
from deap import tools
from shapely.prepared import prep
from shapely.strtree import STRtree


class Vessel:
    def __init__(self, name='Fairmaster'):
        self.name = name
        speed_table_path = os.path.abspath('data/speed_table.xlsx')
        table = pandas.read_excel(speed_table_path,
                                  sheet_name=self.name)
        self.speeds = [round(speed, 1) for speed in table['Speed']]
        self.fuel_rates = {speed: round(table['Fuel'][i], 1)
                           for i, speed in enumerate(self.speeds)}


def logbook():
    log = tools.Logbook()
    log.header = "gen", "evals", "fitness", "size"
    log.chapters["fitness"].header = "min", "avg", "max"
    log.chapters["size"].header = "min", "avg", "max"
    return log


def statistics():
    stats_fit = tools.Statistics(lambda _ind: _ind.fitness.values)
    stats_size = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean, axis=0)
    mstats.register("std", np.std, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)
    return mstats


def populate_rtree(fp, res='c', spl_th=None):
    # Import land obstacles as polygons
    if os.path.exists(fp):
        with open(fp, 'rb') as f:
            polys = pickle.load(f)
    elif spl_th:
        polys = katana.get_split_polygons(res, spl_th)
    else:
        raise FileNotFoundError

    # Prepare polygons
    # polys = list(map(prep, polys))

    # Populate R-tree index with bounds of polygons
    tree = STRtree(polys)

    return tree


def assign_crowding_dist(individuals):
    """Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].fitness.values)

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, _next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (_next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist
