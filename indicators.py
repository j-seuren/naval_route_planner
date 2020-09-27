import numpy as np
from scipy.spatial.distance import cdist
from deap.tools.indicator import hv


def hypervolume(front):
    wobj = np.array([ind.fitness.wvalues for ind in front]) * -1
    ref = np.max(wobj, axis=0) + 1

    return hv.hypervolume(wobj, ref)


def generational_distance(pop, ref):
    if len(pop) > 0:
        refFits = np.array([ind.fitness.values for ind in ref])
        popFits = np.array([ind.fitness.values for ind in pop])
        dist = cdist(popFits, refFits, metric='euclidean')
        return np.average(np.min(dist, axis=0))
    else:
        return 1e+6
