import numpy as np
from scipy.spatial.distance import cdist
from deap.tools.indicator import hv


def hypervolume(front):
    obj = np.array([ind.fitness.values for ind in front])
    ref = np.max(obj, axis=0) + 1

    return hv.hypervolume(obj, ref)


def binary_hypervolume(A, B):
    objA = np.array([ind.fitness.values for ind in A])
    objB = np.array([ind.fitness.values for ind in B])
    objAB = np.append(objA, objB, axis=0)
    ref = np.max(objAB, axis=0) + 1

    hvAB = hv.hypervolume(objAB, ref)
    hvB = hv.hypervolume(objB, ref)

    return hvAB - hvB


def two_sets_coverage(A, B):
    objA = np.array([ind.fitness.values for ind in A])
    objB = np.array([ind.fitness.values for ind in B])

    dominatedB = 0
    for fitIndB in objB:
        B_dominated = False
        for fitIndA in objA:
            A_dominates = True
            for valA, valB in zip(fitIndA, fitIndB):
                if valA >= valB:
                    A_dominates = False
            if A_dominates:
                B_dominated = True
                break
        if B_dominated:
            dominatedB += 1

    return dominatedB / len(B)


def generational_distance(pop, ref):
    if len(pop) > 0:
        refFits = np.array([ind.fitness.values for ind in ref])
        popFits = np.array([ind.fitness.values for ind in pop])
        dist = cdist(popFits, refFits, metric='euclidean')
        return np.average(np.min(dist, axis=0))
    else:
        return 1e+6


