import numpy as np
from scipy.spatial.distance import cdist
from deap.tools.indicator import hv
from deap.tools import ParetoFront


def hypervolume(front):
    obj = np.array([ind.fitness.values for ind in front])
    ref = np.max(obj, axis=0) + 1

    return hv.hypervolume(obj, ref)


def binary_hypervolume(A, B):
    frontAB = ParetoFront()
    frontAB.update(A.items)
    frontAB.update(B.items)

    objAB = np.array([ind.fitness.values for ind in frontAB])
    objB = np.array([ind.fitness.values for ind in B])
    ref = np.max(objAB, axis=0) + 1

    hvAB = hv.hypervolume(objAB, ref)
    hvB = hv.hypervolume(objB, ref)

    return hvAB - hvB


def triple_hypervolume(A, B, C):
    frontBC = ParetoFront()
    frontBC.update(B.items)
    frontBC.update(C.items)

    frontABC = ParetoFront()
    frontABC.update(A.items)
    frontABC.update(frontBC.items)

    objABC = np.array([ind.fitness.values for ind in frontABC])
    objBC = np.array([ind.fitness.values for ind in frontBC])
    ref = np.max(objABC, axis=0) + 1

    hvABC = hv.hypervolume(objABC, ref)
    hvBC = hv.hypervolume(objBC, ref)

    return hvABC - hvBC


def two_sets_coverage(A, B, pop=True):
    objA = np.array([ind.fitness.values for ind in A]) if pop else A
    objB = np.array([ind.fitness.values for ind in B]) if pop else B

    weakly_dominated_B = 0
    for fitIndB in objB:
        B_weakly_dominated = False
        for fitIndA in objA:
            A_weakly_dominates = False
            for valA, valB in zip(fitIndA, fitIndB):
                if valA < valB:
                    A_weakly_dominates = True
                    break
            if A_weakly_dominates:
                B_weakly_dominated = True
                break
        if B_weakly_dominated:
            weakly_dominated_B += 1

    return weakly_dominated_B / len(B)


def generational_distance(pop, ref):
    if len(pop) > 0:
        refFits = np.array([ind.fitness.values for ind in ref])
        popFits = np.array([ind.fitness.values for ind in pop])
        dist = cdist(popFits, refFits, metric='euclidean')
        return np.average(np.min(dist, axis=0))
    else:
        return 1e+6


