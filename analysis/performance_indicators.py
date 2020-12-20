import numpy as np
from scipy.spatial.distance import cdist
from deap.tools.indicator import hv
from deap.tools import ParetoFront


def hypervolume(front):
    obj = np.array([ind.fitness.values for ind in front])
    ref = np.max(obj, axis=0) + 1

    return hv.hypervolume(obj, ref)


def binary_hypervolume_ratio(A, B):
    AB = ParetoFront()
    AB.update(A.items)
    AB.update(B.items)

    fitAB = np.array([fit.values for fit in AB.keys])
    fitA = np.array([fit.values for fit in A.keys])
    ref = np.max(fitAB, axis=0) + 1

    hvAB = hv.hypervolume(fitAB, ref)
    hvA = hv.hypervolume(fitA, ref)

    return hvA / hvAB


def triple_hypervolume_ratio(A, B, C):
    ABC = ParetoFront()
    ABC.update(A.items)
    ABC.update(B.items)
    ABC.update(C.items)

    fitABC = np.array([fit.values for fit in ABC.keys])
    fitA = np.array([fit.values for fit in A.keys])
    ref = np.max(fitABC, axis=0) + 1

    hvABC = hv.hypervolume(fitABC, ref)
    hvA = hv.hypervolume(fitA, ref)

    return hvA / hvABC


def binary_ternary_hv_ratio(A, B, C):
    AB = ParetoFront()
    AB.update(A.items)
    AB.update(B.items)
    fitAB = np.array([fit.values for fit in AB.keys])

    ABC = ParetoFront()
    ABC.update(AB.items)
    ABC.update(C.items)
    fitABC = np.array([fit.values for fit in ABC.keys])

    AC = ParetoFront()
    AC.update(A.items)
    AC.update(C.items)
    fitAC = np.array([fit.values for fit in AC.keys])

    BC = ParetoFront()
    BC.update(B.items)
    BC.update(C.items)
    fitBC = np.array([fit.values for fit in BC.keys])

    fitA = np.array([fit.values for fit in A.keys])
    fitB = np.array([fit.values for fit in B.keys])
    fitC = np.array([fit.values for fit in C.keys])

    refABC = np.max(fitABC, axis=0) + 1
    refAB = np.max(fitAB, axis=0) + 1
    refAC = np.max(fitAC, axis=0) + 1
    refBC = np.max(fitBC, axis=0) + 1

    hvABC = hv.hypervolume(fitABC, refABC)
    hvAB = hv.hypervolume(fitAB, refAB)
    hvAC = hv.hypervolume(fitAC, refAC)
    hvBC = hv.hypervolume(fitBC, refBC)

    bHV_AB = hv.hypervolume(fitA, refAB) / hvAB
    bHV_AC = hv.hypervolume(fitA, refAC) / hvAC
    bHV_BA = hv.hypervolume(fitB, refAB) / hvAB
    bHV_BC = hv.hypervolume(fitB, refBC) / hvBC
    bHV_CA = hv.hypervolume(fitC, refAC) / hvAC
    bHV_CB = hv.hypervolume(fitC, refBC) / hvBC

    tHV_A = hv.hypervolume(fitA, refABC) / hvABC
    tHV_B = hv.hypervolume(fitB, refABC) / hvABC
    tHV_C = hv.hypervolume(fitC, refABC) / hvABC

    return [tHV_A, tHV_B, tHV_C], [bHV_AB, bHV_AC, bHV_BA, bHV_BC, bHV_CA, bHV_CB]


def two_sets_coverage(A, B):
    """
    Input DEAP ParetoFronts
    """

    weakly_dominated_B = 0
    fitsA = [fit.values for fit in A.keys]
    fitsB = [fit.values for fit in B.keys]
    for fitB in fitsB:
        B_weakly_dominated = False
        for fitA in fitsA:
            A_weakly_dominates = True
            not_equal = False
            for valA, valB in zip(fitA, fitB):
                if valA > valB:
                    A_weakly_dominates = False
                    not_equal = True
                    break
                elif valA < valB:
                    not_equal = True
            if A_weakly_dominates and not_equal:
                B_weakly_dominated = True
                break
        if B_weakly_dominated:
            weakly_dominated_B += 1

    C = weakly_dominated_B / len(B)
    return C


def generational_distance(pop, ref):
    if len(pop) > 0:
        refFits = np.array([ind.fitness.values for ind in ref])
        popFits = np.array([ind.fitness.values for ind in pop])
        dist = cdist(popFits, refFits, metric='euclidean')
        return np.average(np.min(dist, axis=0))
    else:
        return 1e+6


