import math
import numpy as np
import os
import pandas

from deap import tools


class KwonsMethod:
    def __init__(self, block, loading, Lpp, volume):
        self.g = 9.81  # gravitational acceleration [m/s^2]
        self.Lpp = Lpp
        self.vol = volume
        self.binsTWA = [30, 60, 150, 180]
        self.betaFuncs = [
            lambda BN: 1,  # Head sea and wind
            lambda BN: (1.7 - 0.03 * (BN - 4) ** 2) / 2,  # Bow sea and wind
            lambda BN: (0.9 - 0.06 * (BN - 6) ** 2) / 2,  # Beam sea and wind
            lambda BN: (0.4 - 0.03 * (BN - 8) ** 2) / 2,  # Following sea and wind
            ]
        UFuncsNormal = [
            lambda Fn: 1.7 - 1.4 * Fn + 7.4 * (Fn ** 2),  # Cb: 0.55
            lambda Fn: 2.2 - 2.5 * Fn + 9.7 * (Fn ** 2),  # Cb: 0.60
            lambda Fn: 2.6 - 3.7 * Fn + 11.6 * (Fn ** 2),  # Cb: 0.65
            lambda Fn: 3.1 - 5.3 * Fn + 12.4 * (Fn ** 2),  # Cb: 0.70
            lambda Fn: 2.4 - 10.6 * Fn + 9.5 * (Fn ** 2),  # Cb: 0.75
            lambda Fn: 2.6 - 13.1 * Fn + 15.1 * (Fn ** 2),  # Cb: 0.80
            lambda Fn: 3.1 - 18.7 * Fn + 28 * (Fn ** 2),  # Cb: 0.85
            ]
        UFuncsBallast = [
            lambda Fn: 2.6 - 12.5 * Fn + 13.5 * (Fn ** 2),  # Cb: 0.75, ballast
            lambda Fn: 3.0 - 16.3 * Fn + 21.6 * (Fn ** 2),  # Cb: 0.80, ballast
            lambda Fn: 3.4 - 20.9 * Fn + 31.8 * (Fn ** 2),  # Cb: 0.85, ballast
            ]

        if loading == 'normal':
            blockList = np.asarray([0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
            idxBlock = find_closest(blockList, block)
            self.UFunc = UFuncsNormal[int(idxBlock)]
            self.formFunc = lambda BN, Delta: 0.5 * BN + math.pow(BN, 6.5) / (2.7 * math.pow(Delta, 2/3))
        else:
            blockList = [0.75, 0.8, 0.85]
            idxBlock = int(np.digitize(block, blockList))
            self.UFunc = UFuncsBallast[idxBlock]
            self.formFunc = lambda BN, Delta: 0.7 * BN + math.pow(BN, 6.5) / (2.7 * math.pow(Delta, 2 / 3))

    def reduced_speed(self, TWD, heading, BN, boat_speed):
        # Speed direction reduction coefficient (beta) from BN and TWA
        # (Kwon, 2008)
        TWA = (TWD - heading + 180) % 360 - 180  # [-180, 180] degrees]
        idxTWA = int(np.digitize(abs(TWA), self.binsTWA))
        # print('TWA + idx', TWA, idxTWA)
        beta = self.betaFuncs[idxTWA](BN)

        # Speed reduction coefficient (U) from Froude number (Fn)

        Fn = (boat_speed * 0.514444) / math.sqrt(self.Lpp * self.g)
        # print('FN', Fn)
        U = self.UFunc(Fn)

        # Ship form coefficient from BN and displaced volume
        form = self.formFunc(BN, self.vol * self.g)
        # print(1 - (beta * U * form) / 100)
        relativeSpeedLoss = (beta * U * form) / 100
        if relativeSpeedLoss > .99:
            relativeSpeedLoss = .99
        # print('beta', round(beta, 1), 'U', round(U, 1), 'form', round(form, 1))
        # print(relativeSpeedLoss)
        return boat_speed * (1 - relativeSpeedLoss)


def find_closest(A, target):
    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


class Vessel:
    def __init__(self, name='Fairmaster', loading='normal'):
        self.name = name
        speed_table_path = os.path.abspath('data/speed_table.xlsx')
        table = pandas.read_excel(speed_table_path,
                                  sheet_name=self.name)
        self.speeds = [round(speed, 1) for speed in table['Speed']]
        self.fuel_rates = {speed: round(table['Fuel'][i], 1)
                           for i, speed in enumerate(self.speeds)}

        Lpp = 150  # Ship length between perpendiculars [m]
        B = 27  # Ship breadth [m]
        D = 8  # Ship draft [m]
        vol = 27000  # Volume of displacement [m^3]
        block = vol / (Lpp * B * D)  # Block coefficient
        self.speed_reduction = KwonsMethod(block, loading, Lpp, vol)

    def reduced_speed(self, boat_speed, BN, TWD, heading):
        return self.speed_reduction.reduced_speed(TWD, heading, BN, boat_speed)


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
    mstats.register("var", np.var, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)
    return mstats


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


# Function to be replaced in Pareto front class
def update(front, population, prevLocalFront):
    """Updates Pareto front with individuals from population and extracts dominance relationships of
     previous local Pareto front and current local Pareto front"""
    dominatedInds, dominatedHofers, currLocalFront = [], [], []

    # Create list of indices of hall-of-famers in prevParetoFront
    indPrevHofers = [i for i, hofer in enumerate(front) if hofer in prevLocalFront]

    for ind in population:
        isDominated = False
        dominatesOne = False
        hasTwin = False
        toRemove = []

        # # First, check dominance with global Pareto front.
        # # If individual is non-dominated by global front,
        # # then check dominance with local Pareto front (next loop)
        for i, hofer in enumerate(front):
            if i in indPrevHofers:  # Check only for global Pareto front
                continue

            # If the hofer dominates the individual,
            # and the individual does not dominate another hofer,
            # then the individual is marked as dominated
            if not dominatesOne and hofer.fitness.dominates(ind.fitness):
                isDominated = True
                break

            # Else if, the individual dominates a hofer,
            # then the hofer needs to be removed
            elif ind.fitness.dominates(hofer.fitness):
                dominatesOne = True
                toRemove.append(i)

            # Else if, the fitness of both are similar, we have a twin
            elif ind.fitness == hofer.fitness and front.similar(ind, hofer):
                hasTwin = True
                break

        # Individual is non-dominated by GLOBAL Pareto front,
        # hence, we add individual to current LOCAL Pareto front.
        currLocalFront.append(ind)

        # # Checking dominance with LOCAL Pareto front
        # Boolean indicating that individual dominates at least one in LOCAL Pareto Front
        dominatesOnePrev = False
        for i, hofer in enumerate(front):
            if i not in indPrevHofers:  # Check only for local Pareto front
                continue

            # If individual does not dominate at least one hofer
            # in LOCAL Pareto front, and hofer dominates individual,
            # then append to dominated individuals list.
            if not dominatesOnePrev and hofer.fitness.dominates(ind.fitness):
                dominatedInds.append(ind)

                # If individual does not dominate at least one hofer
                # in GLOBAL Pareto front, it is dominated
                if not dominatesOne:
                    isDominated = True
                break

            # Else if individual dominates hofer in LOCAL Pareto front,
            # List hofer for removal and append to dominated hofers list,
            # if not already in list
            elif ind.fitness.dominates(hofer.fitness):
                dominatesOnePrev = True
                # If not in dominated hofers list, append hofer
                if hofer not in dominatedHofers:
                    dominatedHofers.append(hofer)
                toRemove.append(i)
            elif ind.fitness == hofer.fitness and front.similar(ind, hofer):
                hasTwin = True
                break

        for i in sorted(toRemove, reverse=True):  # Remove the dominated hofer
            front.remove(i)
        if not isDominated and not hasTwin:
            front.insert(ind)

    return currLocalFront, len(dominatedHofers), len(dominatedInds)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    brng = 194.33649
    twd = 60
    vessel = Vessel()
    _speed = 15  # knots
    BNs = list(range(1, 13))
    newSpeeds = []
    for bn in BNs:
        newSpeeds.append(vessel.reduced_speed(_speed, bn, twd, brng))

    plt.plot(BNs, newSpeeds)

    plt.show()
