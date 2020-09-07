import numpy as np

from deap import tools


def find_closest(A, target):
    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


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