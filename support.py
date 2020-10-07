import functools
import gc
import numpy as np

from datetime import datetime
from deap import tools
from shapely.geometry import Polygon


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


# Function to be replaced in Pareto front class: returns current local front and nr. (non-)dominated hofers
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


antarctic_circle = Polygon([(-180, -66), (180, -66), (180, -89), (-180, -89)])
arctic_circle = Polygon([(-180, 66), (180, 66), (180, 89), (-180, 89)])

locations = {'Agios Nikolaos': (25.726617, 35.152255),
             'Banjul': (-16.852247, 13.532402),
             'Bergen': (5.015621, 60.145956),
             'Brazil': (-23.4166, -7.2574),
             'Brunswick': (-81.322738, 31.106695),
             'Canada': (-53.306878, 46.423969),
             'Cape Town': (18.448971, -33.897487),
             'Caribbean Sea': (-72.3352, 12.8774),
             'Concepcion': (-73.231175, -36.830794),
             'Curacao': (-68.918803, 12.295014),
             'Current1': (-5, 0),
             'Current2': (5, 0),
             'Davao': (125.578056, 6.989483),
             'Dubai': (55.028659, 25.046533),
             'ECA1: Jacksonville': (-78.044447, 27.616446),
             'ECA2: New York': (-67.871890, 40.049950),
             'Flekkefjord': (6.609681, 58.186598),
             'Floro': (4.981360, 61.606503),
             'Freetown': (-13.819225, 8.366437),
             'Gulf of Aden': (48.1425, 12.5489),
             'Gulf of Bothnia': (20.89, 58.46),
             'Gulf of Guinea': (3.14516, 4.68508),
             'Gulf of Mexico': (-94.5968, 26.7012),
             'Halifax': (-63.497639, 44.572532),
             'Havana': (-82.395940, 23.205105),
             'Houston': (-94.657976, 29.348557),
             'Jakarta': (106.829955, -6.013374),
             'Keelung': (121.75, 25.15),
             'KeelungC': (123, 26),
             'Lima': (-77.165124, -12.052136),
             'Luanda': (12.175094, -8.692481),
             'Kristiansand': (8.092751, 58.063953),
             'Malta': (14.061035, 34.996707),
             'Miami': (-75.724478, 26.152992),
             'Mediterranean Sea': (29.188952, 32.842985),
             'New York': (-71.143, 40.356),
             'Normandy': (-5.145, 49.211),
             'North UK': (3.3, 60),
             'Panama North': (-79.931675, 9.472317),
             'Paramaribo': (-55.218390, 5.956055),
             'Perth': (115.027243, -32.071678),
             'Plymouth': (-4.162378, 50.328661),
             'Rotterdam': (4.02, 52.01),
             'Salvador': (-38.585086, -13.057614),
             'Santander': (-3.746229, 43.482803),
             'San Francisco': (-123, 37.75),
             'Sao Paulo': (-46.288550, -24.267331),
             'Singapore': (103.746969, 1.141331),
             'South UK': (-7.5, 47),
             'Sri Lanka': (78, 5),
             'Stavanger': (5.555933, 58.968088),
             'Thessaloniki': (22.933677, 40.616297),
             'Tokyo': (139, 34),
             'Yemen': (49, 12),
             'Valencia': (-0.188091, 39.464972),
             'Veracruz': (-96.100615, 19.203244),
             'Wellington': (174.814171, -41.486011),
             }

# Test weather
# Weather locations

# locations['Thessaloniki']
# locations['Agios Nikolaos']

# Test currents
inputSalLim = {'instance': 'SalLim', 'input': {'from': [('S', locations['Salvador'])], 'to': [('L', locations['Lima'])],
                                               'departureDates': [datetime(2014, 11, 11)]}}

def clear_caches():
    gc.collect()
    wrappers = [a for a in gc.get_objects() if isinstance(a, functools._lru_cache_wrapper)]

    for wrapper in wrappers:
        if wrapper.cache_info()[1] > 0:
            wrapper.cache_clear()


if __name__ == '__main__':
    # Current locations
    westBot, westTop, eastBot, eastTop = np.array([-72, 32]), np.array([-74, 39]), np.array([-50, 38]), np.array(
        [-55, 46])

    west = np.array([westBot, westTop])
    east = np.array([eastBot, eastTop])

    lenWest = np.linalg.norm(west)
    lenEast = np.linalg.norm(east)

    westVec = westTop - westBot
    eastVec = eastTop - eastBot

    westNorm = westVec / lenWest
    eastNorm = eastVec / lenEast

    locRange = np.linspace(0, 1, 4)
    westLocations, eastLocations = [], []
    for f in locRange:
        eastLocations.append(tuple(f * eastVec + eastBot))
        westLocations.append(tuple(f * westVec + westBot))

    westLocations = westLocations[1:3]
    eastLocations = eastLocations[0] + eastLocations[-1]

    print(eastLocations, '\n', westLocations)


