import functools
import random
import numpy as np
import more_itertools as mit

from math import sqrt, pi, cos, sin
from scipy.spatial import distance


def star(f):
    @functools.wraps(f)
    def f_inner(args):
        return f(*args)
    return f_inner


class Operators:
    def __init__(self,
                 e_feasible,
                 vessel,
                 geod,
                 par
                 ):
        self.e_feasible = e_feasible
        self.vessel = vessel                    # Vessel class instance
        self.geod = geod                        # GreatCircle class instance
        self.radius = par['radius']             # Move radius
        self.widthRatio = par['widthRatio']     # Insert width ratio
        self.scaleFactor = par['scaleFactor']
        self.delFactor = par['delFactor']
        self.ops = par['mutationOperators']     # Mutation operators list
        self.gauss = par['gauss']
        self.nMutations = par['nMutations']
        self.moves = np.zeros(len(self.ops))

    def mutate(self, ind, initializing=False, k=None):
        inverseWeights = np.round(self.moves / np.linalg.norm(self.moves), 2)
        weights = 1 - inverseWeights
        weights[-1] = weights[-1] * self.delFactor
        if initializing:
            weights[-1] = min(self.nMutations, 5)
            k = self.nMutations
        if k is None:
            k = random.randint(1, self.nMutations)
        sample_ops = random.choices(self.ops, weights=weights, k=k)
        while sample_ops:
            op = sample_ops.pop()
            if op == 'move' and len(ind) > 2:
                self.move_wp(ind, initializing)
            elif op == 'delete' and len(ind) > 2:
                self.delete_wps(ind, initializing)
            elif op == 'speed':
                self.change_speeds(ind)
            elif op == 'insert':
                self.insert_wps(ind, initializing)
        del ind.fitness.values
        return ind,

    def insert_wps(self, ind, initializing=False, shape='rhombus'):
        weights = [self.geod.distance(ind[i][0], ind[i+1][0]) for i, leg in enumerate(ind[:-1])]
        # Draw gamma int for number of inserted waypoints
        edges = self.sample_sequence(stop=len(ind)-1, weights=weights)

        for i in edges:
            p1, p2 = np.asarray(ind[i][0]), np.asarray(ind[i+1][0])
            assert not np.array_equal(p1, p2) != 0, 'p1 and p2 are equal: {}{}'.format(p1, p2)

            ctr = (p1 + p2) / 2.  # Edge centre
            major = np.linalg.norm(p1 - p2) / 2.  # Ellipse width
            minor = major * self.widthRatio / 2.  # Ellipse height

            trials = 0
            while trials < 100:
                if self.gauss or shape == 'ellipse':
                    if self.gauss:
                        mu, cov = np.zeros(2), np.identity(2) * 0.16691704223  # 95% CI = unit circle
                        xy = np.random.multivariate_normal(mu, cov, 1).squeeze()
                    else:
                        # Generate a random point inside a circle of radius 1
                        rho = random.random()
                        phi = random.random() * 2. * pi
                        xy = sqrt(rho) * np.array([cos(phi), sin(phi)])

                    # Scale x and y to the dimensions of the ellipse
                    ptOrigin = xy * np.array([major, minor])
                elif shape == 'rhombus':
                    v1 = np.array([major, - minor])
                    v2 = np.array([major, minor])
                    offset = np.array([major, .0])
                    ptOrigin = random.random() * v1 + random.random() * v2 - offset
                else:
                    raise ValueError("Shape argument invalid: try 'ellipse' or 'rhombus'")

                # Transform
                dx, dy = p2 - p1
                theta = np.arctan2(dy, dx)
                cosTh, sinTh = np.cos(theta), np.sin(theta)
                rotation_matrix = np.array([[cosTh, -sinTh], [sinTh, cosTh]])
                x, y = ctr + np.dot(rotation_matrix, ptOrigin)
                x -= np.int(x / 180) * 360
                y = np.clip(y, -89.9, 89.9)

                newWP = (x, y)

                if initializing:
                    if self.e_feasible(ind[i][0], newWP) and self.e_feasible(newWP, ind[i+1][0]):
                        ind.insert(i+1, [newWP, ind[i][1]])
                        self.moves[-3] += 1
                        return
                    else:
                        trials += 1
                else:
                    self.moves[-3] += 1
                    ind.insert(i+1, [newWP, ind[i][1]])
                    return
            print('insert trials exceeded')

    def move_wp(self, ind, initializing=False):
        # Find waypoints with largest angles
        legs = list(zip([leg[0] for leg in ind[:-1]], [leg[0] for leg in ind[1:]]))
        bearings = np.array([self.geod.distance(leg[0], leg[1], bearing=True)[1] for leg in legs])    # Get bearings
        normDeg = (bearings[:-1] - bearings[1:]) % 360  # Normalize the difference
        absDiffsDeg = np.minimum(360 - normDeg, normDeg)  # in range [0, 180]

        # Draw gamma int for number of to be moved (tbm) waypoints
        tbm = self.sample_sequence(start=1, stop=len(ind)-1, reverse=False, weights=absDiffsDeg)
        for i in tbm:
            p = np.asarray(ind[i][0])
            trials = 0
            while True:
                if self.gauss:
                    mu, cov = p, np.identity(2) * self.radius * 0.16691704223  # 95% CI = circle centered at p
                    lon, lat = np.random.multivariate_normal(mu, cov, 1).squeeze().T
                else:
                    # Pick random location within a radius from the current waypoint
                    # Square root ensures a uniform distribution inside the circle.
                    r = self.radius * sqrt(random.random())
                    phi = random.random() * 2 * pi
                    lon, lat = p + r * np.array([cos(phi), sin(phi)])

                lon -= np.int(lon / 180) * 360
                lat = np.clip(lat, -89.9, 89.9)
                newWP = (lon, lat)

                # Ensure feasibility during initialization
                if initializing and (not self.e_feasible(ind[i-1][0], newWP) or
                                     not self.e_feasible(newWP, ind[i+1][0])):
                    trials += 1
                    if trials > 100:
                        print('move trials exceeded', end='\n ')
                        break
                    else:
                        continue
                ind[i][0] = newWP
                self.moves[-2] += 1
                break
    
    def delete_wps(self, ind, initializing=False):
        # Draw gamma int for number of to be deleted (tbd) waypoints
        tbd = self.sample_sequence(start=1, stop=len(ind)-1)

        if initializing:  # Ensure feasibility
            # Group consecutive tbd waypoints
            for group in mit.consecutive_groups(tbd):
                group = list(group)
                a, b = group[0], group[-1]
                if not self.e_feasible(ind[a-1][0], ind[b+1][0]):
                    i, j = tbd.index(a), tbd.index(b)
                    del tbd[i:j+1]
        for i in tbd:
            del ind[i]
            self.moves[-1] += 1
            while ind[i-1][0] == ind[i][0]:
                assert i < len(ind) - 1, 'last waypoint cannot be deleted'
                del ind[i]
                self.moves[-1] += 1

    def change_speeds(self, ind):
        size = len(ind)
        k = random.randrange(1, size)
        i = random.randrange(size-k)
        assert i+k < size
        avgCurrentSpeed = sum([wp[1] for wp in ind[i:i+k]]) / k
        newSpeed = random.choice([s for s in self.vessel.speeds if s != avgCurrentSpeed])
        for wp in ind[i:i+k]:
            wp[1] = newSpeed
            self.moves[0] += 1

    def sample_sequence(self, stop, start=0, reverse=True, weights=None):
        scale = stop * self.scaleFactor
        while True:
            k = np.ceil((np.random.exponential(scale=scale, size=1))).item()
            if 0 < k <= stop - start:
                break
        k, xrange = int(k), range(start, stop)
        seq = random.sample(xrange, k=k) if weights is None else random.choices(xrange, k=k)
        seq = sorted(seq, reverse=True) if reverse else seq
        return seq

    def cx_one_point(self, ind1, ind2):
        if min(len(ind1), len(ind2)) > 2:
            # Get waypoints of ind1 and ind2
            ps1 = np.array([row[0] for row in ind1])
            ps2 = np.array([row[0] for row in ind2])

            # Shuffled list of indexes of ind1, excluding start and end
            c1s = random.sample(range(1, len(ind1)-1), k=len(ind1)-2)
            tr = 0  # Trial count
            while c1s:
                tr += 1
                c1 = c1s.pop()
                # Calculate distance between c1 and each point of ind2
                p1 = np.asarray(ps1[c1]).reshape(1, -1)
                distances = distance.cdist(p1, ps2, metric=self.geod.great_circle)

                # Indices of ind2 that would sort distances
                c2s = distances.argsort()

                # Remove start and end indices
                c2s = c2s[(c2s != 0) & (c2s != len(ind2)-1)]

                # Try crossover with c1 and three nearest c2s
                for i in range(min(3, len(ind2)-2)):
                    c2 = c2s[i]

                    # If new edges not feasible, discard c2
                    e1p1, e1p2 = tuple(ps2[c2-1]), tuple(ps1[c1])
                    e2p1, e2p2 = tuple(ps1[c1-1]), tuple(ps2[c2])
                    if e1p1 == e1p2 or e2p1 == e2p2 or not (self.e_feasible(e1p1, e1p2) and
                                                            self.e_feasible(e2p1, e2p2)):
                        continue
                    else:
                        ind1[c1:], ind2[c2:] = ind2[c2:], ind1[c1:]
                    assert len(ind1) > 2 and len(ind2) > 2, 'crossover children length < 3'
                    del ind1.fitness.values
                    del ind2.fitness.values
                    return ind1, ind2
        else:
            print('skipped crossover', end='\n ')
        return ind1, ind2
