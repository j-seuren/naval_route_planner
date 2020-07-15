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
                 tb,
                 vessel,
                 gc,
                 radius=1,
                 width_ratio=0.5,
                 mutation_ops=None
                 ):
        self.tb = tb                    # Function toolbox
        self.vessel = vessel            # Vessel class instance
        self.gc = gc                    # GreatCircle class instance
        self.radius = radius            # Move radius
        self.cov = [[radius, 0],        # Covariance matrix (move)
                    [0, radius]]
        self.width_ratio = width_ratio  # Insert width ratio
        self.shape = 3
        self.scale_factor = 0.1

        # Mutation operators list
        if mutation_ops is None:
            self.ops = ['speed', 'insert', 'move', 'delete']
        else:
            self.ops = mutation_ops

    def mutate(self, ind, initializing=False, cum_weights=None, k=1):
        if cum_weights is None:
            cum_weights = [10, 20, 30, 50]  # Two times higher prob choosing del
        sample_ops = random.choices(self.ops, cum_weights=cum_weights, k=k)
        while sample_ops:
            op = sample_ops.pop()
            if op == 'move' and len(ind) > 2:
                self.move_wp(ind, initializing, gauss=False)
            elif op == 'delete' and len(ind) > 2:
                self.delete_wps(ind, initializing)
            elif op == 'speed':
                self.change_speeds(ind)
            else:
                self.insert_wps(ind, initializing, gauss=False)
        del ind.fitness.values
        return ind,

    def insert_wps(self, ind, initializing=False, gauss=False, shape='rhombus'):
        # Draw gamma int for number of inserted waypoints
        max_k = len(ind) - 1
        scale = max_k * self.scale_factor
        k = round(np.asscalar(np.random.gamma(self.shape, scale, 1)), 0)
        while not 0 < k <= max_k:
            k = round(np.asscalar(np.random.gamma(self.shape, scale, 1)), 0)
        e = random.sample(range(max_k), k=int(k))
        for e in sorted(e, reverse=True):
            p1_tup, p2_tup = ind[e][0], ind[e+1][0]
            p1, p2 = np.asarray(p1_tup), np.asarray(p2_tup)
            ctr = (p1 + p2) / 2  # Edge centre
            width = np.linalg.norm(p1 - p2)  # Ellipse width
            height = width * self.width_ratio  # Ellipse height

            trials = 0
            while trials < 100:
                if gauss or shape == 'ellipse':
                    if gauss:
                        mu, cov = [0, 0], [[1, 0], [0, 1]]
                        xy = np.random.multivariate_normal(mu, cov, 1).squeeze()
                    else:
                        # Generate a random point inside a circle of radius 1
                        rho = random.random()
                        phi = random.random() * 2 * pi
                        xy = sqrt(rho) * np.array([cos(phi), sin(phi)])

                    # Scale x and y to the dimensions of the ellipse
                    pt_origin = xy * np.array([width, height]) / 2.0
                elif shape == 'rhombus':
                    v1 = np.array([width / 2.0, - height / 2.0])
                    v2 = np.array([width / 2.0, height / 2.0])
                    offset = np.array([width / 2.0, 0.0])
                    pt_origin = random.random() * v1 + random.random() * v2 - offset
                else:
                    raise ValueError("Shape argument invalid: "
                                     "try 'ellipse' or 'rhombus'")

                # Transform
                dy, dx = p2[1] - p1[1], p2[0] - p1[0]
                if dx != 0:
                    slope = dy / dx
                    rot_x = 1 / sqrt(1 + slope ** 2)
                    rot_y = slope * rot_x
                else:
                    assert dy != 0, 'p1 and p2 are equal: {}{}'.format(p1_tup, p2_tup)
                    rot_x = 0
                    rot_y = 1

                cos_a, sin_a = rot_x, rot_y
                rotation_matrix = np.array(((cos_a, -sin_a), (sin_a, cos_a)))
                pt_rotated = np.dot(rotation_matrix, pt_origin)  # Rotate
                x, y = ctr + pt_rotated  # Translate
                x -= np.int(x / 180) * 360
                y = np.clip(y, -90, 90)
                new_wp = (x, y)

                if initializing:
                    if self.tb.e_feasible(p1_tup, new_wp) and self.tb.e_feasible(new_wp, p2_tup):
                        ind.insert(e+1, [new_wp, ind[e][1]])
                        return
                    else:
                        trials += 1
                else:
                    ind.insert(e+1, [new_wp, ind[e][1]])
                    return
            print('insert trials exceeded')

    def move_wp(self, ind, initializing=False, indpb=0.2, gauss=False):
        for i in range(1, len(ind)-1):
            if random.random() < indpb:
                wp = np.asarray(ind[i][0])
                trials = 0
                while True:
                    if gauss:
                        x, y = np.random.multivariate_normal(wp, self.cov, 1).squeeze().T
                    else:
                        # Pick random location within a radius from the current waypoint
                        # Square root ensures a uniform distribution inside the circle.
                        r = self.radius * sqrt(random.random())
                        phi = random.random() * 2 * pi
                        x, y = wp + r * np.array([cos(phi), sin(phi)])

                    x -= np.int(x / 180) * 360
                    y = np.clip(y, -90, 90)
                    new_wp = (x, y)

                    # Ensure feasibility during initialization
                    if initializing and (
                            not self.tb.e_feasible(ind[i - 1][0], new_wp) or
                            not self.tb.e_feasible(new_wp, ind[i + 1][0])):
                        trials += 1
                        if trials > 100:
                            print('move trials exceeded', end='\n ')
                            break
                        else:
                            continue
                    ind[i][0] = new_wp
                    break
    
    def delete_wps(self, ind, initializing=False):
        # Draw gamma int for number of to be deleted (tbd) waypoints
        max_k = len(ind) - 2
        scale = max_k * self.scale_factor
        k = round(np.asscalar(np.random.gamma(self.shape, scale, 1)), 0)
        while not 0 < k <= max_k:
            k = round(np.asscalar(np.random.gamma(self.shape, scale, 1)), 0)
        # Draw tbd waypoints
        tbd = sorted(random.sample(range(1, max_k+1), k=int(k)),
                     reverse=True)
        if initializing:  # Ensure feasibility
            # Group consecutive tbd waypoints
            for group in mit.consecutive_groups(tbd):
                group = list(group)
                a, b = group[0], group[-1]
                if not self.tb.e_feasible(ind[a-1][0], ind[b+1][0]):
                    i, j = tbd.index(a), tbd.index(b)
                    del tbd[i:j+1]
        for i in tbd:
            del ind[i]
            while ind[i-1][0] == ind[i][0]:
                print('deleted next similar waypoint', end='\n ')
                assert i < len(ind) - 1, 'last waypoint cannot be deleted'
                del ind[i]

    def change_speeds(self, ind):
        size = len(ind)
        k = random.randrange(1, size)
        i = random.randrange(size-k)
        assert i+k < size
        avg_curr_speed = sum([entry[1] for entry in ind[i:i+k]]) / k
        new_speed = random.choice([s for s in self.vessel.speeds
                                   if s != avg_curr_speed])
        for e in ind[i:i+k]:
            e[1] = new_speed

    def cx_one_point(self, ind1, ind2):
        if min(len(ind1), len(ind2)) > 2:
            # Get waypoints of ind1 and ind2
            ps1 = [row[0] for row in ind1]
            ps2 = [row[0] for row in ind2]
            p2_arr = np.asarray(ps2)

            # Shuffled list of indexes of ind1, excluding start and end
            c1s = random.sample(range(1, len(ind1)-1), k=len(ind1)-2)
            tr = 0  # Trial count
            while c1s:
                tr += 1
                c1 = c1s.pop()
                # Calculate distance between c1 and each point of ind2
                p1c_arr = np.asarray(ps1[c1]).reshape(1, -1)
                distances = distance.cdist(p1c_arr, p2_arr,
                                           metric=self.gc.distance)

                # Indices of ind2 that would sort distances
                c2s = distances.argsort()

                # Remove start and end indices
                c2s = c2s[(c2s != 0) & (c2s != len(ind2)-1)]

                # Try crossover with c1 and three nearest c2s
                for i in range(min(3, len(ind2)-2)):
                    c2 = c2s[i]

                    # Discard c2 if consecutive duplicate waypoints exist in children
                    if ps2[c2-1] == ps1[c1] or ps1[c1-1] == ps2[c2]:
                        continue

                    # Check feasibility
                    new_ind1_feasible = self.tb.e_feasible(ps1[c1-1], ps2[c2])
                    new_ind2_feasible = self.tb.e_feasible(ps2[c2-1], ps1[c1])
                    if new_ind1_feasible and new_ind2_feasible:
                        ind1[c1:], ind2[c2:] = ind2[c2:], ind1[c1:]
                    else:  # If new edges not feasible, discard c2
                        continue
                    assert len(ind1) > 2 and len(ind2) > 2, 'crossover children length < 3'
                    del ind1.fitness.values
                    del ind2.fitness.values
                    return ind1, ind2
        else:
            print('skipped crossover', end='\n ')
        return ind1, ind2

    def cx_one_point_old(self, ind1, ind2):
        size = min(len(ind1), len(ind2))
        if size > 2:
            trials1 = 0
            while trials1 < 100:
                cx_pt1, cx_pt2 = random.randrange(1, size-1), random.randrange(1, size-1)

                # Draw again if consecutive duplicate waypoints exist in children
                trials2 = 0
                while ind2[cx_pt2-1][0] == ind1[cx_pt1][0] or ind1[cx_pt1-1][0] == ind2[cx_pt2][0]:
                    cx_pt1, cx_pt2 = random.randrange(1, size-1), random.randrange(1, size-1)
                    trials2 += 1
                    if trials2 > 100:
                        print('exceeded crossover trials2', end='\n ')
                        return ind1, ind2

                # Check feasibility
                new_ind1_feasible = self.tb.e_feasible(ind1[cx_pt1-1][0], ind2[cx_pt2][0])
                new_ind2_feasible = self.tb.e_feasible(ind2[cx_pt2-1][0], ind1[cx_pt1][0])
                if new_ind1_feasible and new_ind2_feasible:
                    ind1[cx_pt1:], ind2[cx_pt2:] = ind2[cx_pt2:], ind1[cx_pt1:]
                else:
                    trials1 += 1
                    continue
                assert len(ind1) > 2 and len(ind2) > 2, 'crossover children length < 3'
                return ind1, ind2

            print('exceeded crossover trials1', end='\n ')
        else:
            print('skipped crossover', end='\n ')
        return ind1, ind2
