import functools
import itertools
import random
import numpy as np

from operator import itemgetter
from math import sqrt, pi, cos, sin


def star(f):
    @functools.wraps(f)
    def f_inner(args):
        return f(*args)
    return f_inner


class Operators:
    def __init__(self,
                 toolbox,
                 vessel,
                 geod,
                 radius=5,
                 width_ratio=1,
                 mutation_ops=None
                 ):
        self.toolbox = toolbox
        self.vessel = vessel
        self.geod = geod
        self.radius = radius                                                # Parameter for move_random_waypoint
        self.width_ratio = width_ratio                                      # Parameter for insert_random_waypoint
        if mutation_ops is None:
            self.mutation_ops = ['insert', 'move', 'delete', 'speed']
        else:
            self.mutation_ops = mutation_ops

    def mutate(self, ind, initializing=False):
        swap = random.choice(self.mutation_ops)
        if swap == 'move' and len(ind) > 2:
            self.move_waypoints(ind, initializing)
        elif swap == 'delete' and len(ind) > 2:
            self.delete_random_waypoints(ind, initializing)
        elif swap == 'speed':
            self.change_speed(ind)
        else:
            self.insert_waypoint(ind, initializing)
        assert no_similar_points(ind), 'similar points in individual after crossover'
        return ind,

    def insert_waypoint(self, ind, initializing, shape='rhombus'):
        if initializing:
            nr_inserts = 1
        else:
            nr_inserts = random.randint(1, len(ind) - 1)
        for i in range(nr_inserts):
            # Pick edge for insertion
            e = random.randint(0, len(ind) - 2)
            p1_tup, p2_tup = ind[e][0], ind[e+1][0]
            p1, p2 = np.asarray(p1_tup), np.asarray(p2_tup)
            ctr = (p1 + p2) / 2  # Edge centre
            width = np.linalg.norm(p1 - p2)  # Ellipse width
            height = width * 0.5  # Ellipse height

            trials = 0
            while True:
                if shape == 'ellipse':
                    # Generate a random point inside a circle of radius 1
                    rho = random.random()
                    phi = random.random() * 2 * pi
                    unit_circle = sqrt(rho) * np.array([cos(phi), sin(phi)])

                    # Scale x and y to the dimensions of the ellipse
                    pt_origin = unit_circle * np.array([width, height]) / 2.0
                elif shape == 'rhombus':
                    v1, v2 = np.array([width / 2.0, - height / 2.0]), np.array([width / 2.0, height / 2.0])
                    pt_origin = random.random() * v1 + random.random() * v2 - np.array([width / 2.0, 0.0])
                else:
                    raise ValueError("Shape argument invalid: try 'ellipse' or 'rhombus'")

                # Transform
                dy, dx = p2[1] - p1[1], p2[0] - p1[0]
                if dx != 0:
                    slope = dy / dx
                    rot_x = 1 / sqrt(1 + slope ** 2)
                    rot_y = slope * rot_x
                else:
                    assert dy != 0, 'p1 and p2 are the same'
                    rot_x = 0
                    rot_y = 1

                cos_a, sin_a = rot_x, rot_y
                rotation_matrix = np.array(((cos_a, -sin_a), (sin_a, cos_a)))
                pt_rotated = np.dot(rotation_matrix, pt_origin)  # Rotate
                pt = ctr + pt_rotated  # Translate
                new_wp = tuple(np.clip(pt, [-180, -90], [180, 90]))

                if initializing and (not self.toolbox.edge_feasible(p1_tup, new_wp) or
                                     not self.toolbox.edge_feasible(new_wp, p2_tup)):
                    trials += 1
                    if trials > 100:
                        print('no waypoint inserted in edge')
                        break
                    else:
                        continue
                # Insert waypoint
                ind.insert(e + 1, [new_wp, ind[e][1]])
                break
        return

    def move_waypoints(self, ind, initializing):
        if len(ind) == 2:
            print('skipped move')
            return
        if initializing:
            n_wps = 1
        else:
            n_wps = random.randint(1, len(ind) - 2)
        first = random.randrange(1, len(ind) - n_wps)
        for wp_idx in range(first, first + n_wps):
            wp = np.asarray(ind[wp_idx][0])
            trials = 0
            while True:
                # Pick a random location within a radius from the current waypoint
                r = self.radius * sqrt(random.random())  # Square root ensures a uniform distribution inside the circle.
                phi = random.random() * 2 * pi
                new_wp_arr = wp + r * np.array([cos(phi), sin(phi)])
                new_wp = tuple(np.clip(new_wp_arr, [-180, -90], [180, 90]))

                if new_wp == ind[wp_idx-1][0] or new_wp == ind[wp_idx+1][0]:
                    print('no wp move')
                    break

                # Ensure feasibility during initialization
                if initializing and (not self.toolbox.edge_feasible(ind[wp_idx-1][0], new_wp) or
                                     not self.toolbox.edge_feasible(new_wp, ind[wp_idx+1][0])):
                    trials += 1
                    if trials > 100:
                        print('exceeded move trials')
                        break
                    else:
                        continue
                ind[wp_idx][0] = new_wp
                break
        return    
    
    def delete_random_waypoints(self, ind, initializing):
        if len(ind) == 2:
            print('skipped delete')
            return
        # Get random sample of to be deleted waypoints indices of random length of max size(ind) - 3
        tbd_len = random.randint(1, len(ind) - 2)
        tbd = sorted(random.sample(range(1, len(ind)-1), k=tbd_len))
        if initializing:  # Ensure feasibility
            # Group consecutive waypoints
            tbd_copy = tbd[:]
            for k, g in itertools.groupby(enumerate(tbd_copy), key=star(lambda u, x: u - x)):
                consecutive_nodes = list(map(itemgetter(1), g))
                first = consecutive_nodes[0]
                last = consecutive_nodes[-1]

                # If to be created edge is not feasible, remove to be deleted nodes from list
                if not self.toolbox.edge_feasible(ind[first-1][0], ind[last+1][0]):
                    del tbd[tbd.index(first):tbd.index(last) + 1]
        for pt_idx in sorted(tbd, reverse=True):
            del ind[pt_idx]
            while ind[pt_idx - 1][0] == ind[pt_idx][0]:
                print('deleted next similar waypoint')
                assert pt_idx < len(ind) - 1, 'last waypoint cannot be deleted'
                del ind[pt_idx]
        return
    
    def change_speed(self, ind):
        if len(ind) == 2:
            print('skipped change speed')
            return
        n_edges = random.randint(1, len(ind) - 1)
        first = random.randrange(len(ind) - n_edges)
        new_speed = random.choice([speed for speed in self.vessel.speeds if speed])
        for item in ind[first:first + n_edges]:
            item[1] = new_speed
        return

    def crossover(self, ind1, ind2):
        size = min(len(ind1), len(ind2))
        if size == 2:
            print('skipped crossover')
            return ind1, ind2
        cx_pt1, cx_pt2 = random.randint(1, size - 1), random.randint(1, size - 1)
        trials = 0
        while ind2[cx_pt2-1][0] == ind1[cx_pt1][0] or ind1[cx_pt1-1][0] == ind2[cx_pt2][0]:
            cx_pt1, cx_pt2 = random.randint(1, size - 1), random.randint(1, size - 1)
            trials += 1
            if trials > 100:
                print('exceeded crossover trials')
                return ind1, ind2

        # Check feasibility
        if self.toolbox.edge_feasible(ind1[cx_pt1-1][0], ind2[cx_pt2][0]) \
                and self.toolbox.edge_feasible(ind2[cx_pt2-1][0], ind1[cx_pt1][0]):
            ind1[cx_pt1:], ind2[cx_pt2:] = ind2[cx_pt2:], ind1[cx_pt1:]
        assert no_similar_points(ind1) and no_similar_points(ind2), 'similar points in individual after crossover'
        assert len(ind1) > 2 and len(ind2) > 2, 'crossover children length < 3'
        return ind1, ind2


def no_similar_points(ind):
    for i, row in enumerate(ind[:-1]):
        if row[0] == ind[i+1][0]:
            return False
    return True
