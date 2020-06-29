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
                 radius=1,
                 width_ratio=0.5,
                 mutation_ops=None
                 ):
        self.toolbox = toolbox
        self.vessel = vessel
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
    
    def insert_waypoint(self, ind, initializing):
        if initializing:
            nr_inserts = 1
        else:
            nr_inserts = random.randint(1, len(ind)-2)
        for i in range(nr_inserts):
            e = random.randint(0, len(ind) - 2)
            p1_tup, p2_tup = ind[e][0], ind[e + 1][0]
            p1, p2 = np.asarray(p1_tup), np.asarray(p2_tup)

            # Edge center
            ctr = (p1 + p2) / 2
            dy, dx = p2[1] - p1[1], p2[0] - p1[0]

            try:
                slope = dy / (dx + 0.0)
            except ZeroDivisionError:
                slope = 'inf'

            # Get half length of the edge's perpendicular bisector
            half_width = 0.5 * self.width_ratio * sqrt(dy ** 2 + dx ** 2)

            # Calculate outer points of polygon
            if slope == 'inf':
                y_ab = np.ones(2) * ctr[1]
                x_ab = np.array([1, -1]) * half_width + ctr[0]
            elif slope == 0:
                x_ab = np.array([ctr[0], ctr[0]])
                y_ab = np.array([1, -1]) * half_width + ctr[1]
            else:
                square_root = sqrt(half_width ** 2 / (1 + 1 / slope ** 2))
                x_ab = np.array([1, -1]) * square_root + ctr[0]
                y_ab = -(1 / slope) * (x_ab - ctr[0]) + ctr[1]

            a, b = np.array([x_ab[0], y_ab[0]]), np.array([x_ab[1], y_ab[1]])
            v1, v2 = a - p1, b - p1
            trials = 0
            while True:
                new_wp = tuple(p1 + random.random() * v1 + random.random() * v2)  # Random point in quadrilateral

                if initializing and (not self.toolbox.edge_feasible(p1_tup, new_wp) or
                                     not self.toolbox.edge_feasible(new_wp, p2_tup)):
                    trials += 1
                    if trials > 100:
                        print('No waypoint inserted in edge')
                        break
                    else:
                        continue
                # Insert waypoint
                ind.insert(e+1, [new_wp, ind[e][1]])
                break
        return
    
    def move_waypoints(self, ind, initializing):
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
                u1, u2 = random.random(), random.random()
                r = self.radius * sqrt(u2)  # Square root for a uniform probability of choosing a point in the circle
                a = u1 * 2 * pi
                c_s = np.array([cos(a), sin(a)])
                new_wp = tuple(wp + r * c_s)
    
                # Check if waypoint and edges do not intersect a polygon
                if initializing and (not self.toolbox.edge_feasible(ind[wp_idx-1][0], new_wp) or
                                     not self.toolbox.edge_feasible(new_wp, ind[wp_idx+1][0])):
                    trials += 1
                    if trials > 100:
                        # print('No waypoint moved')
                        # x = np.array([ind[wp_idx-1][0][0], new_wp[0], ind[wp_idx+1][0][0]])
                        # y = np.array([ind[wp_idx-1][0][1], new_wp[1], ind[wp_idx+1][0][1]])
                        #
                        # margin = 5
                        # left, right = max(math.floor(min(x)) - margin, -180), min(math.ceil(max(x)) + margin, 180)
                        # bottom, top = max(math.floor(min(y)) - margin, -90), min(math.ceil(max(y)) + margin, 90)
                        #
                        # m = Basemap(projection='merc', resolution='c', llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left,
                        #             urcrnrlon=right)
                        # m.drawparallels(np.arange(-90., 90., 10.), labels=[1, 0, 0, 0], fontsize=10)
                        # m.drawmeridians(np.arange(-180., 180., 10.), labels=[0, 0, 0, 1], fontsize=10)
                        # m.drawcoastlines()
                        # m.fillcontinents()
                        #
                        # # Plot edges
                        # waypoints = list(zip(x, y))
                        # edges = zip(waypoints[:-1], waypoints[1:])
                        # for i, e in enumerate(edges):
                        #     m.drawgreatcircle(e[0][0], e[0][1], e[1][0], e[1][1], linewidth=2, color='black', zorder=1)
                        # for i, (x, y) in enumerate(waypoints):
                        #     m.scatter(x, y, latlon=True, color='dimgray', marker='o', s=5, zorder=2)
                        #
                        # plt.show()
                        break
                    else:
                        continue
    
                ind[wp_idx][0] = new_wp
                break
        return    
    
    def delete_random_waypoints(self, ind, initializing):
        # List of "to be deleted" waypoints
        tbd = sorted(random.sample(range(1, len(ind)-1), k=random.randint(1, len(ind)-2)))
        if initializing:
            while tbd:
                # Group consecutive waypoints
                tbd_copy = tbd[:]
                for k, g in itertools.groupby(enumerate(tbd_copy), key=star(lambda u, x: u - x)):
                    consecutive_nodes = list(map(itemgetter(1), g))
                    first = consecutive_nodes[0]
                    last = consecutive_nodes[-1]

                    # If to be created edge is not feasible, remove to be deleted nodes from list
                    if not self.toolbox.edge_feasible(ind[first - 1][0], ind[last + 1][0]):
                        del tbd[tbd.index(first):tbd.index(last) + 1]

                # Delete waypoints
                for i, el in enumerate(tbd):
                    del ind[el - i]
                return
            return
        else:
            for i, el in enumerate(tbd):
                del ind[el - i]
            return
    
    def change_speed(self, ind):
        n_edges = random.randint(1, len(ind) - 1)
        first = random.randrange(len(ind) - n_edges)
        new_speed = random.choice([speed for speed in self.vessel.speeds if speed])
        for item in ind[first:first + n_edges]:
            item[1] = new_speed
        return

    def crossover(self, ind1, ind2):
        p1s = [[i, el[0]] for i, el in enumerate(ind1)][1:-2]
        random.shuffle(p1s)
        p2s = [[i, el[0]] for i, el in enumerate(ind2)][2:-1]
        random.shuffle(p2s)
        while p1s:
            p1_idx, p1 = p1s.pop()
            p2s_copy = p2s[:]
            while p2s_copy:
                p2_idx, p2 = p2s_copy.pop()
                if self.toolbox.edge_feasible(p1, p2):
                    child = ind1[:p1_idx + 1] + ind2[p2_idx:]
                    return child
        return False
