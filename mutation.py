from itertools import groupby
from operator import itemgetter
import functools
import random
import numpy as np
from math import sqrt, pi, cos, sin


def insert_waypoint(toolbox, width_ratio, individual, initializing):
    e = random.randint(0, len(individual) - 2)
    u_tup, v_tup = individual[e][0], individual[e+1][0]
    u, v = np.asarray(u_tup), np.asarray(v_tup)

    # Edge center
    center = (u + v) / 2
    dy = v[1] - u[1]
    dx = v[0] - u[0]

    try:
        slope = dy / dx
    except ZeroDivisionError:
        slope = 'inf'

    # Get half length of the edge's perpendicular bisector
    half_width = 1 / 2 * width_ratio * sqrt(dy ** 2 + dx ** 2)

    # Calculate outer points of polygon
    if slope == 'inf':
        y_ab = np.ones(2) * center[1]
        x_ab = np.array([1, -1]) * half_width + center[0]
    elif slope == 0:
        x_ab = np.array([center[0], center[0]])
        y_ab = np.array([1, -1]) * half_width + center[1]
    else:
        square_root = sqrt(half_width ** 2 / (1 + 1 / slope ** 2))
        x_ab = np.array([1, -1]) * square_root + center[0]
        y_ab = -(1 / slope) * (x_ab - center[0]) + center[1]

    a, b = np.array([x_ab[0], y_ab[0]]), np.array([x_ab[1], y_ab[1]])
    v1, v2 = a - u, b - u
    while True:
        new_wp = u + random.random() * v1 + random.random() * v2  # Random point in quadrilateral
        new_wp_tup = (new_wp[0], new_wp[1])

        if initializing and (not toolbox.edge_feasible(u_tup, new_wp_tup) or
                             not toolbox.edge_feasible(new_wp_tup, v_tup)):
            continue
        # Insert waypoint
        individual.insert(e+1, [new_wp_tup, individual[e][1]])
        return


def move_waypoints(toolbox, radius, individual, initializing):
    if len(individual) < 3:
        print('move')
        return
    n_waypoints = random.randint(1, len(individual) - 2)
    if initializing:
        n_waypoints = 1
    first = random.randrange(1, len(individual) - n_waypoints)
    for wp_idx in range(first, first + n_waypoints):
        wp = individual[wp_idx][0]
        wp_arr = np.asarray(wp)
        while True:
            # Pick a random location within a radius from the current waypoint
            u1, u2 = random.random(), random.random()
            r = radius * sqrt(u2)  # Square root for a uniform probability of choosing a point in the circle
            a = u1 * 2 * pi
            c_s = np.array([cos(a), sin(a)])
            new_wp_arr = wp_arr + r * c_s
            new_wp = (new_wp_arr[0], new_wp_arr[1])

            # Check if waypoint and edges do not intersect a polygon
            if initializing and (not toolbox.edge_feasible(individual[wp_idx - 1][0], new_wp) or
                                 not toolbox.edge_feasible(new_wp, individual[wp_idx+1][0])):
                continue

            individual[wp_idx][0] = new_wp
            break
    return


def star(f):
    @functools.wraps(f)
    def f_inner(args):
        return f(*args)
    return f_inner


def delete_random_waypoints(toolbox, individual):
    if len(individual) < 3:
        print('del')
        return
    to_be_deleted = sorted(random.sample(range(1, len(individual)-1), k=random.randint(1, len(individual)-2)))
    while to_be_deleted:
        # Group consecutive waypoints
        tbd_copy = to_be_deleted[:]
        for k, g in groupby(enumerate(tbd_copy), key=star(lambda u, x: u - x)):
            consecutive_nodes = list(map(itemgetter(1), g))
            first = consecutive_nodes[0]
            last = consecutive_nodes[-1]

            # If to be created edge is not feasible, remove to be deleted nodes from list
            if not toolbox.edge_feasible(individual[first-1][0], individual[last+1][0]):
                del to_be_deleted[to_be_deleted.index(first):to_be_deleted.index(last) + 1]

        # Delete waypoints
        for i, element in enumerate(to_be_deleted):
            del individual[element - i]
        return
    return


def change_speed(vessel, individual):
    n_edges = random.randint(1, len(individual) - 1)
    first = random.randrange(len(individual) - n_edges)
    new_speed = random.choice([speed for speed in vessel.speeds if speed])
    for item in individual[first:first + n_edges]:
        item[1] = new_speed
    return


def mutate(toolbox, mutation_ops, individual, initializing=False):
    swap = random.choice(mutation_ops)
    if swap == 'insert':
        toolbox.insert(individual, initializing)
    elif swap == 'move':
        toolbox.move(individual, initializing)
    elif swap == 'delete':
        toolbox.delete(individual)
    elif swap == 'speed':
        toolbox.speed(individual)
