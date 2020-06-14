import random
import numpy as np
from math import sqrt, pi, cos, sin
from haversine import haversine


def insert_waypoint(toolbox, individual, width_ratio=0.5, e=False):
    if not e:
        e = random.randint(0, len(individual) - 2)
    u = np.transpose(individual[e, 0:2])
    v = np.transpose(individual[e+1, 0:2])

    # Edge center
    center = np.transpose(np.array([(u + v) / 2]))
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

        if not toolbox.edge_feasible(u, new_wp) or not toolbox.edge_feasible(new_wp, v):
            continue

        # Insert waypoint
        np.insert(individual, e+1, np.append(new_wp, individual[e, 2]), 0)
        return individual


def delete_random_waypoint(toolbox, max_distance, individual):
    waypoints = [[i, wp] for i, wp in enumerate(individual[:, 0:2])][1:-1]
    random.shuffle(waypoints)
    while waypoints:
        # Pop waypoint from shuffled list and get its index
        wp_idx, wp = waypoints.pop()

        # Create new edge
        pre, nex = individual[wp_idx-1, 0:2], individual[wp_idx+1, 0:2]

        # Check if edge is greater than max distance or intersects a polygon
        if haversine((pre[0], pre[1]), (nex[0], nex[1])) > max_distance or not toolbox.edge_feasible(pre, nex):
            continue

        np.delete(individual, wp_idx, axis=0)
        return individual
    print("No waypoint deleted")


def change_speed(vessel, individual):
    n_edges = random.randint(1, len(individual) - 1)
    first = random.randrange(len(individual) - n_edges)
    new_speed = random.choice([speed for speed in vessel.speeds])
    individual[first:first + n_edges, 2] = float(new_speed)
    return individual


def move_waypoint(toolbox, individual, wp=False, radius=0.01):
    if not wp:
        row_i = random.randint(1, len(individual)-2)
        wp = individual[row_i, 0:2]
    assert np.all(wp != individual[0, 0:2], axis=0) and np.all(wp != individual[-1, 0:2], axis=0), \
        'First or last waypoint cannot be moved'
    wp_idx = int(np.where(np.all(wp == individual[:, 0:2], axis=1))[0])
    while True:
        # Pick a random location within a radius from the current waypoint
        u1, u2 = random.random(), random.random()
        r = radius * sqrt(u2)  # Square root for a uniform probability of choosing a point in the circle
        a = u1 * 2 * pi
        c_s = np.array([cos(a), sin(a)])
        new_wp = wp + r * c_s

        # Check if waypoint and edges do not intersect a polygon
        if not toolbox.edge_feasible(individual[wp_idx-1, 0:2], new_wp) \
                or not toolbox.edge_feasible(new_wp, individual[wp_idx+1, 0:2]):
            continue
        individual[wp_idx, 0:2] = new_wp
        return individual


def mutate(toolbox, swaps, individual):
    mutations = random.randint(1, 20)

    for i in range(mutations):
        swap = random.choice(swaps)
        if swap == 'insert':
            individual = toolbox.insert(individual)
        elif swap == 'move':
            # wp = random.choice(self.waypoints[1:-1])
            individual = toolbox.move(individual)
        elif swap == 'delete':
            individual = toolbox.move(individual)
        elif swap == 'speed':
            individual = toolbox.speed(individual)

    return individual
