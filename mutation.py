import random
import numpy as np
from math import sqrt, pi, cos, sin


def insert_waypoint(toolbox, width_ratio, individual):
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

        if toolbox.edge_feasible(u_tup, new_wp_tup) and toolbox.edge_feasible(new_wp_tup, v_tup):
            # Insert waypoint
            individual.insert(e+1, [new_wp_tup, individual[e][1]])
            return


def move_waypoints(toolbox, radius, individual):
    wp_idx = random.randint(1, len(individual)-2)
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
        if toolbox.edge_feasible(individual[wp_idx-1][0], new_wp) \
                and toolbox.edge_feasible(new_wp, individual[wp_idx+1][0]):
            individual[wp_idx][0] = new_wp
            return


def delete_random_waypoint(toolbox, individual):
    waypoints = [[i, row[0]] for i, row in enumerate(individual)][1:-1]
    random.shuffle(waypoints)
    while waypoints:
        # Pop waypoint from shuffled list and get its index
        wp_idx, wp = waypoints.pop()

        # Create new edge
        pre, nex = individual[wp_idx-1][0], individual[wp_idx+1][0]

        # Check if edge is greater than max distance or intersects a polygon
        if toolbox.edge_feasible(pre, nex):
            del individual[wp_idx]
            return True
    return False


def change_speed(vessel, individual):
    n_edges = random.randint(2, len(individual) - 1)
    first = random.randrange(len(individual) - n_edges)
    new_speed = random.choice([speed for speed in vessel.speeds])
    for item in individual[first:first + n_edges]:
        item[1] = new_speed
    return


def mutate(toolbox, swaps, individual):
    for i in range(random.randint(1, len(individual))):
        swap = random.choice(swaps)
        if swap == 'insert':
            toolbox.insert(individual)
        elif swap == 'move':
            toolbox.move(individual)
        elif swap == 'delete':
            wp_deleted = toolbox.delete(individual)
            if not wp_deleted:
                toolbox.insert(individual)
        elif swap == 'speed':
            toolbox.speed(individual)
