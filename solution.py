from haversine import haversine
from shapely.geometry import LineString


distances_cache = dict()


def evaluate(vessel, individual, return_longest_edge=False):  # Add memory
    global distances_cache
    distance = 0
    fuel_consumption = 0
    travel_time = 0
    longest_edge = 0
    longest_edge_distance = 0
    for i in range(len(individual) - 1):
        u = individual[i, 0:2]
        v = individual[i+1, 0:2]
        if u[0] < v[0]:
            key = (u[0], u[1], v[0], v[1])
        else:
            key = (v[0], v[1], u[0], u[1])
        if key in distances_cache:
            edge_distance = distances_cache[key]
        else:
            edge_distance = haversine((v[0], v[1]), (u[0], u[1]), unit='nmi')
            distances_cache[key] = edge_distance
        distance += edge_distance
        if not return_longest_edge:
            edge_travel_time = edge_distance / individual[i, 2]  # Hours

            fuel_rate = vessel.fuel_rates[individual[i, 2]]
            edge_fuel_consumption = fuel_rate * edge_travel_time

            travel_time += edge_travel_time
            fuel_consumption += edge_fuel_consumption
        elif edge_distance > longest_edge_distance:
            longest_edge = i
            longest_edge_distance = edge_distance

    if return_longest_edge:
        return longest_edge, longest_edge_distance
    else:
        return travel_time, fuel_consumption


def edge_feasible(rtree_idx, prep_geoms, max_distance, u, v):
    global distances_cache

    if u[0] < v[0]:
        key = (u[0], u[1], v[0], v[1])
    else:
        key = (v[0], v[1], u[0], u[1])
    if key in distances_cache:
        distance = distances_cache[key]
    else:
        distance = haversine((v[0], v[1]), (u[0], u[1]), unit='nmi')
        distances_cache[key] = distance
    if distance > max_distance:
        return False

    u_x, u_y = float(u[0]), float(u[1])
    v_x, v_y = float(v[0]), float(v[1])
    line_bounds = (min(u_x, v_x), min(u_y, v_y),
                   max(u_x, v_x), max(u_y, v_y))

    # Returns the geometry indices of the minimum bounding rectangles that intersect the bounding box of edge
    mbr_intersections = rtree_idx.intersection(line_bounds)
    if mbr_intersections:  # Create LineString if there is at least one minimum bounding rectangle intersection
        shapely_line = LineString([(u[0], u[1]), (v[0], v[1])])
        for i in mbr_intersections:
            if prep_geoms[i].intersects(shapely_line):
                return False
    return True
