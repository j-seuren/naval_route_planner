from haversine import haversine
from shapely.geometry import LineString


def evaluate(vessel, individual, return_longest_edge=False):  # Add memory
    distance = 0
    fuel_consumption = 0
    travel_time = 0
    longest_edge = 0
    longest_edge_distance = 0
    for i in range(len(individual) - 1):
        edge_distance = haversine(individual[i, 0:2], individual[i+1, 0:2], unit='nmi')
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


def edge_x_geometry(rtree_idx, geometries, u, v):
    u_x, u_y = float(u[0]), float(u[1])
    v_x, v_y = float(v[0]), float(v[1])
    line_bounds = (min(u_x, v_x), min(u_y, v_y),
                   max(u_x, v_x), max(u_y, v_y))

    # Returns the geometry indices of the minimum bounding rectangles that intersect the bounding box of edge
    mbr_intersections = rtree_idx.intersection(line_bounds)
    if mbr_intersections:  # Create LineString if there is at least one minimum bounding rectangle intersection
        shapely_line = LineString([(u[0], u[1]), (v[0], v[1])])
        for i in mbr_intersections:
            if geometries[i].intersects(shapely_line):
                return False
    return False
