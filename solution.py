from haversine import haversine
from shapely.geometry import LineString

distances_cache = dict()


def evaluate(vessel, individual):  # Add memory
    global distances_cache

    # Initialize variables
    travel_time, fuel_consumption = 0, 0

    # Compute objective values for each edge
    for i in range(len(individual) - 1):
        u = individual[i][0]
        v = individual[i+1][0]
        speed = individual[i][1]

        if u[0] < v[0]:
            key = (u, v)
        else:
            key = (v, u)

        # If distance in cache, get distance. Otherwise, calculate distance and save to cache
        if key in distances_cache:
            edge_distance = distances_cache[key]
        else:
            edge_distance = haversine(u, v, unit='nmi')
            distances_cache[key] = edge_distance

        edge_travel_time = edge_distance / speed  # Hours
        fuel_rate = vessel.fuel_rates[speed]  # Tons / Hour
        edge_fuel_consumption = fuel_rate * edge_travel_time  # Tons

        # Increment objective values
        travel_time += edge_travel_time
        fuel_consumption += edge_fuel_consumption

    return travel_time, fuel_consumption


def edge_feasible(rtree_idx, prep_geoms, max_distance, u, v):
    global distances_cache

    if u[0] < v[0]:
        key = (u, v)
    else:
        key = (v, u)

    # If distance in cache, get distance. Otherwise, calculate distance and save to cache
    if key in distances_cache:
        distance = distances_cache[key]
    else:
        distance = haversine(u, v, unit='nmi')
        distances_cache[key] = distance

    # If distance is larger than maximum edge length, return infeasible
    if distance > max_distance:
        return False

    # Compute line bounds
    u_x, u_y = u
    v_x, v_y = v
    line_bounds = (min(u_x, v_x), min(u_y, v_y),
                   max(u_x, v_x), max(u_y, v_y))

    # Returns the geometry indices of the minimum bounding rectangles of polygons that intersect the edge bounds
    mbr_intersections = rtree_idx.intersection(line_bounds)
    if mbr_intersections:
        # Create LineString if there is at least one minimum bounding rectangle intersection
        shapely_line = LineString([u, v])

        # For every mbr intersection check if its polygon is actually intersect by the edge
        for i in mbr_intersections:
            if prep_geoms[i].intersects(shapely_line):
                return False
    return True
