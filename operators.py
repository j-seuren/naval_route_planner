import random
from math import cos, sin, pi, sqrt
from shapely.geometry import Point, Polygon
from solution import Waypoint, Edge, Route
from haversine import haversine
from copy import deepcopy


def crossover(route_a, route_b, polygons, max_distance=100):
    iteration_limit = 100
    count_limit = 200
    iteration = 0
    while iteration < iteration_limit:
        iteration += 1
        # Pick random edge and adjacent waypoints
        edge_a_idx = random.randint(1, len(route_a.edges) - 2)
        edge_b_idx = random.randint(1, len(route_b.edges) - 2)
        edge_a = route_a.edges[edge_a_idx]
        edge_b = route_b.edges[edge_b_idx]

        count = 0
        while (edge_a.v == edge_b.w or edge_b.v == edge_a.w) or count > count_limit:
            count += 1
            edge_a_idx = random.randint(1, len(route_a.edges) - 2)
            edge_b_idx = random.randint(1, len(route_b.edges) - 2)
            edge_a = route_a.edges[edge_a_idx]
            edge_b = route_b.edges[edge_b_idx]

        x_v_a, y_v_a = edge_a.v.x, edge_a.v.y
        x_w_a, y_w_a = edge_a.w.x, edge_a.w.y
        x_v_b, y_v_b = edge_b.v.x, edge_b.v.y
        x_w_b, y_w_b = edge_b.w.x, edge_b.w.y

        distance_va_wb = 0.539957 * haversine((y_v_a, x_v_a), (y_w_b, x_w_b))  # Nautical miles
        distance_vb_wa = 0.539957 * haversine((y_v_b, x_v_b), (y_w_a, x_w_a))  # Nautical miles

        if distance_va_wb == 0 or distance_vb_wa == 0:
            print('DISTANCE  DISTANCE  DISTANCE  DISTANCE  DISTANCE')

        if distance_va_wb < max_distance:  # Connect vertex v of route A with vertex w of route B
            new_edge = Edge(edge_a.v, edge_b.w, edge_a.speed)
            if new_edge.crosses_polygon(polygons):
                continue
            edges = route_a.edges[0:edge_a_idx]
            edges.append(new_edge)
            edges.extend(route_b.edges[edge_b_idx+1:])

            # Remove potential loops
            loops_exist = True
            while loops_exist:
                loops_exist = False
                for e1, edge1 in enumerate(edges):
                    if len(edges) > e1 + 1:
                        for e2, edge2 in enumerate(edges[e1 + 2:]):
                            if edge1.w == edge2.v:
                                edges = edges[:e1 + 1] + edges[e1 + e2 + 2:]
                                loops_exist = True
                                break
                        if loops_exist:
                            break
            new_route = Route(edges)
            new_route.history.extend(route_a.history)
            new_route.history.extend('-')
            new_route.history.extend(route_b.history)
            new_route.history.append('Crossover waypoints {0}, {1}'.format(edge_a_idx-1, edge_a_idx))

            # Check edges
            for e, edge in enumerate(new_route.edges[:-1]):
                if edge.w != new_route.edges[e+1].v:
                    print('Error: not linking edges')
            return new_route

        elif distance_vb_wa < max_distance:  # Connect vertex v of route B with vertex w of route A
            new_edge = Edge(edge_b.v, edge_a.w, edge_b.speed)
            if new_edge.crosses_polygon(polygons):
                continue
            edges = route_b.edges[0:edge_b_idx]
            edges.append(new_edge)
            edges.extend(route_a.edges[edge_a_idx+1:])

            # Remove potential loops
            loops_exist = True
            old_edges = deepcopy(edges)
            while loops_exist:
                loops_exist = False
                for e1, edge1 in enumerate(edges):
                    if len(edges) > e1 + 1:
                        for e2, edge2 in enumerate(edges[e1 + 2:]):
                            if edge1.w == edge2.v:
                                edges = edges[:e1 + 1] + edges[e1 + e2 + 2:]
                                loops_exist = True
                                break
                        if loops_exist:
                            break
            new_route = Route(edges)
            new_route.history.append('Crossover waypoints {0}, {1}'.format(edge_b_idx-1, edge_b_idx))

            # Check edges
            for e, edge in enumerate(new_route.edges[:-1]):
                if edge.w != new_route.edges[e+1].v:
                    print('Error: not linking edges')
            return new_route
        else:
            continue
    return False


def get_one_crossover_from_routes(routes, polygons):
    random.shuffle(routes)
    for route_a in routes:
        for route_b in routes:
            if route_a is not route_b:
                crossover_route = crossover(route_a, route_b, polygons)
                if crossover_route:
                    return crossover_route
    return False
