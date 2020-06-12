import random
from math import sqrt
from solution import Edge, Route
from haversine import haversine


def crossover(route_a, route_b, rtree_idx, polygons, max_distance):
    edges_a = [[i, edge] for i, edge in enumerate(route_a.edges) if i != 0]
    random.shuffle(edges_a)
    while edges_a:
        edge_a_idx, edge_a = edges_a.pop()

        edges_b = [[i, edge] for i, edge in enumerate(route_b.edges) if i != len(route_b.edges) - 1]
        random.shuffle(edges_b)
        selected_edge = False
        while edges_b:
            edge_b_idx, edge_b = edges_b.pop()
            if haversine((edge_a.v.lon, edge_a.v.lat), (edge_b.w.lon, edge_b.w.lat)) < max_distance:
                new_edge = Edge(edge_a.v, edge_b.w, edge_a.speed)
                if not new_edge.x_geometry(rtree_idx, polygons):
                    selected_edge = True
                    break

        if not selected_edge:
            continue
        else:
            edges = route_a.edges[0:edge_a_idx] + [new_edge] + route_b.edges[edge_b_idx + 1:]

            new_route = Route(edges)
            new_route.history.append('Crossover waypoints {0}, {1}'.format(edge_a_idx - 1, edge_a_idx))

            # # Check if edges are linked
            # for e, edge in enumerate(new_route.edges[:-1]):
            #     if edge.w != new_route.edges[e + 1].v:
            #         print('Error: not linking edges')

            return new_route
    return False
