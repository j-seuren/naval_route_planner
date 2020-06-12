import random
from classes import Edge, Route
from haversine import haversine


def crossover(route_a, route_b, rtree_idx, polygons, max_distance):
    edges_a = [[i, edge] for i, edge in enumerate(route_a.edges) if i != 0]
    random.shuffle(edges_a)
    edges_b = [[i, edge] for i, edge in enumerate(route_b.edges) if i != len(route_b.edges) - 1]
    random.shuffle(edges_b)
    while edges_a:
        edge_a_idx, edge_a = edges_a.pop()
        shuffled_b = edges_b[:]
        while shuffled_b:
            edge_b_idx, edge_b = shuffled_b.pop()
            if haversine((edge_a.v.lon, edge_a.v.lat), (edge_b.w.lon, edge_b.w.lat)) < max_distance:
                new_edge = Edge(edge_a.v, edge_b.w, edge_a.speed)
                if not new_edge.x_geometry(rtree_idx, polygons):
                    edges = route_a.edges[0:edge_a_idx] + [new_edge] + route_b.edges[edge_b_idx + 1:]
                    new_route = Route(edges)
                    return new_route
    print('No crossover performed')
    return False
