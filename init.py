import build_hexagon_graph
import heapq
import itertools
import networkx as nx
import numpy as np

from haversine import haversine


def get_path(graph, canal_nodes, print_crossing=True):
    """
    Calculate shortest path from
    Returns:
         path: list of sub paths of a shortest path from start to end
         x_canal_pairs: list with tuples of crossed canal nodes in order of crossing
    """
    path = [nx.shortest_path(graph, 'start', 'end', weight='miles', method='dijkstra')]
    x_canal_nodes = [wp for wp in path[0] if wp in canal_nodes]

    if x_canal_nodes:
        x_canal_pairs = list(zip(x_canal_nodes[::2], x_canal_nodes[1::2]))
        reversed_pairs = x_canal_pairs[::-1]

        while reversed_pairs:
            x_canal_pair = reversed_pairs.pop()
            sub_path = path.pop()  # Get sub path in which canal node is found
            i = sub_path.index(x_canal_pair[0])  # Get index of canal_node in sub path
            path.extend([sub_path[:i+1], sub_path[i+1:]])  # Split sub path at canal, and extend to path
            if print_crossing:
                print('Crossing canal from {0} to {1}'.format(x_canal_pair[0], x_canal_pair[1]))
    else:
        x_canal_pairs = []
    return path, x_canal_pairs


def get_global_routes(container, res, d, vd, split_polygons, rtree_idx, start, end, vessel):
    graph_dir = 'output/variable_density_geodesic_grids/'
    paths = []

    # Load the graph file
    try:
        G = nx.read_gpickle(graph_dir + 'res_{}_d{}_vd{}.gpickle'.format(res, d, vd))
    except FileNotFoundError:
        G = build_hexagon_graph.get_graph(res, d, vd, split_polygons, rtree_idx)

    # Compute distances to start and end locations
    d_start = {n: haversine(start, lon_lat) for n, lon_lat in nx.get_node_attributes(G, 'lon_lat').items()}
    d_end = {n: haversine(end, lon_lat) for n, lon_lat in nx.get_node_attributes(G, 'lon_lat').items()}

    # Add start and end nodes
    G.add_node('start', lon_lat=start)
    G.add_node('end', lon_lat=end)

    # Add three shortest edges to start and end point
    for node in heapq.nsmallest(3, d_start, key=d_start.get):
        G.add_edge('start', node, miles=d_start[node])
    for node in heapq.nsmallest(3, d_end, key=d_end.get):
        G.add_edge('end', node, miles=d_end[node])

    canals = {'Panama': ['panama_south', 'panama_north'],
              'Suez': ['suez_south', 'suez_north'],
              'Dardanelles': ['dardanelles_south', 'dardanelles_north']}
    canal_nodes = [n for element in canals.values() for n in element]

    path, x_canals = get_path(G, canal_nodes)
    paths.append(path)
    if x_canals:
        route_combinations = itertools.product(*[(True, False)] * len(x_canals))
        for rc in route_combinations:
            excl_canals = [canal for i, canal in enumerate(x_canals) if rc[i]]
            G.remove_edges_from(excl_canals)
            try:
                alternative_path, _ = get_path(G, canal_nodes, print_crossing=False)
                G.add_edges_from(excl_canals)
            except nx.exception.NetworkXNoPath:
                G.add_edges_from(excl_canals)
                continue
            if alternative_path not in paths:
                paths.append(alternative_path)

    # Create individual of each sub path
    global_routes = []
    for path in paths:
        path_individuals = []
        for sub_path in path:
            waypoints = [(G.nodes[node]['lon_lat'][0], G.nodes[node]['lon_lat'][1]) for node in sub_path]
            speeds = np.asarray([vessel.speeds[0]] * (len(sub_path) - 1) + [None])
            individual = []
            for i, waypoint in enumerate(waypoints):
                individual.append([waypoint, speeds[i]])
            path_individuals.append(container(individual))
        global_routes.append(path_individuals)

    n_paths = {i: len(global_routes[i]) for i in range(len(global_routes))}
    return global_routes, n_paths


def init_individual(toolbox, individual):
    # Mutate graph route to obtain a population of initial routes
    init_ind = toolbox.clone(individual)
    for i in range(max(100, len(init_ind) // 10)):
        toolbox.mutate(init_ind, initializing=True)
    return init_ind
