from math import acos, radians, cos, sin
import networkx as nx
from heapq import nsmallest
import numpy as np


def arc_length(xyz1, xyz2):
    return 3440 * acos(sum(p * q for p, q in zip(xyz1, xyz2)))


def graph_route(container, start, end, vessel):
    # Load the graph file
    G = nx.read_gpickle("output/final_d6_vd0.gpickle")

    # Compute distances to start and end locations
    lon_s, lat_s, lon_e, lat_e = radians(start[0]), radians(start[1]), radians(end[0]), radians(end[1])
    xyz_s = (cos(lat_s) * cos(lon_s), cos(lat_s) * sin(lon_s), sin(lat_s))
    xyz_e = (cos(lat_e) * cos(lon_e), cos(lat_e) * sin(lon_e), sin(lat_e))
    distances_to_start = {n: arc_length(xyz_s, xyz) for n, xyz in nx.get_node_attributes(G, 'xyz').items()}
    distances_to_end = {n: arc_length(xyz_e, xyz) for n, xyz in nx.get_node_attributes(G, 'xyz').items()}

    # Add start and end nodes
    G.add_node('start', lon_lat=start, xyz=xyz_s)
    G.add_node('end', lon_lat=end, xyz=xyz_e)

    # Add three shortest edges to start and end point
    for node in nsmallest(3, distances_to_start, key=distances_to_start.get):
        G.add_edge('start', node, miles=distances_to_start[node])
    for node in nsmallest(3, distances_to_end, key=distances_to_end.get):
        G.add_edge('end', node, miles=distances_to_end[node])

    # Calculate shortest path
    path = nx.shortest_path(G, 'start', 'end', weight='miles', method='dijkstra')

    waypoints = [(G.nodes[node]['lon_lat'][0], G.nodes[node]['lon_lat'][1]) for node in path]
    speeds = np.asarray([vessel.speeds[0]] * (len(path) - 1) + [None])
    individual_list = []
    for i, waypoint in enumerate(waypoints):
        individual_list.append([waypoint, speeds[i]])
    return container(individual_list)


def init_individual(toolbox, graph_ind):
    # Mutate graph route to obtain a population of initial routes
    init_ind = toolbox.clone(graph_ind)
    for i in range(max(100, len(init_ind))):
        toolbox.mutate(init_ind, check_feasible=True)
    return init_ind
