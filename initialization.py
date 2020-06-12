# Importing required modules
import networkx as nx
import classes
import pickle
import sys
from math import radians, cos, sin, acos
from operator import attrgetter
from heapq import nsmallest
from copy import deepcopy

import matplotlib.pyplot as plt


def arc_length(xyz1, xyz2):
    return 3440 * acos(sum(p * q for p, q in zip(xyz1, xyz2)))


# Generate as set of initial solutions
def initialization(start, end, vessel_f, population_size, rtree_idx, polygons, max_edge_length_f):
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

    # pos = nx.get_node_attributes(G, 'lon_lat')
    # path_edges = set(zip(path, path[1:]))
    # nx.draw(G, pos, node_color='k', node_size=1)
    # nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='r', node_size=1)
    # nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=1)
    # plt.axis('equal')
    # plt.show()

    # Create waypoints from path nodes
    waypoints = [classes.Waypoint(G.nodes[node]['lon_lat'][0], G.nodes[node]['lon_lat'][1]) for node in path]

    # Create edges from waypoints
    edge_list = zip(waypoints[:-1], waypoints[1:])
    edges = [classes.Edge(e[0], e[1], vessel_f.speeds[0]) for e in edge_list]

    # Create and save graph route from edges
    graph_route = classes.Route(edges)
    with open('output/graph_route', 'wb') as f:
        pickle.dump(graph_route, f)

    # Mutate graph route to obtain a population of initial routes
    initial_routes = []
    for i in range(population_size):
        # Printing process
        sys.stdout.write('\rGenerating initial routes {0}/{1}'.format(i + 1, population_size))
        sys.stdout.flush()

        init_route = deepcopy(graph_route)
        waypoints_inserted = 0
        while True:
            longest_edge = max(init_route.edges, key=attrgetter('miles'))

            # At least add 10 waypoints and make sure there exist no long edges
            if longest_edge.miles < max_edge_length_f and waypoints_inserted > 10:
                break
            init_route.insert_waypoint(width_ratio=0.5, rtree_idx=rtree_idx, polygons=polygons,
                                       edge=longest_edge)
            waypoints_inserted += 1
        initial_routes.append(init_route)

    print('\n')
    return initial_routes
