import great_circle
import hexagraph
import heapq
import itertools
import math
import networkx as nx
import numpy as np


class Initializer:
    def __init__(self,
                 start,
                 end,
                 geod,
                 vessel,
                 resolution,
                 prep_polys,
                 rtree_idx,
                 toolbox,
                 graph_d=6,
                 graph_vd=4):
        self.start = start                                                  # Start location
        self.end = end                                                      # End location
        self.geod = geod
        self.vessel = vessel                                                # Vessel class
        self.resolution = resolution                                        # Resolution of shorelines
        self.prep_polys = prep_polys
        self.rtree_idx = rtree_idx
        self.toolbox = toolbox
        self.graph_d = graph_d                                              # Density recursion number, graph
        self.graph_vd = graph_vd                                            # Variable density recursion number, graph

        # Load or build graph
        graph_dir = 'output/variable_density_geodesic_grids/'
        try:
            self.graph = nx.read_gpickle(graph_dir + 'res_{}_d{}_vd{}.gpickle'.format(self.resolution, self.graph_d,
                                                                                      self.graph_vd))
        except FileNotFoundError:
            # Initialize "Hexagraph"
            hexagon_graph = hexagraph.Hexagraph(self.prep_polys, self.rtree_idx, self.resolution,
                                                self.graph_d, self.graph_vd)
            self.graph = hexagon_graph.get_graph()

        # Add start and end node to graph
        # Compute distances to start and end locations
        d_start = {n: great_circle.distance(self.start[0], self.start[1], lon_lat[0], lon_lat[1], self.geod)
                   for n, lon_lat in nx.get_node_attributes(self.graph, 'lon_lat').items()}
        d_end = {n: great_circle.distance(self.end[0], self.end[1], lon_lat[0], lon_lat[1], self.geod)
                 for n, lon_lat in nx.get_node_attributes(self.graph, 'lon_lat').items()}

        # Add start and end nodes
        x_s, x_e = math.cos(self.start[0]) * math.cos(self.start[1]), math.cos(self.end[0]) * math.cos(self.end[1])
        y_s, y_e = math.sin(self.start[0]) * math.cos(self.start[1]), math.sin(self.end[0]) * math.cos(self.end[1])
        z_s, z_e = math.sin(self.start[1]), math.sin(self.end[1])
        self.graph.add_node('start', lon_lat=self.start, xyz=(x_s, y_s, z_s))
        self.graph.add_node('end', lon_lat=self.end, xyz=(x_e, y_e, z_e))

        # Add three shortest edges to start and end point after checking feasibility
        nr_edges = 0
        for node in heapq.nsmallest(10, d_start, key=d_start.get):
            p2 = self.graph.nodes[node]['lon_lat']
            if self.toolbox.edge_feasible(self.start, p2):
                nr_edges += 1
                self.graph.add_edge('start', node, miles=d_start[node])
                if nr_edges > 2:
                    break
            else:
                print('Shortest edge to start not feasible')
        assert nr_edges > 0
        nr_edges = 0
        for node in heapq.nsmallest(10, d_end, key=d_end.get):
            p2 = self.graph.nodes[node]['lon_lat']
            if self.toolbox.edge_feasible(self.end, p2):
                nr_edges += 1
                self.graph.add_edge('end', node, miles=d_end[node])
                if nr_edges > 2:
                    break
            else:
                print('Shortest edge to end not feasible')
        assert nr_edges > 0

        self.canals = {'Panama': ['panama_south', 'panama_north'],
                       'Suez': ['suez_south', 'suez_north'],
                       'Dardanelles': ['dardanelles_south', 'dardanelles_north']}
        self.canal_nodes = [n for element in self.canals.values() for n in element]

    def dist_heur(self, n1, n2):
        (lon1, lat1) = self.graph.nodes[n1]['lon_lat']
        (lon2, lat2) = self.graph.nodes[n2]['lon_lat']
        return great_circle.distance(lon1, lat1, lon2, lat2, self.geod)

    def get_path(self, print_crossing=True):
        """
        Calculate shortest path from
        Returns:
             path: list of sub paths of a shortest path from start to end
             x_canal_pairs: list with tuples of crossed canal nodes in order of crossing
        """

        # path = [nx.astar_path(self.graph, 'start', 'end', heuristic=self.dist_heur, weight='miles')]
        path = [nx.shortest_path(self.graph, 'start', 'end', weight='miles', method='dijkstra')]
        x_canal_nodes = [wp for wp in path[0] if wp in self.canal_nodes]

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

    def get_init_routes(self, container):
        path, x_canal_pairs = self.get_path()
        paths = [path]
        if x_canal_pairs:
            route_combinations = itertools.product(*[(True, False)] * len(x_canal_pairs))
            for rc in route_combinations:
                excl_canals = [canal for i, canal in enumerate(x_canal_pairs) if rc[i]]
                self.graph.remove_edges_from(excl_canals)
                try:
                    alternative_path, _ = self.get_path(print_crossing=False)
                    self.graph.add_edges_from(excl_canals)
                except nx.exception.NetworkXNoPath:
                    self.graph.add_edges_from(excl_canals)
                    continue
                if alternative_path not in paths:
                    paths.append(alternative_path)

        # Create individual of each sub path
        init_routes = []
        for path in paths:
            path_individuals = []
            for sub_path in path:
                waypoints = [(self.graph.nodes[node]['lon_lat'][0], self.graph.nodes[node]['lon_lat'][1])
                             for node in sub_path]
                speeds = np.asarray([self.vessel.speeds[0]] * (len(sub_path) - 1) + [None])
                individual = []
                for i, waypoint in enumerate(waypoints):
                    individual.append([waypoint, speeds[i]])
                path_individuals.append(container(individual))
            init_routes.append(path_individuals)
        return init_routes


def init_individual(toolbox, individual_in):
    # Mutate graph route to obtain a population of initial routes
    mutant = toolbox.clone(individual_in)
    for i in range(max(100, len(individual_in) // 10)):
        mutant, = toolbox.mutate(mutant, initializing=True)
    return mutant
