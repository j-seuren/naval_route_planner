import hexagraph
import heapq
import itertools
import networkx as nx
import os

from haversine import haversine
from math import cos, sin


class Initializer:
    def __init__(self,
                 pt_s,
                 pt_e,
                 vessel,
                 res,
                 prep_polys,
                 rtree_idx,
                 toolbox,
                 gc,
                 dens=6,
                 var_dens=4):
        self.pt_s = pt_s              # Start point
        self.pt_e = pt_e              # End point
        self.vessel = vessel          # Vessel class
        self.res = res                # Resolution of shorelines
        self.prep_polys = prep_polys  # Prepared land polygons
        self.rtree_idx = rtree_idx    # prep_polys' R-tree spatial index
        self.toolbox = toolbox        # Function toolbox
        self.gc = gc                  # Geod class instance
        self.dens = dens              # Density recursion number, graph
        self.var_dens = var_dens      # Variable density recursion number, graph

        # Load or build graph
        graph_dir = 'output/variable_density_geodesic_grids/'
        graph_fn = 'res_{}_d{}_vd{}.gpickle'.format(self.res,
                                                    self.dens,
                                                    self.var_dens)
        try:
            self.G = nx.read_gpickle(os.path.join(graph_dir, graph_fn))
        except FileNotFoundError:
            # Initialize "Hexagraph"
            hex_graph = hexagraph.Hexagraph(dens, var_dens, res, prep_polys,
                                            rtree_idx)
            self.G = hex_graph.get_g()

        # Compute distances to start and end locations
        d_s = {n: haversine(self.pt_s, n_deg, unit='nmi')
               for n, n_deg in nx.get_node_attributes(self.G, 'deg').items()}
        d_e = {n: haversine(self.pt_e, n_deg, unit='nmi')
               for n, n_deg in nx.get_node_attributes(self.G, 'deg').items()}

        # Add start and end nodes to graph
        lo_s, la_s = self.pt_s
        lo_e, la_e = self.pt_e
        x_s, x_e = cos(lo_s) * cos(la_s), cos(lo_e) * cos(la_e)
        y_s, y_e = sin(lo_s) * cos(la_s), sin(lo_e) * cos(la_e)
        z_s, z_e = sin(la_s), sin(la_e)
        self.G.add_node('start', deg=self.pt_s, xyz=(x_s, y_s, z_s))
        self.G.add_node('end', deg=self.pt_e, xyz=(x_e, y_e, z_e))

        # Add three shortest edges to start and end point
        # after checking feasibility
        nr_edges = 0
        for n in heapq.nsmallest(10, d_s, key=d_s.get):
            p2 = self.G.nodes[n]['deg']
            if self.toolbox.e_feasible(self.pt_s, p2):
                nr_edges += 1
                self.G.add_edge('start', n, miles=d_s[n])
                if nr_edges > 2:
                    break
            else:
                print('Shortest edge to start not feasible')
        assert nr_edges > 0
        nr_edges = 0
        for n in heapq.nsmallest(10, d_e, key=d_e.get):
            p2 = self.G.nodes[n]['deg']
            if self.toolbox.e_feasible(self.pt_e, p2):
                nr_edges += 1
                self.G.add_edge('end', n, miles=d_e[n])
                if nr_edges > 2:
                    break
            else:
                print('Shortest edge to end not feasible')
        assert nr_edges > 0

        self.canals = {'Panama': ['panama_south', 'panama_north'],
                       'Suez': ['suez_south', 'suez_north']}
        self.canal_nodes = [n for val in self.canals.values() for n in val]

    def dist_heuristic(self, n1, n2):
        return self.gc.distance(self.G.nodes[n1]['deg'],
                                self.G.nodes[n2]['deg'])

    def get_path(self):
        """
        Calculate shortest path from 'start' to 'end' on self.G
        Returns:
             path: list of sub paths of the shortest path
             x_cpairs: list of tuples of crossed canal nodes
                       in order of crossing
        """
        # Calculate shortest path on graph from 'start' to 'end'
        path = [nx.astar_path(self.G, 'start', 'end',
                              heuristic=self.dist_heuristic,
                              weight='miles')]
        # path = [nx.shortest_path(self.graph, 'start', 'end', weight='miles',
        #                          method='dijkstra')]

        # Get crossed canal nodes in path
        x_cnodes = [wp for wp in path[0] if wp in self.canal_nodes]
        if x_cnodes:
            # Zip crossed canal nodes to crossed canal pairs
            x_cpairs = list(zip(x_cnodes[::2], x_cnodes[1::2]))
            pairs_reversed = x_cpairs[::-1]
            while pairs_reversed:
                # Get sub path in which canal node is found
                x_canal_pair = pairs_reversed.pop()
                sp = path.pop()

                # Split sub path at canal, and extend to path
                i = sp.index(x_canal_pair[0])
                path.extend([sp[:i+1], sp[i+1:]])
                print('Crossing canal {}'.format(x_canal_pair))
        else:
            x_cpairs = []
        return path, x_cpairs

    def get_init_inds(self, container):
        path, x_cpairs = self.get_path()
        paths = [path]
        if x_cpairs:
            # Calculate route combinations
            rcs = itertools.product(*[(True, False)] * len(x_cpairs))
            for rc in rcs:
                # Get list of canal pairs to be removed from graph
                excl_cs = [c for i, c in enumerate(x_cpairs) if rc[i]]
                self.G.remove_edges_from(excl_cs)
                try:
                    alternative_path, _ = self.get_path()
                    self.G.add_edges_from(excl_cs)
                except nx.exception.NetworkXNoPath:
                    self.G.add_edges_from(excl_cs)
                    continue
                if alternative_path not in paths:
                    paths.append(alternative_path)

        # Create individual of each sub path
        init_inds = []
        for path in paths:
            path_inds = []
            for sp in path:
                wps = [self.G.nodes[n]['deg'] for n in sp]

                # Set initial boat speed to max boat speed
                speeds = [self.vessel.speeds[0]] * (len(sp) - 1) + [None]
                ind = [list(tup) for tup in zip(wps, speeds)]
                path_inds.append(container(ind))
            init_inds.append(path_inds)
        return init_inds


def init_individual(toolbox, ind_in):
    mutant = toolbox.clone(ind_in)
    nr_mutations = max(100, len(ind_in) // 10)
    for i in range(nr_mutations):
        mutant, = toolbox.mutate(mutant, initializing=True)
    return mutant
