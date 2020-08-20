from data_config import hexagraph
import heapq
import itertools
import networkx as nx
import os
import random

from haversine import haversine
from math import cos, sin


class Initializer:
    def __init__(self,
                 vessel,
                 res,
                 tree,
                 ecaTree,
                 toolbox,
                 geod,
                 container,
                 dens=6,
                 varDens=4):
        self.vessel = vessel          # Vessel class
        self.res = res                # Resolution of shorelines
        self.tree = tree              # R-tree spatial index for shorelines
        self.ecaTree = ecaTree        # R-tree spatial index for ECAs
        self.toolbox = toolbox        # Function toolbox
        self.geod = geod              # Geod class instance
        self.container = container    # Container for individual
        self.dens = dens              # Density recursion number, graph
        self.varDens = varDens        # Variable density recursion number, graph

        # Load or build graph
        graphDir = 'output/variable_density_geodesic_grids/'
        graphFN = 'res_{}_d{}_vd{}.gpickle'.format(self.res,
                                                   self.dens,
                                                   self.varDens)
        try:
            self.G = nx.read_gpickle(os.path.join(graphDir, graphFN))
        except FileNotFoundError:
            # Initialize "Hexagraph"
            hexGraph = hexagraph.Hexagraph(dens,
                                           varDens,
                                           res,
                                           tree,
                                           ecaTree)
            self.G = hexGraph.graph

        self.canals = {'Panama': ['panama_south', 'panama_north'],
                       'Suez': ['suez_south', 'suez_north']}
        self.canalNodes = [n for val in self.canals.values() for n in val]

    def dist_heuristic(self, n1, n2):
        return self.geod.distance(self.G.nodes[n1]['deg'],
                                  self.G.nodes[n2]['deg'])

    def get_path(self):
        """
        Calculate shortest path from 'start' to 'end' on self.G
        Returns:
             path: list of sub paths of the shortest path
             x_cpairs: list of tuples of crossed canal nodes
                       in order of crossing
        """
        # Compute time shortest path on graph from 'start' to 'end'
        timePath = nx.astar_path(self.G, 'start', 'end',
                                 heuristic=self.dist_heuristic,
                                 weight='miles')

        # Get crossed canal nodes in path
        xCanalNodes = [wp for wp in timePath if wp in self.canalNodes]

        if xCanalNodes:
            print('Crossing canals {}'.format(xCanalNodes), end='\n ')

            cuts = [timePath.index(i) for i in xCanalNodes[1::2]]
            subPath = [timePath[:cuts[0]]]
            subPath.extend([timePath[cuts[i]:cuts[i+1]]
                            for i in range(len(cuts) - 1)])
            subPath.append(timePath[cuts[-1]:])
            path = {i: {'time': sp} for i, sp in enumerate(subPath)}

            # Compute ECA shortest path for each sub path
            for k in path:
                start = path[k]['time'][0]
                end = path[k]['time'][-1]
                path[k]['eca'] = nx.astar_path(self.G, start, end,
                                               heuristic=self.dist_heuristic,
                                               weight='eca_weight')
        else:
            path = {0: {'time': timePath,
                        'eca': nx.astar_path(self.G, 'start', 'end',
                                             heuristic=self.dist_heuristic,
                                             weight='eca_weight')
                        }
                    }

        return path, list(zip(xCanalNodes[::2], xCanalNodes[1::2]))

    def get_initial_paths(self, start, end):
        """
        First, add start and end nodes to graph, and create edges to three nearest nodes.
        Next, compute shortest path from start to end over graph using A*.
        If the path crosses either Panama or Suez canal, also compute paths excluding each canal.
        Return dictionary of initial paths.
        """

        # Add start and end nodes to graph, and create edges to three nearest nodes.
        points = {'start': start, 'end': end}
        for ptKey, pt in points.items():
            # Compute distances to pt location
            d = {n: haversine(pt, nDeg, unit='nmi') for n, nDeg in nx.get_node_attributes(self.G, 'deg').items()}

            # Add pt node to graph
            lon, lat = pt
            x, y, z = cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)

            self.G.add_node(ptKey, deg=pt, xyz=(x, y, z))

            # Add three shortest edges to  point
            # after checking feasibility
            nrEdges = 0
            distsToPt = iter(heapq.nsmallest(10, d, key=d.get))
            while nrEdges < 3:
                n = next(distsToPt)
                neighbour = self.G.nodes[n]['deg']
                if self.toolbox.e_feasible(pt, neighbour):
                    nrEdges += 1
                    self.G.add_edge(ptKey, n, miles=d[n])
                else:
                    print('Shortest edge to {} not feasible'.format(ptKey))

        # Get shortest path on graph, and node pairs of crossed canals in order of crossing
        path, xCanalPairs = self.get_path()
        paths = {0: path}
        if xCanalPairs:
            # Calculate route combinations
            routeCombs = itertools.product(*[(True, False)] * len(xCanalPairs))
            pathKey = 0
            for routeComb in routeCombs:
                # Get list of canal pairs to be removed from graph
                delCanals = [c for i, c in enumerate(xCanalPairs) if routeComb[i]]
                self.G.remove_edges_from(delCanals)
                alterPath, _ = self.get_path()
                self.G.add_edges_from(delCanals)
                if alterPath not in paths.values():
                    pathKey += 1
                    paths[pathKey] = alterPath

        # Create individual of each sub path
        initInds = {}
        for pathKey, path in paths.items():
            initInds[pathKey] = {}
            for spKey, subPath in path.items():
                initInds[pathKey][spKey] = {}
                for objKey, objPath in subPath.items():
                    wps = [self.G.nodes[n]['deg'] for n in objPath]

                    # Set initial boat speed to max boat speed
                    speeds = [self.vessel.speeds[0]] * (len(objPath)-1) + [None]
                    ind = [list(tup) for tup in zip(wps, speeds)]
                    initInds[pathKey][spKey][objKey] = self.container(ind)

        # Delete graph from memory
        del self.G

        return initInds


def init_individual(toolbox, indsIn, i):  # swaps: ['speed', 'insert', 'move', 'delete']
    mutants = []
    if i == 0:
        return list(indsIn.values())
    for indIn in indsIn.values():
        cumWeights = [10, 20, 30, 100]
        k = random.randint(1, 8)
        mutant, = toolbox.mutate(toolbox.clone(indIn), initializing=True,
                                 cumWeights=cumWeights, k=k)
        mutants.append(mutant)

    return mutants


def init_repeat_list(func, n):
    """Call a list with a generator function corresponding
    to the calling *n* times the function *func*.

    :param func: The function that will be called n times to fill the
                 container.
    :param n: The number of times to repeat func.
    :returns: A list filled with data from returned list of func.
    """
    return [el for i in range(n) for el in func(i=i)]
