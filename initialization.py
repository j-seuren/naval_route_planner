from data_config import hexagraph
import heapq
import itertools
import networkx as nx
import random

from haversine import haversine
from math import cos, sin


class Initializer:
    def __init__(self,
                 vessel,
                 tree,
                 ecaTree,
                 toolbox,
                 geod,
                 parameters,
                 container,
                 dens=6,
                 varDens=4):
        self.vessel = vessel          # Vessel class
        self.res = parameters['res']  # Resolution of shorelines
        self.tree = tree              # R-tree spatial index for shorelines
        self.ecaTree = ecaTree        # R-tree spatial index for ECAs
        self.toolbox = toolbox        # Function toolbox
        self.geod = geod              # Geod class instance
        self.container = container    # Container for individual
        self.dens = dens              # Density recursion number, graph
        self.varDens = varDens        # Variable density recursion number, graph
        self.canals = {'Panama': ['panama_south', 'panama_north'],
                       'Suez': ['suez_south', 'suez_north']}
        self.hexagraph = hexagraph.Hexagraph(self.tree, self.ecaTree, parameters)

    def get_path(self, graph):
        """
        Calculate shortest path from 'start' to 'end' on self.G
        Returns:
             path: list of sub paths of the shortest path
             x_cpairs: list of tuples of crossed canal nodes
                       in order of crossing
        """

        def dist_heuristic(n1, n2):
            return self.geod.distance(graph.nodes[n1]['deg'], graph.nodes[n2]['deg'])

        # Compute time shortest path on graph from 'start' to 'end'
        timePath = nx.astar_path(graph, 'start', 'end', heuristic=dist_heuristic, weight='miles')

        # Get crossed canal nodes in path
        canalNodes = [n for val in self.canals.values() for n in val]
        xCanalNodes = [wp for wp in timePath if wp in canalNodes]

        canalNodeDict = {canalNodes[0]: canal for canal, canalNodes in self.canals.items()}
        xCanals = [canalNodeDict[xCanalNode] for xCanalNode in xCanalNodes if xCanalNode in canalNodeDict]

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
                path[k]['eca'] = nx.astar_path(graph, start, end, heuristic=dist_heuristic, weight='eca_weight')
        else:
            path = {0: {'time': timePath,
                        'eca': nx.astar_path(graph, 'start', 'end', heuristic=dist_heuristic, weight='eca_weight')
                        }
                    }

        return path, list(zip(xCanalNodes[::2], xCanalNodes[1::2])), xCanals

    def get_initial_paths(self, start, end):
        """
        First, add start and end nodes to graph, and create edges to three nearest nodes.
        Next, compute shortest path from start to end over graph using A*.
        If the path crosses either Panama or Suez canal, also compute paths excluding each canal.
        Return dictionary of initial paths.
        """

        graph = self.hexagraph.get_graph()

        # Add start and end nodes to graph, and create edges to three nearest nodes.
        points = {'start': start, 'end': end}
        for ptKey, endPoint in points.items():
            # Compute distances of points in neighborhood to pt location
            neighborhood = {n: d['deg'] for n, d in graph.nodes(data=True)
                            if abs(d['deg'][0] - endPoint[0]) < 1 and abs(d['deg'][1] - endPoint[1]) < 1}
            assert len(neighborhood) > 0, 'No nearest graph point found for {} location {}'.format(ptKey, endPoint)
            d = {n: haversine(endPoint, nDeg, unit='nmi') for n, nDeg in neighborhood.items()}

            # Add pt node to graph
            lon, lat = endPoint
            x, y, z = cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)

            graph.add_node(ptKey, deg=endPoint, xyz=(x, y, z))

            # Add three shortest edges to  point
            # after checking feasibility
            nrEdges = infEdge = 0
            distsToPtList = heapq.nsmallest(100, d, key=d.get)
            distsToPt = iter(distsToPtList)
            nearestNeighbor = distsToPtList[0]
            while nrEdges < 3 and infEdge < 6:
                try:
                    n = next(distsToPt)
                except StopIteration:
                    print('Too few points in neighborhood')
                    break
                neighbor = graph.nodes[n]['deg']
                if self.toolbox.e_feasible(endPoint, neighbor):
                    nrEdges += 1
                    graph.add_edge(ptKey, n, miles=d[n])
                else:
                    infEdge += 1

            # If no feasible nearest edges, set nearest neighbor as start/end location
            if nrEdges == 0:
                print('No feasible shortest edges to {} location'.format(ptKey))
                graph.nodes[ptKey]['deg'] = tuple(x+.00001 for x in graph.nodes[nearestNeighbor]['deg'])
                graph.add_edge(ptKey, nearestNeighbor, miles=0)

        # Get shortest path on graph, and node pairs of crossed canals in order of crossing
        route, xCanalPairs, xCanals = self.get_path(graph)
        routes = {0: {'path': route, 'xCanals': xCanals}}
        if xCanalPairs:
            # Calculate route combinations
            routeCombs = itertools.product(*[(True, False)] * len(xCanalPairs))
            rKey = 0
            print(list(routeCombs))
            for routeComb in routeCombs:
                # Get list of canal pairs to be removed from graph
                delCanals = [c for i, c in enumerate(xCanalPairs) if routeComb[i]]
                graph.remove_edges_from(delCanals)
                alterPath, _, xCanals = self.get_path(graph)
                graph.add_edges_from(delCanals)

                paths = [route['path'] for route in routes.values()]
                if alterPath not in paths:
                    rKey += 1
                    routes[rKey] = {'path': alterPath, 'xCanals': xCanals}

        # Create individual of each sub path
        initialRoutes = {}
        for rKey, route in routes.items():
            initialRoutes[rKey] = {'xCanals': route['xCanals'], 'path': {}}
            for spKey, subPath in route['path'].items():
                initialRoutes[rKey]['path'][spKey] = {}
                for objKey, objPath in subPath.items():
                    # Set initial boat speed to max boat speed
                    wps = [graph.nodes[n]['deg'] for n in objPath]
                    speeds = [self.vessel.speeds[0]] * (len(objPath)-1) + [None]

                    ind = [list(tup) for tup in zip(wps, speeds)]
                    initialRoutes[rKey]['path'][spKey][objKey] = self.container(ind)

        return initialRoutes


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
