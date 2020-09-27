from data_config import hexagraph
import heapq
import itertools
import networkx as nx
import random

from haversine import haversine
from math import cos, sin


class Initializer:
    def __init__(self,
                 evaluator,
                 vessel,
                 tree,
                 ecaTree,
                 geod,
                 p,
                 container):
        self.evaluator = evaluator
        self.vessel = vessel          # Vessel class
        self.tree = tree              # R-tree spatial index for shorelines
        self.ecaTree = ecaTree        # R-tree spatial index for ECAs
        self.geod = geod              # Geod class instance
        self.container = container    # Container for individual
        self.canals = {'Panama': ['panama_south', 'panama_north'],
                       'Suez': ['suez_south', 'suez_north']}
        self.hexagraph = hexagraph.Hexagraph(self.tree, self.ecaTree, p)

    def get_path(self, graph):
        """
        Find shortest path from 'start' to 'end' on graph
        Returns:
             path: list of sub paths of the shortest path
             x_cpairs: list of tuples of crossed canal nodes
                       in order of crossing
        """

        def dist_heuristic(n1, n2):
            return self.geod.distance(graph.nodes[n1]['deg'], graph.nodes[n2]['deg'])

        # Compute time shortest path on graph from 'start' to 'end'
        timePath = nx.astar_path(graph, 'start', 'end', heuristic=dist_heuristic, weight='dist')

        # Get crossed canal nodes in path
        canalNodes = [n for val in self.canals.values() for n in val]
        xCanalNodes = [wp for wp in timePath if wp in canalNodes]

        canalNodeDict = {canalNodes[0]: canal for canal, canalNodes in self.canals.items()}
        xCanals = [canalNodeDict[xCanalNode] for xCanalNode in xCanalNodes if xCanalNode in canalNodeDict]

        if xCanalNodes:
            cuts = [timePath.index(i) for i in xCanalNodes[1::2]]
            timeSubPaths = [timePath[:cuts[0]]]
            timeSubPaths.extend([timePath[cuts[i]:cuts[i+1]]
                            for i in range(len(cuts) - 1)])
            timeSubPaths.append(timePath[cuts[-1]:])

            path = [{'time': timeSubPath} for timeSubPath in timeSubPaths]

            # Compute ECA shortest path for each sub path
            for subPath in path:
                start, end = subPath['time'][0], subPath['time'][-1]
                subPath['eca'] = nx.astar_path(graph, start, end, heuristic=dist_heuristic, weight='eca')
        else:
            path = [{'time': timePath,
                     'eca': nx.astar_path(graph, 'start', 'end', heuristic=dist_heuristic, weight='eca')}]

        return path, xCanals

    def get_initial_routes(self, start, end):
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
            radius = 0
            while True:
                radius += 0.2
                neighborhood = {n: d['deg'] for n, d in graph.nodes(data=True)
                                if abs(d['deg'][0] - endPoint[0]) < radius and abs(d['deg'][1] - endPoint[1]) < radius}
                if len(neighborhood) > 3:
                    print('neighborhood radius', radius, ', neighbors', len(neighborhood))
                    break
            d = {n: haversine(endPoint, nDeg, unit='nmi') for n, nDeg in neighborhood.items()}

            # Add pt node to graph
            lon, lat = endPoint
            x, y, z = cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)

            graph.add_node(ptKey, deg=endPoint, xyz=(x, y, z))

            # Add three shortest feasible edges to  point
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
                if self.evaluator.e_feasible(endPoint, neighbor):
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
        path, xCanals = self.get_path(graph)
        paths = [{'path': path, 'xCanals': xCanals}]
        if xCanals:
            print('Crossing canals {}'.format(xCanals), end='\n ')
            # Calculate route combinations
            routeCombs = []
            for r in range(len(xCanals)):
                routeCombs.extend(list(itertools.combinations(xCanals, r+1)))

            for routeComb in routeCombs:
                # Get list of canal pairs to be removed from graph
                delCanalNodes = [self.canals[c] for c in routeComb]
                graph.remove_edges_from(delCanalNodes)
                alterPath, alterXCanals = self.get_path(graph)
                graph.add_edges_from(delCanalNodes)

                if alterXCanals not in [pathDict['xCanals'] for pathDict in paths]:
                    paths.append({'path': alterPath, 'xCanals': alterXCanals})

        # Create individual of each sub path
        initialRoutes = self.paths_to_routes(paths, graph)

        return initialRoutes

    def paths_to_routes(self, paths, graph):
        routes = []
        for pathDict in paths:
            route = {'route': [], 'xCanals': pathDict['xCanals']}
            for subPath in pathDict['path']:
                subRoute = {}
                for obj, objPath in subPath.items():
                    # Set initial boat speed to max boat speed
                    wps = [graph.nodes[n]['deg'] for n in objPath]
                    speeds = [self.vessel.speeds[0]] * (len(objPath) - 1) + [None]

                    ind = [list(tup) for tup in zip(wps, speeds)]
                    subRoute[obj] = self.container(ind)
                route['route'].append(subRoute)
            routes.append(route)

        return routes


def init_individual(toolbox, indsIn, i):  # swaps: ['speed', 'insert', 'move', 'delete']
    mutants = []
    if i == 0:
        return list(indsIn.values())
    for indIn in indsIn.values():
        k = random.randint(1, 8)
        mutant, = toolbox.mutate(toolbox.clone(indIn), initializing=True, k=k)
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
