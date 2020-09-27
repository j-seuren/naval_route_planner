# http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

from copy import deepcopy
from data_config.navigable_area import NavigableAreaGenerator
from evaluation import geo_x_geos
from analysis.geodesic import Geodesic
from mpl_toolkits.mplot3d import Axes3D
from math import atan2, degrees, sqrt, radians, cos, sin
from pathlib import Path
from heapq import nsmallest
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class Triangle:
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3


class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Hexagraph:
    def __init__(self,
                 treeDict,
                 ecaTreeDict,
                 p,
                 DIR=Path('D:/')):
        self.treeDict = treeDict
        self.ecaTreeDict = ecaTreeDict
        self.p = p

        # Check whether Antarctic and Arctic circles are included as impassable areas
        aC = aAc = 'incl'
        if p['avoidAntarctic'] and p['avoidArctic']:
            aC = aAc = 'avoid'
        elif p['avoidAntarctic']:
            aAc = 'excl'
        elif p['avoidArctic']:
            aC = 'excl'

        # Generate file path
        graphFN = 'res_{}_d{}_vd{}_{}Ant_{}Arc.gpickle'.format(p['res'], p['graphDens'], p['graphVarDens'], aAc, aC)
        self.graphFP = DIR / 'data' / graphFN

    def get_graph(self):
        # Load or construct graph
        try:
            graph = nx.read_gpickle(self.graphFP)
            print('Loaded graph file from: ', self.graphFP)
            return graph
        except FileNotFoundError:
            # Initialize "Hexagraph"
            constructor = self.GraphConstructor(self.treeDict, self.ecaTreeDict, self.p)
            graph = constructor.construct()

            # Save graph to file
            nx.write_gpickle(graph,  self.graphFP)
            print('Built and saved graph to: ', self.graphFP)
            return graph

    def plot_graph(self, draw='both', showLongEdges=False):
        graph = self.get_graph()
        if not showLongEdges:
            edgeDataDict = deepcopy(graph.edges())
            for n1, n2 in edgeDataDict:
                # Get positions of nodes from current edge
                p1, p2 = graph.nodes[n1]['deg'], graph.nodes[n2]['deg']
                if geo_x_geos(self.treeDict, p1, p2):
                    graph.remove_edge(n1, n2)
        pos = nx.get_node_attributes(graph, 'deg')
        fig, ax = plt.subplots()
        if draw == 'both':
            nx.draw(graph, pos=pos, node_size=1, ax=ax)
        elif draw == 'nodes':
            print('drawing nodes only')
            nx.draw_networkx_nodes(graph, pos=pos, node_size=1, ax=ax)
        else:
            print('drawing edges only')
            nx.draw(graph, pos=pos, node_size=0, ax=ax)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        return fig, ax

    def plot_sphere(self, elevationAngle=0, azimuthAngle=0, graph=None):
        if not graph:
            graph = self.get_graph()
        # 3D plot of unit sphere nodes
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(azim=azimuthAngle, elev=elevationAngle)
        xyz = nx.get_node_attributes(graph, 'xyz')
        xyz = np.array([val for val in xyz.values()])
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, color='black')
        return fig, ax

    def plot_sphere_edges(self, elevationAngle=0, azimuthAngle=0, graph=None):
        if not graph:
            graph = self.get_graph()
        xyz = nx.get_node_attributes(graph, 'xyz')
        edges = np.array(graph.edges)
        lines = np.empty([len(edges), 2, 3])
        for idx, edge in enumerate(edges):
            try:
                e1, e2 = int(edge[0]), int(edge[1])
            except ValueError:
                continue
            lines[idx] = [np.asarray(xyz[e1]), np.asarray(xyz[e2])]
        fig = plt.figure(figsize=[6, 6])
        ax = fig.gca(projection='3d')
        ax.plot([], [], [], 'k.', markersize=5, alpha=0.9)
        ax.view_init(azim=azimuthAngle, elev=elevationAngle)
        ax.add_collection(Line3DCollection(lines, colors='black', linewidths=1))
        minmax = [-.6, .6]
        ax.set_xlim(minmax)
        ax.set_ylim(minmax)
        ax.set_zlim(minmax)
        return fig, ax

    class GraphConstructor:
        def __init__(self, treeDict, ecaTreeDict, p):
            self.treeDict = treeDict
            self.ecaTreeDict = ecaTreeDict
            self.exteriorTreeDict = NavigableAreaGenerator(p).get_shoreline_rtree(getExterior=True)
            self.distance = Geodesic(dist_calc='great_circle').distance
            self.recursionLevel = p['graphDens']
            self.varRecursionLevel = p['graphVarDens']
            self.graph = nx.Graph()
            self.nodeIdx = 0
            self.mpCache, self.points3D = {}, {}
            self.canals = {'Panama': {'nodes': {'panama_south': {'deg': (-79.540932, 8.894197), 'xyz': (.0,) * 3},
                                                'panama_north': {'deg': (-79.919005, 9.391057), 'xyz': (.0,) * 3}},
                                      'dist': 43},
                           'Suez': {'nodes': {'suez_south': {'deg': (32.5164, 29.9159), 'xyz': (.0,) * 3},
                                              'suez_north': {'deg': (32.3678, 31.2678), 'xyz': (.0,) * 3}},
                                    'dist': 89},
                           'Dardanelles': {'nodes': {'dardanelles_south': {'deg': (26.1406, 40.0136), 'xyz': (.0,) * 3},
                                                     'dardanelles_north': {'deg': (26.9961, 40.4861), 'xyz': (.0,) * 3}
                                                     },
                                           'dist': 52}}
            for c in self.canals:
                for n in self.canals[c]['nodes']:
                    lo, la = (radians(deg) for deg in self.canals[c]['nodes'][n]['deg'])
                    self.canals[c]['nodes'][n]['xyz'] = (cos(la) * cos(lo), cos(la) * sin(lo), sin(la))

        def add_vertex(self, p):
            """ Add vertex to graph, fix position to be on unit sphere, return
            index """

            # Use length of vector to fix position on unit sphere, and create 3D point
            length = sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
            x, y, z = p.x / length, p.y / length, p.z / length
            p3d = Point3D(x, y, z)

            # Calculate longitude and latitude from x, y, z
            lon = degrees(atan2(y, x))
            lat = degrees(atan2(z, sqrt(x * x + y * y)))

            # Add node to graph
            self.graph.add_node(self.nodeIdx, xyz=(x, y, z), point3D=p3d, deg=(lon, lat))
            self.points3D[self.nodeIdx] = Point3D(x, y, z)
            self.nodeIdx += 1
            return self.nodeIdx - 1

        def get_middle_point(self, idx1, idx2):
            """Return index of point in the middle of p1 and p2"""
            # First, check if point is already in cache
            key = tuple(sorted([idx1, idx2]))
            if key in self.mpCache:
                return self.mpCache[key]

            # If not in cache: calculate the point
            p1 = self.graph.nodes[idx1]['point3D']
            p2 = self.graph.nodes[idx2]['point3D']
            middle = Point3D((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)

            # Add vertex makes sure point is on unit sphere
            idx = self.add_vertex(middle)
            self.mpCache[key] = idx
            return idx

        def remove_nodes_in_polys(self):
            cnt, nNodes = 0, len(self.graph.nodes)
            sys.stdout.write("Removing nodes from graph:  {}%".format(int(100 * cnt / nNodes)))
            sys.stdout.flush()
            nodeDataDict = deepcopy(self.graph.nodes(data=True))
            removed = 0
            for node_data in nodeDataDict:
                # Printing progress
                cnt += 1
                if cnt % 100 == 0:
                    sys.stdout.write("\rRemoving nodes from graph:  {}%".format(int(100 * cnt / nNodes)))
                    sys.stdout.flush()

                # Get indices and positions of nodes from current edge
                n = node_data[0]
                p = self.graph.nodes[n]['deg']

                # Remove nodes on land
                if geo_x_geos(self.treeDict, p):
                    self.graph.remove_node(n)
                    removed += 1
            sys.stdout.write("\rRemoved {} nodes\n".format(removed))
            sys.stdout.flush()

        def remove_edges_x_polys(self):
            cnt, nEdges = 0, len(self.graph.edges)
            sys.stdout.write("Removing edges from graph:  {}%".format(int(100 * cnt / nEdges)))
            sys.stdout.flush()
            edgeDataDict = deepcopy(self.graph.edges())
            removed = 0
            for n1, n2 in edgeDataDict:
                # Printing progress
                cnt += 1
                if cnt % 100 == 0:
                    sys.stdout.write("\rRemoving edges from graph:  {}%".format(int(100 * cnt / nEdges)))
                    sys.stdout.flush()

                # Get positions of nodes from current edge
                p1, p2 = self.graph.nodes[n1]['deg'], self.graph.nodes[n2]['deg']

                # Remove edges intersecting land, but skip border edges
                # since they always intersect polygons if projected on 2D
                if abs(p1[0] - p2[0]) < 340 and geo_x_geos(self.treeDict, p1, p2):
                    self.graph.remove_edge(n1, n2)
                    removed += 1
            sys.stdout.write("\rRemoved {} edges\n".format(removed))
            sys.stdout.flush()

        def construct(self):
            print('Constructing hexagraph')
            # Create 12 vertices of the icosahedron
            phi = (1 + sqrt(5)) / 2  # Golden ratio
            [self.add_vertex(Point3D(j, i, 0)) for i in [phi, -phi] for j in [-1, 1]]
            [self.add_vertex(Point3D(0, j, i)) for i in [phi, -phi] for j in [-1, 1]]
            [self.add_vertex(Point3D(i, 0, j)) for i in [phi, -phi] for j in [-1, 1]]

            # Create 20 triangles of the icosahedron
            # 5 faces around point 0
            tris = [Triangle(0, 11, 5),
                    Triangle(0, 5, 1),
                    Triangle(0, 1, 7),
                    Triangle(0, 7, 10),
                    Triangle(0, 10, 11)]

            # 5 adjacent faces
            tris.extend([Triangle(1, 5, 9),
                        Triangle(5, 11, 4),
                        Triangle(11, 10, 2),
                        Triangle(10, 7, 6),
                        Triangle(7, 1, 8)])

            # 5 faces around point 3
            tris.extend([Triangle(3, 9, 4),
                         Triangle(3, 4, 2),
                         Triangle(3, 2, 6),
                         Triangle(3, 6, 8),
                         Triangle(3, 8, 9)])

            # 5 adjacent faces
            tris.extend([Triangle(4, 9, 5),
                        Triangle(2, 4, 11),
                        Triangle(6, 2, 10),
                        Triangle(8, 6, 7),
                        Triangle(9, 8, 1)])

            # Refine triangles
            for i in range(self.recursionLevel):
                print('\rAdding nodes. Recursion level: {:2d}/{}'.format(i+1, self.recursionLevel),
                      end='')
                triangles2 = []
                for tri in tris:
                    # Replace triangle by 4 triangles
                    a = self.get_middle_point(tri.v1, tri.v2)
                    b = self.get_middle_point(tri.v2, tri.v3)
                    c = self.get_middle_point(tri.v3, tri.v1)
                    triangles2.append(Triangle(tri.v1, a, c))
                    triangles2.append(Triangle(tri.v2, b, a))
                    triangles2.append(Triangle(tri.v3, c, b))
                    triangles2.append(Triangle(a, b, c))
                tris = triangles2
            print('\rAdded nodes to graph')

            # Add triangles to mesh
            nTris = len(tris)
            for i, tri in enumerate(tris):
                print('\rCreating triangles connecting nodes: {:2d}/{}'.format(i+1, nTris), end='')
                self.graph.add_edge(tri.v1, tri.v2)
                self.graph.add_edge(tri.v2, tri.v3)
                self.graph.add_edge(tri.v3, tri.v1)
            print('\rCreated {} triangles'.format(nTris))

            # Refine graph near shorelines
            print('Refine graph near shorelines... ')
            for i in range(self.varRecursionLevel):
                triCache = {}  # Reinitialize triangle cache
                print('\r Refinement level: {}/{}'.format(i+1, self.varRecursionLevel), end='')
                edgesCopy = deepcopy(self.graph.edges)
                for edge in edgesCopy:
                    # Get indices and long/lat positions of nodes from current edge
                    n1, n2 = edge[0], edge[1]
                    p1, p2 = self.graph.nodes[n1]['deg'], self.graph.nodes[n2]['deg']

                    # Skip border edges since they always intersect polygons in a 2D grid map
                    if abs(p1[0] - p2[0]) > 340:
                        continue

                    # If edge crosses a polygon exterior, refine two adjacent triangles
                    if geo_x_geos(self.exteriorTreeDict, p1, p2):
                        # Get adjacent triangles of intersected edge
                        n3s = [e1[1] for e1 in edgesCopy(n1) for e2 in edgesCopy(n2) if e1[1] == e2[1]]
                        adjacentTris = [(n1, n2, n3) for n3 in n3s if tuple(sorted((n1, n2, n3))) not in triCache]

                        # Subdivide each triangle into four equal triangles
                        for tri in adjacentTris:
                            triCache[tuple(sorted(tri))] = 1  # Store to avoid splitting in the future

                            # Find middle point of edge u,v
                            # and add node and edges to graph
                            n1, n2, n3 = tri
                            a = self.get_middle_point(n1, n2)
                            # Subdivide edge into two equal edges
                            if self.graph.has_edge(n1, n2):
                                self.graph.remove_edge(n1, n2)
                            self.graph.add_edge(n1, a)
                            self.graph.add_edge(a, n2)

                            # Find middle point of edge v,w
                            # and add node and edges to graph
                            b = self.get_middle_point(n2, n3)
                            # Subdivide edge into two equal edges
                            if self.graph.has_edge(n2, n3):
                                self.graph.remove_edge(n2, n3)
                            self.graph.add_edge(n2, b)
                            self.graph.add_edge(b, n3)

                            # Find middle point of edge w,u
                            # and add node and edges to graph
                            c = self.get_middle_point(n3, n1)
                            # Subdivide edge into two equal edges
                            if self.graph.has_edge(n3, n1):
                                self.graph.remove_edge(n3, n1)
                            self.graph.add_edge(n3, c)
                            self.graph.add_edge(c, n1)

                            # Add inner edges of subdivided triangle
                            self.graph.add_edge(a, b)
                            self.graph.add_edge(b, c)
                            self.graph.add_edge(c, a)
            print('\r\r Refined graph at coastlines')

            # Delete 3DPoint attribute since it is not needed in the future
            for n in self.graph.nodes:
                del self.graph.nodes[n]['point3D']

            # Remove edges crossing polygons
            self.remove_nodes_in_polys()
            self.remove_edges_x_polys()

            for canal in self.canals:
                for n in self.canals[canal]['nodes']:
                    self.graph.add_node(n, deg=self.canals[canal]['nodes'][n]['deg'],
                                        xyz=self.canals[canal]['nodes'][n]['xyz'])

            # Link each canal node to its three nearest nodes
            for canal in self.canals:
                n1, n2 = (n for n in self.canals[canal]['nodes'])
                p1, p2 = (self.canals[canal]['nodes'][n]['deg'] for n in self.canals[canal]['nodes'])
                self.graph.add_edge(n1, n2)
                dist1 = {n: self.distance(p1, p) for n, p in nx.get_node_attributes(self.graph, 'deg').items()
                         if n != n1}
                dist2 = {n: self.distance(p2, p) for n, p in nx.get_node_attributes(self.graph, 'deg').items()
                         if n != n2}

                # Add edges from canal nodes to three nearest neighbours
                for nn in nsmallest(3, dist1, key=dist1.get):
                    self.graph.add_edge(n1, nn, miles=dist1[nn])
                for nn in nsmallest(3, dist2, key=dist2.get):
                    self.graph.add_edge(n2, nn, miles=dist2[nn])

            # Set edge attributes
            print('Setting edge attributes... ', end='')
            for n1, n2 in self.graph.edges():
                p1, p2 = self.graph.nodes[n1]['deg'], self.graph.nodes[n2]['deg']
                miles = self.distance(p1, p2)
                self.graph[n1][n2]['dist'] = miles

                if geo_x_geos(self.ecaTreeDict, p1, p2):
                    self.graph[n1][n2]['eca'] = miles * 2
                else:
                    self.graph[n1][n2]['eca'] = miles
            print('done')

            # Set canal edge weights to predefined weights
            for canal in self.canals:
                n1, n2 = (n for n in self.canals[canal]['nodes'])
                self.graph[n1][n2]['dist'] = self.canals[canal]['dist']

            print('Removing isolates... ', end='')
            nx.isolates(self.graph)
            self.graph.remove_nodes_from(list(nx.isolates(self.graph)))
            print('done')

            # Remove all connected components but the largest
            print('Removing isolate connected components... ', end='')
            for cc in sorted(nx.connected_components(self.graph), key=len, reverse=True)[1:]:
                self.graph.remove_nodes_from(cc)
            print('done')
            return self.graph
