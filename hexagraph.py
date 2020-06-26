# http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
import fiona
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import rtree
import sys

from mpl_toolkits.mplot3d import Axes3D
from math import atan2, degrees, acos, sqrt, radians, cos, sin
from shapely.geometry import shape, Point, LineString
from shapely.prepared import prep
from heapq import nsmallest
from katana import get_split_polygons


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


def edge_x_geos(p1, p2, rtree_idx_geos, geos):
    line_bounds = (min(p1[0], p2[0]), min(p1[1], p2[1]),
                   max(p1[0], p2[0]), max(p1[1], p2[1]))

    # Returns the geometry indices of the minimum bounding rectangles that intersect the bounding box of line u,v
    mbr_intersections = list(rtree_idx_geos.intersection(line_bounds))
    if mbr_intersections:  # Create LineString if there is at least one minimum bounding rectangle intersection
        shapely_line = LineString([p1, p2])
        for idx in mbr_intersections:
            if geos[idx].intersects(shapely_line):
                return True
    return False


class HexagraphBuilder:
    def __init__(self,
                 d,
                 vd,
                 res,
                 prep_polys,
                 rtree_idx):
        self.d = d
        self.vd = vd
        self.res = res
        self.prep_polys = prep_polys
        self.rtree_idx = rtree_idx
        self.graph = nx.Graph()
        self.index = 0
        self.mp_cache = {}
        self.points = {}
        self.tri_cache = []
        self.canals = {'Panama': {'nodes': {'panama_south': {'lon_lat': (-79.540932, 8.894197), 'xyz': (.0,) * 3},
                                            'panama_north': {'lon_lat': (-79.919005, 9.391057), 'xyz': (.0,) * 3}},
                                  'dist': 43},
                       'Suez': {'nodes': {'suez_south': {'lon_lat': (32.5164, 29.9159), 'xyz': (.0,) * 3},
                                          'suez_north': {'lon_lat': (32.3678, 31.2678), 'xyz': (.0,) * 3}},
                                'dist': 89},
                       'Dardanelles': {'nodes': {'dardanelles_south': {'lon_lat': (26.1406, 40.0136), 'xyz': (.0,) * 3},
                                                 'dardanelles_north': {'lon_lat': (26.9961, 40.4861), 'xyz': (.0,) * 3}},
                                       'dist': 52}}
        for canal in self.canals:
            for n in self.canals[canal]['nodes']:
                lon_r, lat_r = (radians(pos) for pos in self.canals[canal]['nodes'][n]['lon_lat'])
                self.canals[canal]['nodes'][n]['xyz'] = (cos(lat_r) * cos(lon_r), cos(lat_r) * sin(lon_r), sin(lat_r))

        self.sphere = None

    def add_vertex(self, p):
        """ Add vertex to mesh, fix position to be on unit sphere, return index """
        length = sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        x, y, z = p.x / length, p.y / length, p.z / length
        p3d = Point3D(x, y, z)
        self.graph.add_node(self.index, xyz=(x, y, z))
        self.graph.nodes[self.index]['point'] = p3d
        lat = degrees(atan2(z, sqrt(x * x + y * y)))
        lon = degrees(atan2(y, x))
        self.graph.nodes[self.index]['lon_lat'] = (lon, lat)
        self.points[self.index] = Point3D(x, y, z)
        self.index += 1
        return self.index

    def get_middle_point(self, idx1, idx2):
        """Return index of point in the middle of p1 and p2"""
        # First, check if point is already in cache
        if idx1 < idx2:
            small_idx, great_idx = idx1, idx2
        else:
            small_idx, great_idx = idx2, idx1
        key = (small_idx << 32) + great_idx
        if key in self.mp_cache:
            return self.mp_cache[key]

        # If not in cache: calculate the point
        p1 = self.graph.nodes[idx1]['point']
        p2 = self.graph.nodes[idx2]['point']
        middle = Point3D((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)

        # Add vertex makes sure point is on unit sphere. i - 1, since we need the current index
        idx = self.add_vertex(middle) - 1
        self.mp_cache[key] = idx
        return idx

    def node_x_polys(self, n):
        lon, lat = n['lon_lat'][0], n['lon_lat'][1]
        p_bounds = (lon, lat, lon, lat)

        # Returns the geometry indices of the minimum bounding rectangles that intersect the bounding box of node n
        mbr_intersections = list(self.rtree_idx.intersection(p_bounds))
        if mbr_intersections:  # Create LineString if there is at least one minimum bounding rectangle intersection
            shapely_point = Point(lon, lat)
            for poly_idx in mbr_intersections:
                if self.prep_polys[poly_idx].intersects(shapely_point):
                    return True
        return False

    def remove_nodes_x_polys(self):
        cnt, nr_edges = 0, len(self.graph.edges)
        sys.stdout.write("removing nodes from graph:  {}%".format(int(100 * cnt / nr_edges)))
        sys.stdout.flush()
        graph2 = self.graph.copy()
        for n_data in self.graph.nodes(data=True):
            # Printing progress
            cnt += 1
            if cnt % 100 == 0:
                sys.stdout.write("\rremoving nodes from graph:  {}%".format(int(100 * cnt / nr_edges)))
                sys.stdout.flush()

            # If node is in polygon, remove from graph
            if self.node_x_polys(n_data[1]):
                graph2.remove_node(n_data[0])
        sys.stdout.write("\rremoving nodes from graph: 100%\n")
        sys.stdout.flush()
        return graph2

    def remove_edges_x_polys(self):
        cnt, nr_edges = 0, len(self.graph.edges)
        sys.stdout.write("removing edges from graph:  {}%".format(int(100 * cnt / nr_edges)))
        sys.stdout.flush()
        graph_copy = self.graph.copy()
        for edge_data in self.graph.edges(data=True):
            # Printing progress
            cnt += 1
            if cnt % 100 == 0:
                sys.stdout.write("\rremoving edges from graph:  {}%".format(int(100 * cnt / nr_edges)))
                sys.stdout.flush()

            # Get indices and positions of nodes from current edge
            p1, p2 = edge_data[0], edge_data[1]
            pos1, pos2 = self.graph.nodes[p1]['lon_lat'], self.graph.nodes[p2]['lon_lat']

            # Skip border edges since they always intersect polygons in a 2D grid map
            if abs(pos1[0] - pos2[0]) > 340:
                continue

            # If edge intersects polygons, remove from graph
            if edge_x_geos(pos1, pos2, self.rtree_idx, self.prep_polys):
                graph_copy.remove_edge(p1, p2)
        sys.stdout.write("\rremoving edges from graph: 100%\n")
        sys.stdout.flush()
        return graph_copy

    def build_hexa_graph(self):
        print('Building Hexagraph')
        # Create 12 vertices of the icosahedron
        t = (1 + sqrt(5)) / 2

        self.add_vertex(Point3D(-1, t, 0))
        self.add_vertex(Point3D(1, t, 0))
        self.add_vertex(Point3D(-1, -t, 0))
        self.add_vertex(Point3D(1, -t, 0))

        self.add_vertex(Point3D(0, -1, t))
        self.add_vertex(Point3D(0, 1, t))
        self.add_vertex(Point3D(0, -1, -t))
        self.add_vertex(Point3D(0, 1, -t))

        self.add_vertex(Point3D(t, 0, -1))
        self.add_vertex(Point3D(t, 0, 1))
        self.add_vertex(Point3D(-t, 0, -1))
        self.add_vertex(Point3D(-t, 0, 1))

        # Create 20 triangles of the icosahedron
        triangles = [Triangle(0, 11, 5), Triangle(0, 5, 1), Triangle(0, 1, 7), Triangle(0, 7, 10), Triangle(0, 10, 11),
                     Triangle(1, 5, 9), Triangle(5, 11, 4), Triangle(11, 10, 2), Triangle(10, 7, 6), Triangle(7, 1, 8),
                     Triangle(3, 9, 4), Triangle(3, 4, 2), Triangle(3, 2, 6), Triangle(3, 6, 8), Triangle(3, 8, 9),
                     Triangle(4, 9, 5), Triangle(2, 4, 11), Triangle(6, 2, 10), Triangle(8, 6, 7), Triangle(9, 8, 1)]

        # Refine triangles
        for i in range(self.d):
            triangles2 = []
            for tri in triangles:
                # Replace triangle by 4 triangles
                a = self.get_middle_point(tri.v1, tri.v2)
                b = self.get_middle_point(tri.v2, tri.v3)
                c = self.get_middle_point(tri.v3, tri.v1)

                triangles2.append(Triangle(tri.v1, a, c))
                triangles2.append(Triangle(tri.v2, b, a))
                triangles2.append(Triangle(tri.v3, c, b))
                triangles2.append(Triangle(a, b, c))

            triangles = triangles2

        # Add triangles to mesh
        for tri in triangles:
            self.graph.add_edge(tri.v1, tri.v2)
            self.graph.add_edge(tri.v2, tri.v3)
            self.graph.add_edge(tri.v3, tri.v1)

        # Refine graph near shorelines

        # Get polygon exteriors and populate R-Tree
        shorelines = fiona.open('C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/{0}/GSHHS_{0}_L1.shp'.format(self.res))
        polys = [shape(poly_shape['geometry']) for poly_shape in iter(shorelines)]
        exteriors = [poly.exterior for poly in polys]
        prep_exteriors = [prep(exterior) for exterior in exteriors]
        exterior_rtree_idx = rtree.index.Index()
        for pos, exterior in enumerate(exteriors):
            exterior_rtree_idx.insert(pos, exterior.bounds)

        graph_copy = self.graph.copy()
        for edge in graph_copy.edges():
            # Get indices and long/lat positions of nodes from current edge
            n1, n2 = edge[0], edge[1]
            pos1, pos2 = self.graph.nodes[n1]['lon_lat'], self.graph.nodes[n2]['lon_lat']

            # Skip border edges since they always intersect polygons in a 2D grid map
            if abs(pos1[0] - pos2[0]) > 340:  # FIND ALTERNATIVE. Transform polygons to spherical coordinates?
                continue

            # If edge crosses a polygon exterior, refine two adjacent triangles
            if edge_x_geos(pos1, pos2, exterior_rtree_idx, prep_exteriors):
                # Get adjacent triangles of intersected edge
                n3s = [e1[1] for e1 in graph_copy.edges(n1) for e2 in graph_copy.edges(n2) if e1[1] == e2[1]]
                new_triangles = [(n1, n2, n3) for n3 in n3s if sorted((n1, n2, n3)) not in self.tri_cache]
                if new_triangles:
                    for tri in new_triangles:
                        self.tri_cache.append(sorted(tri))
                else:
                    continue

                for i in range(self.vd):
                    triangles = new_triangles
                    new_triangles = []
                    # Subdivide each triangle into four equal triangles
                    for tri in triangles:
                        # Find middle point of edge u,v and add node and edges to graph
                        n1, n2, n3 = tri[0], tri[1], tri[2]
                        a = self.get_middle_point(n1, n2)
                        # Subdivide edge into two equal edges
                        if self.graph.has_edge(n1, n2):
                            self.graph.remove_edge(n1, n2)
                        self.graph.add_edge(n1, a)
                        self.graph.add_edge(a, n2)

                        # Find middle point of edge v,w and add node and edges to graph
                        b = self.get_middle_point(n2, n3)
                        # Subdivide edge into two equal edges
                        if self.graph.has_edge(n2, n3):
                            self.graph.remove_edge(n2, n3)
                        self.graph.add_edge(n2, b)
                        self.graph.add_edge(b, n3)

                        # Find middle point of edge w,u and add node and edges to graph
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
                        new_triangles.extend([(n1, a, c), (a, n2, b), (a, b, c), (c, b, n3)])

        # Delete 3DPoint attribute since it is not needed in the future
        for n in self.graph.nodes:
            del self.graph.nodes[n]['point']

        # Remove nodes and edges crossing polygons
        self.graph = self.remove_nodes_x_polys()
        self.graph = self.remove_edges_x_polys()

        self.sphere = self.graph

        for canal in self.canals:
            for n in self.canals[canal]['nodes']:
                self.graph.add_node(n,
                                    lon_lat=self.canals[canal]['nodes'][n]['lon_lat'],
                                    xyz=self.canals[canal]['nodes'][n]['xyz'])

        # Link each canal node to its three nearest nodes
        for canal in self.canals:
            n1, n2 = (n for n in self.canals[canal]['nodes'])
            xyz1, xyz2 = (self.canals[canal]['nodes'][n]['xyz'] for n in self.canals[canal]['nodes'])
            self.graph.add_edge(n1, n2)
            dist1 = {n: arc_length(xyz1, xyz) for n, xyz in nx.get_node_attributes(self.graph, 'xyz').items()
                     if n is not n1}
            dist2 = {n: arc_length(xyz2, xyz) for n, xyz in nx.get_node_attributes(self.graph, 'xyz').items()
                     if n is not n2}

            # Add edges from canal nodes to three nearest neighbours
            for nn in nsmallest(3, dist1, key=dist1.get):
                self.graph.add_edge(n1, nn, miles=dist1[nn])
            for nn in nsmallest(3, dist2, key=dist2.get):
                self.graph.add_edge(n2, nn, miles=dist2[nn])

        # Set edge weights to arc length (nautical miles)
        for e in self.graph.edges():
            xyz1 = self.graph.nodes[e[0]]['xyz']
            xyz2 = self.graph.nodes[e[1]]['xyz']
            self.graph[e[0]][e[1]]['miles'] = arc_length(xyz1, xyz2)

        # Set canal edge weights to predefined weights
        for canal in self.canals:
            n1, n2 = (n for n in self.canals[canal]['nodes'])
            self.graph[n1][n2]['miles'] = self.canals[canal]['dist']

        # Remove all connected components but the largest
        for cc in sorted(nx.connected_components(self.graph), key=len, reverse=True)[1:]:
            self.graph.remove_nodes_from(cc)

        return self.graph

    def plot_sphere(self):
        # 3D plot of unit sphere nodes
        fig = plt.figure()
        ax = Axes3D(fig)
        xyz = nx.get_node_attributes(self.sphere, 'xyz')
        xyz = np.array([value for value in xyz.values()])
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, color='black')

    def plot_sphere_edges(self):
        # 3D plot of unit sphere edges
        fig = plt.figure()
        ax = Axes3D(fig)
        xyz = nx.get_node_attributes(self.sphere, 'xyz')  # Plots nodes for which attribute 'pt_sphere' exists
        cnt = 0
        for edge in self.sphere.edges:
            cnt += 1
            print('plotting sphere edges', round(cnt / len(self.sphere.edges), 2))
            u, v = xyz[edge[0]], xyz[edge[1]]
            x = np.array([u[0], v[0]])
            y = np.array([u[1], v[1]])
            z = np.array([u[2], v[2]])
            ax.plot(x, y, z, color='black', linewidth=1)


# Distance function
def arc_length(xyz1, xyz2):
    return 3440 * acos(sum(p1 * p2 for p1, p2 in zip(xyz1, xyz2)))


class Hexagraph:
    def __init__(self,
                 d,
                 vd,
                 res,
                 prep_polys,
                 rtree_idx):
        self.d = d
        self.vd = vd
        self.res = res
        self.prep_polys = prep_polys
        self.rtree_idx = rtree_idx

        # Get Graph
        graph_dir = "output/variable_density_geodesic_grids/"
        try:
            self.graph = nx.read_gpickle(graph_dir + "res_{}_d{}_vd{}.gpickle".format(self.res, self.d, self.vd))
            print('Loaded graph from: ' + graph_dir + "res_{}_d{}_vd{}.gpickle".format(self.res, self.d, self.vd))
        except FileNotFoundError:
            # Initialize "HexaGraphBuilder"
            hexa_graph_builder = HexagraphBuilder(self.d, self.vd, self.res, self.prep_polys, self.rtree_idx)
            self.graph = hexa_graph_builder.build_hexa_graph()

            # Save graph to file
            nx.write_gpickle(self.graph, graph_dir + "res_{}_d{}_vd{}.gpickle".format(self.res, self.d, self.vd))
            print('Built and saved graph to: ' + graph_dir + "res_{}_d{}_vd{}.gpickle".format(self.res, self.d, self.vd))

    def get_graph(self):
        return self.graph

    def plot_graph(self):
        pos = nx.get_node_attributes(self.graph, 'lon_lat')
        fig, ax = plt.subplots()
        nx.draw(self.graph, pos=pos, node_size=2, ax=ax)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


if __name__ == '__main__':
    resolution = 'c'
    max_poly_size = 4
    graph_d = 5
    graph_vd = 4

    # Get split polys and prepare
    try:
        with open('output/split_polygons/res_{0}_treshold_{1}'.format(resolution, max_poly_size),
                  'rb') as f:
            split_polys = pickle.load(f)
    except FileNotFoundError:
        split_polys = get_split_polygons(resolution, max_poly_size)
    _prep_polys = [prep(poly) for poly in split_polys]

    # Populate R-tree index with bounds of polygons
    poly_rtree_idx = rtree.index.Index()
    for _poly_idx, split_poy in enumerate(split_polys):
        poly_rtree_idx.insert(_poly_idx, split_poy.bounds)

    # Initialize "HexagraphBuilder"
    hexagraph = Hexagraph(graph_d, graph_vd, resolution, _prep_polys, poly_rtree_idx)
    hexagraph.plot_graph()

    plt.show()
