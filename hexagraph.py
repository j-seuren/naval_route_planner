# http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
import fiona
import katana
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import rtree
import sys

from evaluation import edge_x_geos
from mpl_toolkits.mplot3d import Axes3D
from math import atan2, degrees, acos, sqrt, radians, cos, sin
from shapely.geometry import shape
from shapely.prepared import prep
from heapq import nsmallest


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


class HexagraphBuilder:
    def __init__(self,
                 dens,
                 var_dens,
                 res,
                 prep_polys,
                 rtree_idx,
                 ecas,
                 rtree_idx_eca):
        self.dens = dens
        self.var_dens = var_dens
        self.res = res
        self.prep_polys = prep_polys
        self.rtree_idx = rtree_idx
        self.rtree_idx_eca = rtree_idx_eca
        self.ecas = ecas
        self.G = nx.Graph()
        self.index = 0
        self.mp_cache = {}
        self.points = {}
        self.tri_cache = []
        self.canals = {'Panama': {'nodes': {'panama_south': {'deg': (-79.540932, 8.894197),
                                                             'xyz': (.0,) * 3},
                                            'panama_north': {'deg': (-79.919005, 9.391057),
                                                             'xyz': (.0,) * 3}}, 'dist': 43},
                       'Suez': {'nodes': {'suez_south': {'deg': (32.5164, 29.9159),
                                                         'xyz': (.0,) * 3},
                                          'suez_north': {'deg': (32.3678, 31.2678),
                                                         'xyz': (.0,) * 3}}, 'dist': 89},
                       'Dardanelles': {'nodes': {'dardanelles_south': {'deg': (26.1406, 40.0136),
                                                                       'xyz': (.0,) * 3},
                                                 'dardanelles_north': {'deg': (26.9961, 40.4861),
                                                                       'xyz': (.0,) * 3}}, 'dist': 52}}
        for c in self.canals:
            for n in self.canals[c]['nodes']:
                lo, la = (radians(deg) for deg in
                          self.canals[c]['nodes'][n]['deg'])
                self.canals[c]['nodes'][n]['xyz'] = (
                    cos(la) * cos(lo), cos(la) * sin(lo), sin(la))

    def add_vertex(self, p):
        """ Add vertex to mesh, fix position to be on unit sphere, return
        index """
        length = sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        x, y, z = p.x / length, p.y / length, p.z / length
        p3d = Point3D(x, y, z)
        self.G.add_node(self.index, xyz=(x, y, z))
        self.G.nodes[self.index]['point'] = p3d
        lon = degrees(atan2(y, x))
        lat = degrees(atan2(z, sqrt(x * x + y * y)))
        self.G.nodes[self.index]['deg'] = (lon, lat)
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
        p1 = self.G.nodes[idx1]['point']
        p2 = self.G.nodes[idx2]['point']
        middle = Point3D((p1.x + p2.x) / 2,
                         (p1.y + p2.y) / 2,
                         (p1.z + p2.z) / 2)

        # Add vertex makes sure point is on unit sphere. i - 1,
        # since we need the current index
        idx = self.add_vertex(middle) - 1
        self.mp_cache[key] = idx
        return idx

    # def remove_nodes_x_polys(self):
    #     cnt, nr_edges = 0, len(self.G.edges)
    #     sys.stdout.write(
    #         "removing nodes from graph:  {}%".format(int(100 * cnt / nr_edges)))
    #     sys.stdout.flush()
    #     graph2 = self.G.copy()
    #     for n_data in self.G.nodes(data=True):
    #         # Printing progress
    #         cnt += 1
    #         if cnt % 100 == 0:
    #             sys.stdout.write("\rremoving nodes from graph:  {}%".format(
    #                 int(100 * cnt / nr_edges)))
    #             sys.stdout.flush()
    #
    #         # If node is in polygon, remove from graph
    #         p_n = n_data[1]['deg']
    #         if edge_x_geos(p_n, p_n, self.rtree_idx, self.prep_polys):
    #             graph2.remove_node(n_data[0])
    #     sys.stdout.write("\rremoving nodes from graph: 100%\n")
    #     sys.stdout.flush()
    #     return graph2

    def remove_edges_x_polys(self):
        cnt, nr_edges = 0, len(self.G.edges)
        sys.stdout.write(
            "removing edges from graph:  {}%".format(int(100 * cnt / nr_edges)))
        sys.stdout.flush()
        graph_copy = self.G.copy()
        for edge_data in self.G.edges(data=True):
            # Printing progress
            cnt += 1
            if cnt % 100 == 0:
                sys.stdout.write("\rremoving edges from graph:  {}%".format(
                    int(100 * cnt / nr_edges)))
                sys.stdout.flush()

            # Get indices and positions of nodes from current edge
            p1, p2 = edge_data[0], edge_data[1]
            deg1, deg2 = self.G.nodes[p1]['deg'], self.G.nodes[p2]['deg']

            # Skip border edges
            # since they always intersect polygons in a 2D grid map
            if abs(deg1[0] - deg2[0]) > 340:
                continue

            # If edge intersects polygons, remove from graph
            if edge_x_geos(deg1, deg2, self.rtree_idx, self.prep_polys):
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
        tris = [Triangle(0, 11, 5), Triangle(0, 5, 1),
                Triangle(0, 1, 7), Triangle(0, 7, 10),
                Triangle(0, 10, 11), Triangle(1, 5, 9),
                Triangle(5, 11, 4), Triangle(11, 10, 2),
                Triangle(10, 7, 6), Triangle(7, 1, 8),
                Triangle(3, 9, 4), Triangle(3, 4, 2),
                Triangle(3, 2, 6), Triangle(3, 6, 8),
                Triangle(3, 8, 9), Triangle(4, 9, 5),
                Triangle(2, 4, 11), Triangle(6, 2, 10),
                Triangle(8, 6, 7), Triangle(9, 8, 1)]

        # Refine triangles
        for i in range(self.dens):
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

        # Add triangles to mesh
        for tri in tris:
            self.G.add_edge(tri.v1, tri.v2)
            self.G.add_edge(tri.v2, tri.v3)
            self.G.add_edge(tri.v3, tri.v1)

        # Refine graph near shorelines
        # Get polygon exteriors and populate R-tree
        gshhg_dir = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp'
        gshhg_fp = '{0}/GSHHS_{0}_L1.shp'.format(self.res)
        gshhg = fiona.open(os.path.join(gshhg_dir, gshhg_fp))
        polys = [shape(poly_shape['geometry']) for poly_shape in iter(gshhg)]
        exteriors = [poly.exterior for poly in polys]
        prep_exteriors = [prep(exterior) for exterior in exteriors]
        exterior_rtree_idx = rtree.index.Index()
        for pos, exterior in enumerate(exteriors):
            exterior_rtree_idx.insert(pos, exterior.bounds)

        G_copy = self.G.copy()
        for edge in G_copy.edges():
            # Get indices and long/lat positions of nodes from current edge
            n1, n2 = edge[0], edge[1]
            deg1, deg2 = self.G.nodes[n1]['deg'], self.G.nodes[n2]['deg']

            # Skip border edges
            # since they always intersect polygons in a 2D grid map
            if abs(deg1[0] - deg2[0]) > 340:
                continue

            # If edge crosses a polygon exterior, refine two adjacent triangles
            if edge_x_geos(deg1, deg2, exterior_rtree_idx, prep_exteriors):
                # Get adjacent triangles of intersected edge
                n3s = [e1[1] for e1 in G_copy.edges(n1) for e2 in
                       G_copy.edges(n2) if e1[1] == e2[1]]
                new_tris = [(n1, n2, n3) for n3 in n3s if
                            sorted((n1, n2, n3)) not in self.tri_cache]
                if new_tris:
                    for tri in new_tris:
                        self.tri_cache.append(sorted(tri))
                else:
                    continue

                for i in range(self.var_dens):
                    tris = new_tris
                    new_tris = []
                    # Subdivide each triangle into four equal triangles
                    for tri in tris:
                        # Find middle point of edge u,v
                        # and add node and edges to graph
                        n1, n2, n3 = tri
                        a = self.get_middle_point(n1, n2)
                        # Subdivide edge into two equal edges
                        if self.G.has_edge(n1, n2):
                            self.G.remove_edge(n1, n2)
                        self.G.add_edge(n1, a)
                        self.G.add_edge(a, n2)

                        # Find middle point of edge v,w
                        # and add node and edges to graph
                        b = self.get_middle_point(n2, n3)
                        # Subdivide edge into two equal edges
                        if self.G.has_edge(n2, n3):
                            self.G.remove_edge(n2, n3)
                        self.G.add_edge(n2, b)
                        self.G.add_edge(b, n3)

                        # Find middle point of edge w,u
                        # and add node and edges to graph
                        c = self.get_middle_point(n3, n1)
                        # Subdivide edge into two equal edges
                        if self.G.has_edge(n3, n1):
                            self.G.remove_edge(n3, n1)
                        self.G.add_edge(n3, c)
                        self.G.add_edge(c, n1)

                        # Add inner edges of subdivided triangle
                        self.G.add_edge(a, b)
                        self.G.add_edge(b, c)
                        self.G.add_edge(c, a)
                        new_tris.extend([(n1, a, c),
                                         (a, n2, b),
                                         (a, b, c),
                                         (c, b, n3)])

        # Delete 3DPoint attribute since it is not needed in the future
        for n in self.G.nodes:
            del self.G.nodes[n]['point']

        # Remove edges crossing polygons
        # self.G = self.remove_nodes_x_polys()
        self.G = self.remove_edges_x_polys()

        for canal in self.canals:
            for n in self.canals[canal]['nodes']:
                self.G.add_node(n,
                                deg=self.canals[canal]['nodes'][n]['deg'],
                                xyz=self.canals[canal]['nodes'][n]['xyz'])

        # Link each canal node to its three nearest nodes
        for canal in self.canals:
            n1, n2 = (n for n in self.canals[canal]['nodes'])
            xyz1, xyz2 = (self.canals[canal]['nodes'][n]['xyz']
                          for n in self.canals[canal]['nodes'])
            self.G.add_edge(n1, n2)
            dist1 = {n: arc_length(xyz1, xyz)
                     for n, xyz in nx.get_node_attributes(self.G, 'xyz').items()
                     if n is not n1}
            dist2 = {n: arc_length(xyz2, xyz)
                     for n, xyz in nx.get_node_attributes(self.G, 'xyz').items()
                     if n is not n2}

            # Add edges from canal nodes to three nearest neighbours
            for nn in nsmallest(3, dist1, key=dist1.get):
                self.G.add_edge(n1, nn, miles=dist1[nn])
            for nn in nsmallest(3, dist2, key=dist2.get):
                self.G.add_edge(n2, nn, miles=dist2[nn])

        # Set edge attributes
        for n1, n2 in self.G.edges():
            xyz1 = self.G.nodes[n1]['xyz']
            xyz2 = self.G.nodes[n2]['xyz']
            miles = arc_length(xyz1, xyz2)
            self.G[n1][n2]['miles'] = miles

            p1, p2 = self.G.nodes[n1]['deg'], self.G.nodes[n2]['deg']
            if edge_x_geos(p1, p2, self.rtree_idx_eca, self.ecas):
                self.G[n1][n2]['eca_weight'] = miles * 2
            else:
                self.G[n1][n2]['eca_weight'] = miles

        # Set canal edge weights to predefined weights
        for canal in self.canals:
            n1, n2 = (n for n in self.canals[canal]['nodes'])
            self.G[n1][n2]['miles'] = self.canals[canal]['dist']

        print('Removing isolates... ', end='')
        nx.isolates(self.G)
        self.G.remove_nodes_from(list(nx.isolates(self.G)))
        print('done')

        # Remove all connected components but the largest
        print('Removing isolate connected components... ', end='')
        for cc in sorted(nx.connected_components(self.G),
                         key=len, reverse=True)[1:]:
            self.G.remove_nodes_from(cc)
        print('done')

        return self.G

    def plot_sphere(self):
        # 3D plot of unit sphere nodes
        fig = plt.figure()
        ax = Axes3D(fig)
        xyz = nx.get_node_attributes(self.G, 'xyz')
        xyz = np.array([val for val in xyz.values()])
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, color='black')

    def plot_sphere_edges(self):
        # 3D plot of unit sphere edges
        fig = plt.figure()
        ax = Axes3D(fig)
        xyz = nx.get_node_attributes(self.G, 'xyz')
        cnt = 0
        for edge in self.G.edges:
            cnt += 1
            print('plotting sphere edges', round(cnt / len(self.G.edges), 2))
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
                 rtree_idx,
                 ecas,
                 rtree_idx_eca):
        self.d = d
        self.vd = vd
        self.res = res
        self.prep_polys = prep_polys
        self.rtree_idx = rtree_idx
        self.ecas = ecas
        self.rtree_idx_eca = rtree_idx_eca

        # Get Graph
        G_dir = "output/variable_density_geodesic_grids/"
        G_fn = "res_{}_d{}_vd{}.gpickle".format(self.res, self.d, self.vd)
        G_fp = os.path.join(G_dir, G_fn)
        try:
            self.graph = nx.read_gpickle(G_fp)
            print('Loaded graph from: ', G_fp)
        except FileNotFoundError:
            # Initialize "HexaGraphBuilder"
            builder = HexagraphBuilder(self.d, self.vd, self.res,
                                       self.prep_polys, self.rtree_idx,
                                       self.ecas, self.rtree_idx_eca)
            self.graph = builder.build_hexa_graph()

            # Save graph to file
            nx.write_gpickle(self.graph, G_fp)
            print('Built and saved graph to: ', G_fp)

    def get_g(self):
        return self.graph

    def plot_graph(self):
        pos = nx.get_node_attributes(self.graph, 'deg')
        fig, ax = plt.subplots()
        nx.draw(self.graph, pos=pos, node_size=2, ax=ax)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


if __name__ == '__main__':
    resolution = 'c'
    max_poly_size = 4
    graph_d = 5
    graph_vd = 4

    split_polys_dir = 'output/split_polygons/'
    split_polys_fn = 'res_{0}_threshold_{1}'.format(resolution, max_poly_size)
    split_polys_fp = os.path.join(split_polys_dir, split_polys_fn)

    # Get split polys and prepare
    try:
        with open(split_polys_fp, 'rb') as f:
            split_polys = pickle.load(f)
    except FileNotFoundError:
        split_polys = katana.get_split_polygons(resolution, max_poly_size)
    _prep_polys = [prep(poly) for poly in split_polys]

    # Populate R-tree index with bounds of polygons
    _rtree_idx = rtree.index.Index()
    for _poly_idx, split_poy in enumerate(split_polys):
        _rtree_idx.insert(_poly_idx, split_poy.bounds)



    # Initialize "HexagraphBuilder"
    hexagraph = Hexagraph(graph_d, graph_vd, resolution,
                          _prep_polys, _rtree_idx,
                          _ecas, _rtree_idx_eca)
    hexagraph.plot_graph()

    plt.show()
