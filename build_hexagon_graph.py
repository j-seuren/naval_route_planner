# http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import fiona
import pickle
from mpl_toolkits.mplot3d import Axes3D
from math import atan2, degrees, acos, sqrt, radians, cos, sin
from shapely.geometry import shape, Point, LineString
from shapely.prepared import prep
from rtree import index
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


class IcoSphere:
    def __init__(self):
        self.graph = nx.Graph()
        self.index = 0
        self.middle_point_cache = {}
        self.points = {}
        self.tri_cache = []

    # Add vertex to mesh, fix position to be on unit sphere, return index
    def add_vertex(self, p):
        length = sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        x, y, z = p.x / length, p.y / length, p.z / length
        point_3d = Point3D(x, y, z)
        self.graph.add_node(self.index, xyz=(x, y, z))
        self.graph.nodes[self.index]['point'] = point_3d
        lat = degrees(atan2(z, sqrt(x * x + y * y)))
        lon = degrees(atan2(y, x))
        self.graph.nodes[self.index]['lon_lat'] = (lon, lat)
        self.points[self.index] = Point3D(x, y, z)
        self.index += 1
        return self.index

    # Return index of point in the middle of p1 and p2
    def get_middle_point(self, i1, i2):
        # First, check if point is already in cache
        if i1 < i2:
            smaller_index, greater_index = i1, i2
        else:
            smaller_index, greater_index = i2, i1
        key = (smaller_index << 32) + greater_index
        if key in self.middle_point_cache:
            return self.middle_point_cache[key]

        # If not in cache: calculate the point
        u = self.graph.nodes[i1]['point']
        v = self.graph.nodes[i2]['point']
        middle = Point3D((u.x + v.x) / 2, (u.y + v.y) / 2, (u.z + v.z) / 2)

        # Add vertex makes sure point is on unit sphere. i - 1, since we need the current index
        i = self.add_vertex(middle) - 1
        self.middle_point_cache[key] = i
        return i

    def refine_shoreline_intersections(self, idx, polygons, local_mesh_recursion):
        graph = self.graph.copy()
        for edge in graph.edges():
            # Get indices and long/lat positions of nodes from current edge
            u, v = edge[0], edge[1]
            u_pos, v_pos = self.graph.nodes[u]['lon_lat'], self.graph.nodes[v]['lon_lat']

            # Skip border edges since they always intersect polygons in a 2D grid map
            if abs(u_pos[0] - v_pos[0]) > 340:  # FIND ALTERNATIVE. Transform polygons to spherical coordinates?
                continue

            # If edge crosses a polygon exterior, refine two adjacent triangles
            if edge_x_geometry(u_pos, v_pos, idx, polygons):
                # Get adjacent triangles of intersected edge
                ww = [e1[1] for e1 in graph.edges(u) for e2 in graph.edges(v) if e1[1] == e2[1]]
                new_triangles = [(u, v, w) for w in ww if sorted((u, v, w)) not in self.tri_cache]
                if new_triangles:
                    for tri in new_triangles:
                        self.tri_cache.append(sorted(tri))
                else:
                    continue

                for i in range(local_mesh_recursion):
                    triangles = new_triangles
                    new_triangles = []
                    # Subdivide each triangle into four equal triangles
                    for tri in triangles:
                        # Find middle point of edge u,v and add node and edges to graph
                        u, v, w = tri[0], tri[1], tri[2]
                        a = self.get_middle_point(u, v)
                        # Subdivide edge into two equal edges
                        if self.graph.has_edge(u, v):
                            self.graph.remove_edge(u, v)
                        self.graph.add_edge(u, a)
                        self.graph.add_edge(a, v)

                        # Find middle point of edge v,w and add node and edges to graph
                        b = self.get_middle_point(v, w)
                        # Subdivide edge into two equal edges
                        if self.graph.has_edge(v, w):
                            self.graph.remove_edge(v, w)
                        self.graph.add_edge(v, b)
                        self.graph.add_edge(b, w)

                        # Find middle point of edge w,u and add node and edges to graph
                        c = self.get_middle_point(w, u)
                        # Subdivide edge into two equal edges
                        if self.graph.has_edge(w, u):
                            self.graph.remove_edge(w, u)
                        self.graph.add_edge(w, c)
                        self.graph.add_edge(c, u)

                        # Add inner edges of subdivided triangle
                        self.graph.add_edge(a, b)
                        self.graph.add_edge(b, c)
                        self.graph.add_edge(c, a)
                        new_triangles.extend([(u, a, c), (a, v, b), (a, b, c), (c, b, w)])

    def create(self, global_mesh_recursion, local_mesh_recursion, idx, polygons):
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
        faces = [Triangle(0, 11, 5), Triangle(0, 5, 1), Triangle(0, 1, 7), Triangle(0, 7, 10), Triangle(0, 10, 11),
                 Triangle(1, 5, 9), Triangle(5, 11, 4), Triangle(11, 10, 2), Triangle(10, 7, 6), Triangle(7, 1, 8),
                 Triangle(3, 9, 4), Triangle(3, 4, 2), Triangle(3, 2, 6), Triangle(3, 6, 8), Triangle(3, 8, 9),
                 Triangle(4, 9, 5), Triangle(2, 4, 11), Triangle(6, 2, 10), Triangle(8, 6, 7), Triangle(9, 8, 1)]

        # Refine triangles
        for i in range(global_mesh_recursion):
            faces2 = []
            for tri in faces:
                # Replace triangle by 4 triangles
                a = self.get_middle_point(tri.v1, tri.v2)
                b = self.get_middle_point(tri.v2, tri.v3)
                c = self.get_middle_point(tri.v3, tri.v1)

                faces2.append(Triangle(tri.v1, a, c))
                faces2.append(Triangle(tri.v2, b, a))
                faces2.append(Triangle(tri.v3, c, b))
                faces2.append(Triangle(a, b, c))

            faces = faces2

        # Add triangles to mesh
        for tri in faces:
            self.graph.add_edge(tri.v1, tri.v2)
            self.graph.add_edge(tri.v2, tri.v3)
            self.graph.add_edge(tri.v3, tri.v1)

        # Refine graph near shorelines
        self.refine_shoreline_intersections(idx, polygons, local_mesh_recursion)

        # Delete 3DPoint attribute since it is not needed anymore
        for node in self.graph.nodes:
            del self.graph.nodes[node]['point']

        return self.graph


# Distance function
def arc_length(xyz1, xyz2):
    return 3440 * acos(sum(p * q for p, q in zip(xyz1, xyz2)))


def node_x_geometry(n, rtree_idx, geometries):
    node_bounds = (n['lon_lat'][0], n['lon_lat'][1],
                   n['lon_lat'][0], n['lon_lat'][1])

    # Returns the geometry indices of the minimum bounding rectangles that intersect the bounding box of node n
    mbr_intersections = rtree_idx.intersection(node_bounds)
    if mbr_intersections:  # Create LineString if there is at least one minimum bounding rectangle intersection
        shapely_point = Point(n['lon_lat'][0], n['lon_lat'][1])
        for i in mbr_intersections:
            if geometries[i].intersects(shapely_point):
                return True
    return False


def edge_x_geometry(u, v, rtree_idx, geometries):
    line_bounds = (min(u[0], v[0]), min(u[1], v[1]),
                   max(u[0], v[0]), max(u[1], v[1]))

    # Returns the geometry indices of the minimum bounding rectangles that intersect the bounding box of line u,v
    mbr_intersections = rtree_idx.intersection(line_bounds)
    if mbr_intersections:  # Create LineString if there is at least one minimum bounding rectangle intersection
        shapely_line = LineString([u, v])
        for i in mbr_intersections:
            if geometries[i].intersects(shapely_line):
                return True
    return False


def remove_nodes_x_polygons(graph, idx, polygons):
    cnt = 0
    nr_nodes = len(graph.nodes)
    graph2 = graph.copy
    for node_data in graph.nodes(data=True):
        # Printing progress
        cnt += 1
        if cnt % 100 == 0:
            print('removing nodes:', round(cnt / nr_nodes, 2))

        # If node is in polygon, remove from graph
        if node_x_geometry(node_data[1], idx, polygons):
            graph2.remove_node(node_data[0])
    return graph2


def remove_edges_x_polygons(graph, idx, polygons):
    cnt = 0
    nr_edges = len(graph.edges)
    graph2 = graph.copy
    for edge_data in graph.edges(data=True):
        # Printing progress
        cnt += 1
        if cnt % 100 == 0:
            print('removing edges:', round(cnt / nr_edges, 2))

        # Get indices and positions of nodes from current edge
        u, v = edge_data[0], edge_data[1]
        u_pos, v_pos = graph.nodes[u]['lon_lat'], graph.nodes[v]['lon_lat']

        # Skip border edges since they always intersect polygons in a 2D grid map
        if abs(u_pos[0] - v_pos[0]) > 340:
            continue

        # If edge intersects polygon exterior, remove from graph
        if edge_x_geometry(u_pos, v_pos, idx, polygons):
            graph2.remove_edge(u, v)
    return graph2


def plot_sphere(sphere):
    # 3D plot of unit sphere nodes
    fig = plt.figure()
    ax = Axes3D(fig)
    xyz = nx.get_node_attributes(sphere, 'xyz')
    xyz = np.array([value for value in xyz.values()])
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, color='black')


def plot_sphere_edges(sphere):
    # 3D plot of unit sphere edges
    fig = plt.figure()
    ax = Axes3D(fig)
    xyz = nx.get_node_attributes(sphere, 'xyz')  # Plots nodes for which attribute 'pt_sphere' exists
    cnt = 0
    for edge in sphere.edges:
        cnt += 1
        print('plotting sphere edges', round(cnt / len(sphere.edges), 2))
        u, v = xyz[edge[0]], xyz[edge[1]]
        x = np.array([u[0], v[0]])
        y = np.array([u[1], v[1]])
        z = np.array([u[2], v[2]])
        ax.plot(x, y, z, color='black', linewidth=1)


def plot_grid(graph):
    lon_lat = nx.get_node_attributes(graph, 'lon_lat')
    fig, ax = plt.subplots()
    nx.draw(graph, pos=lon_lat, node_size=5, ax=ax)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


def get_graph(res, d, vd, split_polygons, polygon_rtree_idx):
    try:
        G = nx.read_gpickle("output/variable_density_geodesic_grids/res_{}_d{}_vd{}.gpickle".format(res, d, vd))
    except FileNotFoundError:
        coastlines = fiona.open('C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/{0}/GSHHS_{0}_L1.shp'.format(res))
        polygons = [shape(polygon['geometry']) for polygon in iter(coastlines)]
        exteriors = [polygon.exterior for polygon in polygons]
        prepared_exteriors = [prep(exterior) for exterior in exteriors]

        # Populate R-tree index with bounds of polygon exteriors
        exterior_rtree_idx = index.Index()
        for pos, exterior in enumerate(exteriors):
            exterior_rtree_idx.insert(pos, exterior.bounds)

        # Get refined graph
        G = IcoSphere().create(d, vd, exterior_rtree_idx, prepared_exteriors)

        # Postprocessing graph
        G = remove_nodes_x_polygons(G, polygon_rtree_idx, split_polygons)
        G = remove_edges_x_polygons(G, polygon_rtree_idx, split_polygons)

        # Create canal nodes
        G.add_node('panama_south', lon_lat=(-79.540932, 8.894197))
        G.add_node('panama_north', lon_lat=(-79.919005, 9.391057))
        G.add_node('suez_south', lon_lat=(32.5164, 29.9159))
        G.add_node('suez_north', lon_lat=(32.3678, 31.2678))
        G.add_node('dardanelles_south', lon_lat=(26.1406, 40.0136))
        G.add_node('dardanelles_north', lon_lat=(26.9961, 40.4861))

        canals = {'Panama': ('panama_south', 'panama_north'),
                  'Suez': ('suez_south', 'suez_north'),
                  'Dardanelles': ('dardanelles_south', 'dardanelles_north')}

        # Link each canal node to its three nearest nodes
        for u, v in canals.values():
            G.add_edge(u, v)

            lon_u, lat_u, lon_v, lat_v = radians(G.nodes[u]['lon_lat'][0]), radians(G.nodes[u]['lon_lat'][1]), \
                                         radians(G.nodes[v]['lon_lat'][0]), radians(G.nodes[v]['lon_lat'][1])
            xyz_u = (cos(lat_u) * cos(lon_u), cos(lat_u) * sin(lon_u), sin(lat_u))
            xyz_v = (cos(lat_v) * cos(lon_v), cos(lat_v) * sin(lon_v), sin(lat_v))
            distances_to_u = {n: arc_length(xyz_u, xyz) for n, xyz in nx.get_node_attributes(G, 'xyz').items()}
            distances_to_v = {n: arc_length(xyz_v, xyz) for n, xyz in nx.get_node_attributes(G, 'xyz').items()}
            G.nodes[u]['xyz'] = xyz_u
            G.nodes[v]['xyz'] = xyz_v

            # Add three shortest edges to start and end point
            for node in nsmallest(3, distances_to_u, key=distances_to_u.get):
                G.add_edge(u, node, miles=distances_to_u[node])
            for node in nsmallest(3, distances_to_v, key=distances_to_v.get):
                G.add_edge(v, node, miles=distances_to_v[node])

        # Set weights to great circle distance (nautical miles)
        canal_distances = {'panama_south': 43,
                           'suez_south': 89,
                           'dardanelles_south': 52}
        for e in G.edges():
            if e[0] in canal_distances:
                G[e[0]][e[1]]['miles'] = canal_distances[e[0]]
            elif e[1] in canal_distances:
                G[e[0]][e[1]]['miles'] = canal_distances[e[1]]
            else:
                xyz0 = G.nodes[e[0]]['xyz']
                xyz1 = G.nodes[e[1]]['xyz']

                G[e[0]][e[1]]['miles'] = arc_length(xyz0, xyz1)

        # Get connected components sorted by descending length
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)

        # Remove all connected components but the largest
        for component in Gcc[1:]:
            G.remove_nodes_from(component)

        # Save graph to file
        nx.write_gpickle(G, "output/variable_density_geodesic_grids/res_{}_d{}_vd{}.gpickle".format(res, d, vd))

    return G


if __name__ == '__main__':
    resolution = 'c'
    max_poly_size = 4
    # Import polygons and its exteriors
    shorelines = fiona.open('C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/{0}/GSHHS_{0}_L1.shp'.format(resolution))
    polygon_list = [shape(polygon['geometry']) for polygon in iter(shorelines)]
    exterior_list = [polygon.exterior for polygon in polygon_list]

    # Split and repare polygons
    try:
        with open('output/split_polygons/res_{0}_treshold_{1}'.format(resolution, max_poly_size),
                  'rb') as f:
            split_polys = pickle.load(f)
    except FileNotFoundError:
        split_polys = get_split_polygons(resolution, max_poly_size)

    prepared_polygons = [prep(polygon) for polygon in polygon_list]

    # Populate R-tree index with bounds of polygons
    poly_rtree_idx = index.Index()
    for pos, polygon in enumerate(polygon_list):
        poly_rtree_idx.insert(pos, polygon.bounds)
    G = get_graph(resolution, 6, 4, polygon_list, prepared_polygons, poly_rtree_idx)

    plot_grid(G)
    plt.show()
