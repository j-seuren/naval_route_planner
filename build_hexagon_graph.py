# http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import fiona
from mpl_toolkits.mplot3d import Axes3D
from math import atan2, degrees, acos, sqrt
from shapely.geometry import shape, Point, LineString
from shapely.strtree import STRtree
from copy import deepcopy
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


class IcoSphere:
    def __init__(self):
        self.graph = nx.Graph()
        self.index = 0
        self.middle_point_cache = {}
        self.points = {}

    # Add vertex to mesh, fix position to be on unit sphere, return index
    def add_vertex(self, p):
        length = sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        x, y, z = p.x/length, p.y/length, p.z/length
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
        middle = Point3D((u.x + v.x)/2, (u.y + v.y)/2, (u.z + v.z)/2)

        # Add vertex makes sure point is on unit sphere. i - 1, since we need the current index
        i = self.add_vertex(middle) - 1
        self.middle_point_cache[key] = i
        return i

    def refine_shoreline_intersections(self, polygons, local_mesh_recursion):
        for i in range(local_mesh_recursion):
            cnt = 0
            nr_edges = len(self.graph.edges)
            graph = deepcopy(self.graph)
            for edge in graph.edges():
                # Print progress
                cnt += 1
                if cnt % 100 == 0:
                    print('refining. Level:', i + 1, local_mesh_recursion, 'edge', round(cnt/nr_edges, 2))

                # Get indices and long/lat positions of nodes from current edge
                u, v = edge[0], edge[1]
                u_pos, v_pos = self.graph.nodes[u]['lon_lat'], self.graph.nodes[v]['lon_lat']

                # Skip border edges since they always intersect polygons in a 2D grid map
                if abs(u_pos[0] - v_pos[0]) > 340:  # FIND ALTERNATIVE. Transform polygons to spherical coordinates?
                    continue

                # If edge crosses a polygon exterior, refine two adjacent triangles
                if edge_x_geometry(u_pos, v_pos, polygons):
                    # Get adjacent triangles of intersected edge
                    ww = [e1[1] for e1 in self.graph.edges(u) for e2 in self.graph.edges(v) if e1[1] == e2[1]]
                    triangles = [(u, v, w) for w in ww]

                    # Find middle point of edge u,v and add node and edges to graph
                    a = self.get_middle_point(u, v)
                    if not self.graph.has_node(a):
                        # Subdivide edge into two equal edges
                        self.graph.remove_edge(u, v)
                        self.graph.add_edge(u, a)
                        self.graph.add_edge(a, v)

                    # Subdivide each triangle into four equal triangles
                    for tri in triangles:
                        # Get index of third point of triangle u,v,w
                        w = tri[2]

                        # Find middle point of edge v,w and add node and edges to graph
                        b = self.get_middle_point(v, w)
                        if not self.graph.has_node(b):
                            # Subdivide edge into two equal edges
                            self.graph.remove_edge(v, w)
                            self.graph.add_edge(v, b)
                            self.graph.add_edge(b, w)

                        # Find middle point of edge w,u and add node and edges to graph
                        c = self.get_middle_point(w, u)
                        if not self.graph.has_node(b):
                            # Subdivide edge into two equal edges
                            self.graph.remove_edge(w, u)
                            self.graph.add_edge(w, c)
                            self.graph.add_edge(c, u)

                        # Add inner edges of subdivided triangle
                        self.graph.add_edge(a, b)
                        self.graph.add_edge(b, c)
                        self.graph.add_edge(c, a)

    def create(self, global_mesh_recursion, local_mesh_recursion, polygons):
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

        # Now add triangles to mesh
        for tri in faces:
            self.graph.add_edge(tri.v1, tri.v2)
            self.graph.add_edge(tri.v2, tri.v3)
            self.graph.add_edge(tri.v3, tri.v1)

        # Refine graph near shorelines
        self.refine_shoreline_intersections(polygons, local_mesh_recursion)

        # Delete 3DPoint attribute since it is not needed anymore
        for node in self.graph.nodes:
            del self.graph.nodes[node]['point']

        return self.graph


def node_x_polygon(n, geometries):
    shapely_point = Point(n['lon_lat'][0], n['lon_lat'][1])
    tree = STRtree(geometries)
    intersected_exteriors = tree.query(shapely_point)
    if intersected_exteriors:
        for exterior in intersected_exteriors:
            if shapely_point.intersects(exterior):
                return True
    return False


def edge_x_geometry(u, v, geometries):
    shapely_line = LineString([u, v])
    tree = STRtree(geometries)
    intersected_geometries = tree.query(shapely_line)
    if intersected_geometries:
        for geometry in intersected_geometries:
            if shapely_line.intersects(geometry):
                return True
    return False


def remove_nodes_x_polygons(graph, polygons):
    cnt = 0
    nr_nodes = len(graph.nodes)
    graph2 = deepcopy(graph)
    for node_data in graph.nodes(data=True):
        # Printing progress
        cnt += 1
        if cnt % 100 == 0:
            print('removing nodes:', round(cnt/nr_nodes, 2))

        # If node is in polygon, remove from graph
        if node_x_polygon(node_data[1], polygons):
            graph2.remove_node(node_data[0])
    return graph2


def remove_edges_x_polygons(graph, polygons):
    cnt = 0
    nr_edges = len(graph.edges)
    graph2 = deepcopy(graph)
    for edge_data in graph.edges(data=True):
        # Printing progress
        cnt += 1
        if cnt % 100 == 0:
            print('removing edges:', round(cnt/nr_edges, 2))

        # Get indices and positions of nodes from current edge
        u, v = edge_data[0], edge_data[1]
        u_pos, v_pos = graph.nodes[u]['lon_lat'], graph.nodes[v]['lon_lat']

        # Skip border edges since they always intersect polygons in a 2D grid map
        if abs(u_pos[0] - v_pos[0]) > 340:
            continue

        # If edge intersects polygon exterior, remove from graph
        if edge_x_geometry(u_pos, v_pos, polygons):
            graph2.remove_edge(u, v)
    return graph2


def plot_sphere(sphere):
    # 3D plot of unit sphere nodes
    fig = plt.figure()
    ax = Axes3D(fig)
    pos = nx.get_node_attributes(sphere, 'xyz')
    xyz = np.array([value for value in pos.values()])
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, color='black')
    plt.show()


def plot_sphere_edges(sphere):
    # 3D plot of unit sphere edges
    fig = plt.figure()
    ax = Axes3D(fig)
    pos = nx.get_node_attributes(sphere, 'pt_sphere')  # Plots nodes for which attribute 'pt_sphere' exists
    cnt = 0
    for edge in sphere.edges:
        cnt += 1
        print('plotting sphere edges', round(cnt/len(sphere.edges), 2))
        u, v = pos[edge[0]], pos[edge[1]]
        x = np.array([u[0], v[0]])
        y = np.array([u[1], v[1]])
        z = np.array([u[2], v[2]])
        ax.plot(x, y, z, color='black', linewidth=1)
    plt.show()


def plot_grid(graph):
    pos = nx.get_node_attributes(graph, 'lon_lat')
    fig, ax = plt.subplots()
    nx.draw(graph, pos=pos, node_size=5, ax=ax)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()


shorelines_shp_fp = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp'
shorelines = fiona.open(shorelines_shp_fp)
polygon_list = [shape(polygon['geometry']) for polygon in iter(shorelines)]
exterior_list = [shape(polygon['geometry']).exterior for polygon in iter(shorelines)]

# G = nx.read_gpickle("final_d6_vd0.gpickle")
G = IcoSphere().create(4, 0, exterior_list)

# Postprocessing graph
G1 = remove_nodes_x_polygons(G, polygon_list)
G2 = remove_edges_x_polygons(G1, polygon_list)

# Remove isolate nodes
G2_copy = deepcopy(G2)
G2.remove_nodes_from(nx.isolates(G2_copy))

# Set weights to great circle distance (nautical miles)
for e in G2.edges():
    p1 = G2.nodes[e[0]]['xyz']
    p2 = G2.nodes[e[1]]['xyz']
    x1, y1, z1 = p1[0], p1[1], p1[2]
    x2, y2, z2 = p2[0], p2[1], p2[2]

    G2[e[0]][e[1]]['miles'] = 3440 * acos(x1 * x2 + y1 * y2 + z1 * z2)

# connected_components = [len(c) for c in sorted(nx.connected_components(G2), key=len, reverse=True)]
# print(connected_components)
# # for component in connected_components[1:]:
# #     grid_import.remove_nodes_from(component)

nx.write_gpickle(G2, "final_d4_vd0.gpickle")


