import fiona
import numpy as np
import main
import matplotlib.pyplot as plt
import math
import networkx as nx
import rtree

from shapely.geometry import shape, LineString, Polygon
from deap import creator, tools


def edge_x_geos(_p1, _p2, rtree_idx, geos):
    line = LineString([_p1, _p2])
    # Returns the geometry indices of the minimum bounding rectangles
    # of polygons that intersect the edge bounds
    mbr_intersections = list(rtree_idx.intersection(line.bounds))
    if mbr_intersections:
        # For every mbr intersection
        # check if its polygon is actually intersect by the edge
        for _idx in mbr_intersections:
            if geos[_idx].intersects(line):
                return True
    return False


fps = ['C:/dev/data/test_{}/POLYGON.shp'.format(i) for i in range(1, 4)]

shapes = fiona.open(fps[0])
_geos = {i: [shape(shoreline['geometry']) for shoreline in iter(fiona.open(fp))]
         for i, fp in enumerate(fps)}

instance_idx = 2
polys = _geos[instance_idx]

fig, ax = plt.subplots()
for shape in polys:
    x, y = shape.exterior.coords.xy
    plt.plot(x, y)

print('Create ECA polygon')  # x1, x2, y1, y2
ecas_coordinates = {0: [(2, 6.3, 40, 34.7)],
                    1: [],
                    2: [(21, 39, 6.5, -11),
                        (50, 60, 35, 28),
                        (36, 47, 27, 13)]}

ecas = []
for tup in ecas_coordinates[instance_idx]:
    x1, x2, y1, y2 = tup
    eca_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    x, y = eca_poly.exterior.coords.xy
    plt.plot(x, y)
    ecas.append(eca_poly)

# Populate R-tree index with bounds of polygons
_rtree_idx = rtree.index.Index()
for idx, poly in enumerate(polys):
    _rtree_idx.insert(idx, poly.bounds)
_rtree_idx_eca = rtree.index.Index()
for idx, eca in enumerate(ecas):
    _rtree_idx_eca.insert(idx, eca.bounds)

# Get outer bounds
minx, miny, maxx, maxy = 180, 90, -180, -90
for poly in polys:
    minx_p, miny_p, maxx_p, maxy_p = poly.bounds
    minx, miny = min(minx, minx_p), min(miny, miny_p)
    maxx, maxy = max(maxx, maxx_p), max(maxy, maxy_p)

step_size = 1  # deg
d = 10
len_x, len_y = int((maxx - minx + 2 * d) / step_size), int((maxy - miny + 2 * d) // step_size)
print('Creating graph')
G = nx.grid_graph(dim=[len_x, len_y])

print('Getting positions')
xx = np.linspace(minx - d, maxx + d, len_x)
yy = np.linspace(miny - d, maxy + d, len_y)
pos = {}
for i in range(len_x):
    for j in range(len_y):
        G.nodes[(j, i)]['pos'] = (xx[i], yy[j])
        pos[(j, i)] = (xx[i], yy[j])

print('Removing edges and assigning weights')
G.edges.data('eca', default=1)
G.edges.data('weight', default=1)
for n1, n2 in G.edges():
    p1, p2 = G.nodes[n1]['pos'], G.nodes[n2]['pos']
    if edge_x_geos(p1, p2, _rtree_idx, polys):
        G.remove_edge(n1, n2)
    elif edge_x_geos(p1, p2, _rtree_idx_eca, ecas):
        G[n1][n2]['eca'] = 10


middle = len_y // 2

start, end = (middle, 0), (middle, len_x-1)

G.remove_nodes_from(nx.isolates(G.copy()))

nx.draw_networkx_nodes(G, nodelist=[start, end], pos=pos, node_color='red', node_size=20, with_labes=True)

print('Compute shortest path')
path = nx.shortest_path(G, source=start, target=end, weight='weight')
eca_path = nx.shortest_path(G, source=start, target=end, weight='eca')
path_edges = list(zip(path, path[1:]))
eca_path_edges = list(zip(eca_path, eca_path[1:]))

print('Draw shortest path')
nx.draw(G, pos=pos, node_size=2, node_color='gray', edge_color='lightgray')
nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange', node_size=10)
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=1)
nx.draw_networkx_nodes(G, pos, nodelist=eca_path, node_color='purple', node_size=10)
nx.draw_networkx_edges(G, pos, edgelist=eca_path_edges, edge_color='purple', width=1)
plt.axis('equal')

print('Initialize route planner')

planner = main.RoutePlanner(eca_f=1.2, incl_curr=False)
planner.evaluator.tree = _rtree_idx
planner.evaluator.polys = polys
planner.evaluator.ecas = ecas
planner.evaluator.eca_tree = rtree.index.Index()
for idx, eca in enumerate(planner.evaluator.ecas):
    planner.evaluator.eca_tree.insert(idx, eca.bounds)
planner.tb.register("evaluate", planner.evaluator.evaluate)
planner.tb.decorate("evaluate", tools.DeltaPenalty(planner.tb.feasible, [1e+20, 1e+20]))


def get_shortest_paths(container):
    init_inds = {0: {0: {}}}
    wps = [G.nodes[n]['pos'] for n in path]
    speeds = [planner.vessel.speeds[0]] * (len(path) - 1) + [None]
    _ind = [list(_tup) for _tup in zip(wps, speeds)]
    init_inds[0][0][0] = container(_ind)

    wps = [G.nodes[n]['pos'] for n in eca_path]
    speeds = [planner.vessel.speeds[0]] * (len(path) - 1) + [None]
    _ind = [list(_tup) for _tup in zip(wps, speeds)]
    init_inds[0][0][1] = container(_ind)
    return init_inds


planner.tb.register("get_shortest_paths", get_shortest_paths, creator.Individual)
_paths, _path_logs, _init_routes = planner.nsga2(seed=1)

best_inds = {}
for _sub_paths in _paths.values():
    for _pop in _sub_paths.values():
        min_tt, min_fc = math.inf, math.inf
        tt_ind, fc_ind = None, None
        for min_ind in _pop:
            tt, fc = min_ind.fitness.values
            if tt < min_tt:
                tt_ind = min_ind
                min_tt = tt
            if fc < min_fc:
                fc_ind = min_ind
                min_fc = fc

        best_inds['Minimal fuel: '] = fc_ind
        best_inds['Minimal time: '] = tt_ind

for ind in best_inds.values():
    print(ind.fitness.values)
    waypoints = [item[0] for item in ind]
    x, y = [i for i, j in waypoints], [j for i, j in waypoints]
    plt.plot(x, y)

# for pop in _init_routes:
#     for ind in pop:
#         xy = [item[0] for item in ind]
#         x, y = [i for i, j in xy], [j for i, j in xy]
#         plt.plot(x, y)

print('Show on screen')
plt.show()
