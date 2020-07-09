import fiona
import numpy as np
import main
import matplotlib.pyplot as plt
import math
import networkx as nx
import rtree

from shapely.geometry import shape, LineString, Polygon
from deap import creator, base, tools


def edge_x_geos(p1, p2, rtree_idx, geos):
    line = LineString([p1, p2])
    # Returns the geometry indices of the minimum bounding rectangles
    # of polygons that intersect the edge bounds
    mbr_intersections = list(rtree_idx.intersection(line.bounds))
    if mbr_intersections:
        # For every mbr intersection
        # check if its polygon is actually intersect by the edge
        for idx in mbr_intersections:
            if geos[idx].intersects(line):
                return True
    return False


fps = ['C:/dev/data/test_{}/POLYGON.shp'.format(i) for i in range(1, 4)]

shapes = fiona.open(fps[0])
_geos = [[shape(shoreline['geometry']) for shoreline in iter(fiona.open(fp))]
         for fp in fps]

test_1, test_2, test_3 = _geos

polys = test_1

# Populate R-tree index with bounds of polygons
_rtree_idx = rtree.index.Index()
for idx, poly in enumerate(polys):
    _rtree_idx.insert(idx, poly.bounds)

fig, ax = plt.subplots()
for shape in polys:
    x, y = shape.exterior.coords.xy
    plt.plot(x, y)

# Get outer bounds
minx, miny, maxx, maxy = 180, 90, -180, -90
for poly in polys:
    minx_p, miny_p, maxx_p, maxy_p = poly.bounds
    minx, miny = min(minx, minx_p), min(miny, miny_p)
    maxx, maxy = max(maxx, maxx_p), max(maxy, maxy_p)

stepsize = 1  # deg
d = 10
len_x, len_y = int((maxx - minx + 2 * d) / stepsize), int((maxy - miny + 2 * d) // stepsize)
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


print('Removing edges')
for n1, n2 in G.edges():
    p1, p2 = G.nodes[n1]['pos'], G.nodes[n2]['pos']
    if edge_x_geos(p1, p2, _rtree_idx, polys):
        G.remove_edge(n1, n2)

middle = len_y // 2

start, end = (middle, 0), (middle, len_x-1)
print(G.nodes[start]['pos'], G.nodes[end]['pos'])

G.remove_nodes_from(nx.isolates(G.copy()))

nx.draw_networkx_nodes(G, nodelist=[start, end], pos=pos, node_color='red', node_size=20, with_labes=True)

print('Compute shortest path')
path = nx.shortest_path(G, source=start, target=end)
path_edges = list(zip(path, path[1:]))

print('Draw shortest path')
nx.draw(G, pos=pos, node_size=2, node_color='gray', edge_color='lightgray')
nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange', node_size=10)
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='black', width=1)
plt.axis('equal')

print('Create SECA polygon')
x1, x2, y1, y2 = 2, 6.3, 40, 34.7
seca_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
x, y = seca_poly.exterior.coords.xy
plt.plot(x, y)

print('Initialize route planner')

planner = main.RoutePlanner(seca_factor=9, incl_curr=False)
planner.evaluator.rtree_idx = _rtree_idx
planner.evaluator.prep_polys = polys
planner.evaluator.secas = [seca_poly]
planner.evaluator.rtree_idx_seca = rtree.index.Index()
for idx, seca in enumerate(planner.evaluator.secas):
    planner.evaluator.rtree_idx_seca.insert(idx, seca.bounds)
planner.tb.register("evaluate", planner.evaluator.evaluate)
planner.tb.decorate("evaluate", tools.DeltaPenalty(planner.tb.feasible,
                                                [1e+20, 1e+20]))

# Create Fitness and Individual types
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)


def get_shortest_paths(container):
    wps = [G.nodes[n]['pos'] for n in path]
    speeds = [planner.vessel.speeds[0]] * (len(path) - 1) + [None]
    ind = [list(tup) for tup in zip(wps, speeds)]
    return [[container(ind)]]


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
