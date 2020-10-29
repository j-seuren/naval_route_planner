import fiona
import numpy as np
import main
import matplotlib.pyplot as plt
import networkx as nx
import os

from data_config.navigable_area import get_rtree
from mpl_toolkits.basemap import Basemap
from shapely.geometry import shape, LineString, Polygon


def geo_x_geos(treeDict, pt1, pt2):
    geo = LineString([pt1, pt2])

    # Return a list of all geometries in the R-tree whose extents
    # intersect the extent of geom
    extent_intersections = treeDict['tree'].query(geo)
    if extent_intersections:
        # Check if any geometry in extent_intersections actually intersects line
        for geom in extent_intersections:
            geomIdx = treeDict['indexByID'][id(geom)]
            prepGeom = treeDict['polys'][geomIdx]
            if pt2 is None and prepGeom.contains(geo):
                return True
            elif prepGeom.intersects(geo):
                return True

    return False


def get_polygons(instance):
    # ---- Setting up navigable area
    # Obstacles
    fps = ['test_data/test_{}/POLYGON.shp'.format(i) for i in range(1, 4)]
    shapes = [fiona.open(fp) for fp in fps]
    obstacleList = [[shape(poly['geometry']) for poly in iter(shp)] for shp in shapes]
    obstacles = obstacleList[instance]

    # ECAs
    ecaBounds = [[(2, 6.3, 40, 34.7)], [], [(21, 39, 6.5, -11), (50, 60, 35, 28), (36, 47, 27, 13)]]
    ecas = []
    for bound in ecaBounds[instance]:
        x1, x2, y1, y2 = bound
        ecas.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

    return obstacles, ecas


class CaseStudy:
    def __init__(self, inputParameters, criteria, instance=2):
        self.inputParameters = inputParameters
        self.fig, self.ax = plt.subplots()
        self.obstacles, self.ecas = get_polygons(instance)

        # Get extent
        minExt, maxExt = np.array([180, 90]), np.array([-180, -90])
        for poly in self.obstacles:
            minx, miny, maxx, maxy = poly.bounds
            minExt = np.minimum(minExt, np.array([minx, miny]))
            maxExt = np.maximum(maxExt, np.array([maxx, maxy]))

        margin = 10
        self.extent = [minExt - margin, maxExt + margin]
        self.obstacleRtreeDict, self.ecaRtreeDict = get_rtree(self.obstacles), get_rtree(self.ecas)
        self.graph, self.xv, self.yv, self.pos = self.construct_graph()
        self.path, self.ecaPath = self.graph_paths()
        self.solver, self.initialRoutes, self.processor = self.initialize_planner(criteria)

        self.start = self.initialRoutes[0]['route'][0]['time'][0][0]
        self.end = self.initialRoutes[0]['route'][0]['time'][-1][0]



    # def generate_currents(self):


    def plot_polys(self, m):
        for obstacle in self.obstacles:
            x, y = obstacle.exterior.coords.xy
            m.save_fronts(x, y, latlon=True)

        for eca in self.ecas:
            x, y = eca.exterior.coords.xy
            m.save_fronts(x, y, latlon=True)

    def construct_graph(self, step=1):
        dim = ((self.extent[1] - self.extent[0]) // step).astype(int)
        graph = nx.grid_graph(dim=list(dim))

        # Create node mesh
        xv, yv = [np.linspace(self.extent[0][i], self.extent[1][i], dim[i]) for i in range(2)]
        pos = {}
        for i in range(len(xv)):
            for j in range(len(yv)):
                graph.nodes[(j, i)]['deg'] = (xv[i], yv[j])
                pos[(j, i)] = (xv[i], yv[j])

        # Remove infeasible edges and set eca weights
        graph.edges.data('eca', default=1)
        graph.edges.data('weight', default=1)
        for n1, n2 in graph.edges():
            p1, p2 = graph.nodes[n1]['deg'], graph.nodes[n2]['deg']
            if geo_x_geos(self.obstacleRtreeDict, p1, p2):
                graph.remove_edge(n1, n2)
            elif geo_x_geos(self.ecaRtreeDict, p1, p2):
                graph[n1][n2]['eca'] = 10
        graph.remove_nodes_from(nx.isolates(graph.copy()))  # Remove isolates

        return graph, xv, yv, pos

    def graph_paths(self):
        # Draw graph
        midYIdx = len(self.yv) // 2
        start, end = (midYIdx, 0), (midYIdx, len(self.xv) - 1)

        # Computed shortest path
        path = nx.shortest_path(self.graph, source=start, target=end, weight='weight')
        ecaPath = nx.shortest_path(self.graph, source=start, target=end, weight='eca')
        return path, ecaPath

    def draw_graph(self):
        start, end = self.path[0], self.path[-1]
        pathEdges = list(zip(self.path, self.path[1:]))
        ecaPathEdges = list(zip(self.ecaPath, self.ecaPath[1:]))
        nx.draw_networkx_nodes(self.graph, nodelist=[start, end], pos=self.pos, node_color='red', node_size=20,
                               with_labes=True)
        nx.draw(self.graph, pos=self.pos, node_size=2, node_color='gray', edge_color='lightgray')
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.path, node_color='orange', node_size=10)
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=pathEdges, edge_color='orange', width=1)
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.ecaPath, node_color='purple', node_size=10)
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=ecaPathEdges, edge_color='purple', width=1)
        plt.axis('equal')

    def initialize_planner(self, criteria):
        # ---- Initialize route planner
        os.chdir('..')
        planner = main.RoutePlanner(criteria=criteria, inputParameters=self.inputParameters)
        planner.evaluator.landRtree = self.obstacleRtreeDict
        planner.evaluator.ecaTree = self.ecaRtreeDict
        # planner.evaluator.geod.distance = planner.geod.euclidean
        nsga = planner.NSGAII(planner.tb, planner.evaluator, planner.mstats, planner.front, planner.get_days,
                              planner.stopping_criterion, planner.p).optimize

        # Convert paths to individuals
        paths = [{'path': [{'time': self.path, 'eca': self.ecaPath}], 'xCanals': []}]
        initialRoutes = planner.initializer.paths_to_routes(paths, self.graph)
        processor = planner.post_process

        return nsga, initialRoutes, processor

    def compute(self):
        # Optimize initial paths and postprocess results
        result = self.solver(startEnd=(self.start, self.end), initRoutes=self.initialRoutes,
                             startDate=None, inclCurr=False, inclWeather=False, seed=1)
        postProcessed = self.processor(result)

        return postProcessed

    def plot_basemap(self, postProcessed, plotInitial=True):
        m = Basemap(projection='merc',
                    llcrnrlon=self.extent[0][0], llcrnrlat=self.extent[0][1],
                    urcrnrlon=self.extent[1][0], urcrnrlat=self.extent[1][1],
                    ax=self.ax)
        colors = iter(['red', 'blue'])
        # Plot minimal time and minimal cost routes
        for response in postProcessed['routeResponse']:
            wps = [(wp['lon'], wp['lat']) for wp in response['waypoints']]
            c = next(colors)

            for edge in zip(wps[:-1], wps[1:]):
                ((lon1, lat1), (lon2, lat2)) = edge
                m.drawgreatcircle(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2, color=c)

        if plotInitial:
            for initialRoute in self.initialRoutes:
                for subRoute in initialRoute['route']:
                    for objRoute in subRoute.values():
                        wps = [item[0] for item in objRoute]
                        for edge in zip(wps[:-1], wps[1:]):
                            ((lon1, lat1), (lon2, lat2)) = edge
                            m.drawgreatcircle(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2, color='black')

        return m

    def plot_output(self, postProcessed, plotInitial=True):
        m = self.plot_basemap(postProcessed, plotInitial=plotInitial)
        self.plot_polys(m)
        # self.draw_graph()


parameters = {'gen': 1000, 'gauss': True,
                           'mutationOperators': ['insert',
                                                 'move',
                                                 # 'speed',
                                                 'delete']}
tester = CaseStudy(parameters, criteria={'minimalTime': True,
                                         'minimalCost': True})
output = tester.compute()

tester.plot_output(output, plotInitial=False)

plt.show()
