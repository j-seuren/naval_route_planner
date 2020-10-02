import heapq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import uuid

from haversine import haversine
from mpl_toolkits.basemap import Basemap
from pathlib import Path


class ORTECPlanner:
    def __init__(self, DIR=Path('D:/')):
        dataDir = DIR / 'data/data_20200327'
        self.arcsFile = dataDir / 'routeplanner_output_waypoints.csv'
        self.portsFile = dataDir / 'ports_excel.xlsx'
        self.portsFile_old = dataDir / 'ports_1.csv'
        self.GSavePath0 = DIR / 'data/ortec_planner/ortec_graph_v1'
        self.GSavePath = self.GSavePath0
        self.graph = None
        self.season = None

    def get_shortest_path(self, start, end):
        if self.graph is None:
            self.graph = self.get_graph()

        def dist_heuristic(n1, n2):
            return haversine(self.graph.nodes[n1]['pos'], self.graph.nodes[n2]['pos'], unit='nmi')

        # Add start and end nodes to graph, and create edges to three nearest nodes.
        startID, endID = str(uuid.uuid4()), str(uuid.uuid4())
        points = {startID: start, endID: end}
        for ptKey, endPoint in points.items():
            # Compute distances of points in neighborhood to pt location
            radius = 0
            while True:
                radius += 0.2
                neighborhood = {n: d['pos'] for n, d in self.graph.nodes(data=True) if
                                abs(d['pos'][0] - endPoint[0]) < radius and abs(d['pos'][1] - endPoint[1]) < radius}
                if len(neighborhood) > 3:
                    print('neighborhood radius', radius, ', neighbors', len(neighborhood))
                    break
            d = {n: haversine(endPoint, nDeg, unit='nmi') for n, nDeg in neighborhood.items()}

            self.graph.add_node(ptKey, pos=endPoint)

            # Add three shortest feasible edges to  point
            distsToPtList = heapq.nsmallest(100, d, key=d.get)
            distsToPt = iter(distsToPtList)
            for _ in range(3):
                try:
                    n = next(distsToPt)
                except StopIteration:
                    print('Too few points in neighborhood')
                    break
                self.graph.add_edge(ptKey, n, miles=d[n])

        return nx.astar_path(self.graph, startID, endID, heuristic=dist_heuristic, weight='dist')

    def get_arcs(self, season):
        arcs = pd.read_csv(self.arcsFile, sep=None)
        return arcs.loc[arcs['Season'] == season]

    def get_ports(self):
        ports = pd.read_excel(self.portsFile)
        ports = ports[['portname', 'latitude', 'longitude']]
        ports_old = pd.read_csv(self.portsFile_old, sep=None)
        common = ports_old.merge(ports, on=["portname"])
        ports_old = ports_old[~ports_old.portname.isin(common.portname)]
        ports_old = ports_old.loc[ports_old['latitude'] != 0.]
        ports_old = ports_old[['portname', 'latitude', 'longitude']]

        ports = ports.append(ports_old)
        ports['portname'] = ports['portname'].astype(str)
        return ports

    def get_graph(self, season='Summer'):
        self.GSavePath = self.GSavePath0.as_posix() + '_{}'.format(season)
        if os.path.exists(self.GSavePath):
            print('Loading graph')
            self.graph = nx.read_gpickle(self.GSavePath)
            return self.graph

        arcs = self.get_arcs(season)
        ports = self.get_ports()

        print('Creating graph')
        self.graph = nx.empty_graph()

        # Get all origins
        origins = arcs.NodeFrom.unique()
        for org in origins:
            start = ports.loc[(ports['portname'] == org)]  # Get waypoint of origin port
            if len(start) != 1:  # Port does not exist in port files
                path0 = np.empty((0, 2))
                print('{}'.format(org))
            else:
                lon_s, lat_s = float(start['longitude']), float(start['latitude'])
                path0 = np.array([[lon_s, lat_s]])

            # Get all destinations corresponding to origin
            orgDF = arcs.loc[arcs['NodeFrom'] == org]
            destinations = orgDF.NodeTo.unique()

            for dst in destinations:
                # Get all intermediate waypoints corresponding to origin and destination
                wps = arcs.loc[(arcs['NodeFrom'] == org) & (arcs['NodeTo'] == dst)]
                wps = wps[['WaypointLong', 'WaypointLat']]

                path = np.append(path0, wps.to_numpy(), axis=0)  # Append intermediate waypoints to paths

                end = ports.loc[(ports['portname'] == dst)]  # Get waypoint of destination port
                if len(end) == 1:
                    lon_e, lat_e = float(end['longitude']), float(end['latitude'])
                    path = np.append(path, np.array([[lon_e, lat_e]]), axis=0)

                for i, loc in enumerate(path[:-1]):
                    u, v = loc, path[i+1]
                    for n in [u, v]:
                        self.graph.add_node(tuple(n), pos=tuple(n))
                    self.graph.add_edge(tuple(u), tuple(v))

        # Set attributes
        for n1, n2 in self.graph.edges():
            p1, p2 = self.graph.nodes[n1]['pos'], self.graph.nodes[n2]['pos']
            self.graph[n1][n2]['dist'] = haversine(p1, p2, unit='nmi')

        return self.graph

    def plot_graph(self, paths=None, delLong=True):
        fig, ax = plt.subplots()
        m = Basemap(projection='merc', llcrnrlon=-179, llcrnrlat=-75, urcrnrlon=179, urcrnrlat=75, resolution='l',
                    ax=ax)
        m.drawcoastlines(linewidth=0.5)
        m.drawmapboundary()
        m.fillcontinents(color='lightgray')
        pos = nx.get_node_attributes(self.graph, 'pos')
        for key, val in pos.items():
            pos[key] = m(val[0], val[1])

        if delLong:
            edgeList = []
            for edge in self.graph.edges():
                # Get indices and long/lat positions of nodes from current edge
                n1, n2 = edge[0], edge[1]
                p1, p2 = self.graph.nodes[n1]['pos'], self.graph.nodes[n2]['pos']

                # Skip border edges since they always intersect polygons in a 2D grid map
                if not abs(p1[0] - p2[0]) > 340:
                    edgeList.append(edge)
        else:
            edgeList = self.graph.edges()

        nx.draw_networkx_edges(self.graph, edgelist=edgeList, pos=pos, width=0.5, edge_color='dimgray', ax=ax)

        if paths:
            for path in paths:
                path_edges = list(zip(path, path[1:]))
                # nx.draw_networkx_nodes(self.graph, pos, nodelist=path, node_color='r', ax=ax, node_size=1)
                nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='r', width=1, ax=ax)

    def plot_cc(self):
        sorted_cc = sorted(nx.connected_components(self.graph), key=len, reverse=True)
        sgs = (self.graph.subgraph(c).copy() for c in sorted_cc)
        for index, component in enumerate(sgs):
            pos = nx.get_node_attributes(component, 'pos')
            nx.draw(component, pos=pos)

    def save_graph(self):
        if self.graph:
            nx.write_gpickle(self.graph, self.GSavePath)
        else:
            print('No graph created')


if __name__ == '__main__':
    from support import locations

    planner = ORTECPlanner()
    _graph = planner.get_graph(season='Summer')
    planner.save_graph()

    starts = [locations['Lima'], locations['Banjul'], locations['Sao Paulo'], locations['Rotterdam'],
              locations['Wellington']]
    ends = [locations['San Francisco'], locations['Singapore'], locations['Luanda'], locations['Houston'],
            locations['Perth']]

    _paths = [planner.get_shortest_path(start, end) for start, end in zip(starts, ends)]

    planner.plot_graph(_paths, delLong=True)

    sorted_cc = sorted(nx.connected_components(_graph), key=len, reverse=True)
    plt.savefig('figure.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.show()

    print(len(sorted_cc))
    print(sorted_cc)

