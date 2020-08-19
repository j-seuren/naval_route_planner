import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd

from mpl_toolkits.basemap import Basemap
from pathlib import Path


class OrtecPlanner:
    def __init__(self):
        self.dataLoc = Path('C:/dev/data/data_20200327')
        self.arcsFile = Path('routeplanner_output_waypoints.csv')
        self.portsFile = Path('ports.csv')
        self.GSavePath = Path('C:/dev/projects/naval_route_planner/data/ortec_planner/ortec_graph_v1')
        self.G = None
        self.arcs = None

    def get_arcs(self):
        if not self.arcs:
            arcs = pd.read_csv(self.dataLoc / self.arcsFile, sep=None)
            return arcs.loc[arcs['Season'] == 'Summer']

    def get_graph(self):
        if os.path.exists(self.GSavePath):
            print('Loading graph')
            self.G = nx.read_gpickle(self.GSavePath)
        else:
            print('Creating graph')
            self.G = nx.empty_graph()
            self.arcs = self.get_arcs()

            ports = pd.read_csv(self.dataLoc / self.portsFile)
            ports = ports[['portname', 'latitude', 'longitude']]

            u_list = self.arcs.NodeFrom.unique()

            for u in u_list:
                start = ports.loc[(ports['portname'] == u)]
                if len(start) != 1:
                    locs1 = np.array([[]])
                else:
                    lon_s, lat_s = float(start['longitude']), float(start['latitude'])
                    locs1 = np.array([[lon_s, lat_s]])
                df = self.arcs.loc[self.arcs['NodeFrom'] == u]
                v_list = df.NodeTo.unique()
                for v in v_list:
                    end = ports.loc[(ports['portname'] == v)]
                    wps = self.arcs.loc[(self.arcs['NodeFrom'] == u) & (self.arcs['NodeTo'] == v)]
                    locs_df = wps[['WaypointLong', 'WaypointLat']]
                    if locs1.size > 0:
                        locs = np.append(locs1, locs_df.to_numpy(), axis=0)
                    else:
                        locs = locs_df.to_numpy()
                    if len(end) == 1:
                        lon_e, lat_e = float(end['longitude']), float(end['latitude'])
                        locs = np.append(locs, np.array([[lon_e, lat_e]]), axis=0)

                    for i, loc in enumerate(locs[:-1]):
                        uu, vv = loc, locs[i+1]
                        for n in [uu, vv]:
                            self.G.add_node(tuple(n), pos=n)
                        self.G.add_edge(tuple(uu), tuple(vv))
        return self.G

    def plot_graph(self):
        fig, ax = plt.subplots()
        m = Basemap(projection='merc', llcrnrlon=-180, llcrnrlat=-80, urcrnrlon=180, urcrnrlat=80, resolution='l', ax=ax)
        m.drawcoastlines()
        m.fillcontinents(alpha=0.5)
        pos = nx.get_node_attributes(self.G, 'pos')
        for key, val in pos.items():
            pos[key] = m(val[0], val[1])
        nx.draw(self.G, pos=pos, node_size=2, ax=ax)

    def save_graph(self):
        if self.G:
            nx.write_gpickle(self.G, self.GSavePath)
        else:
            print('No graph created')


if __name__ == '__main__':
    planner = OrtecPlanner()
    graph = planner.get_graph()
    sorted_cc = sorted(nx.connected_components(graph), key=len, reverse=True)
    planner.plot_graph()


    print(len(sorted_cc))

    plt.show()

