import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
import pyvisgraph as vg


pd.set_option('display.max_columns', None)

# Parameters
data_fp = 'C:\dev\data'
startPort = 'SCHEVENINGEN'
endPort = 'KINGSTON UPON HULL'
sea = 'North Sea'

# Get Data
# World Seas
World_Seas_IHO_v3_fp = os.path.join(data_fp, 'World_Seas_IHO_v3', 'World_Seas_IHO_v3.shp')
World_Seas_gdf = gpd.read_file(World_Seas_IHO_v3_fp)
sea_gdf = World_Seas_gdf.loc[World_Seas_gdf['NAME'] == sea]
subset = sea_gdf.loc[:, 'min_X':'max_Y']
bbox = tuple(subset.iloc[0, :])

# World Shorelines
GSHHS_l_L1_fp = os.path.join(data_fp, 'gshhg-shp-2.3.7', 'GSHHS_shp', 'l', 'GSHHS_l_L1.shp')
GSHHS_gdf = gpd.read_file(GSHHS_l_L1_fp, bbox=bbox)

# World Ports
World_Ports_fp = os.path.join(data_fp, 'World_Port_Index', 'World_Port_Index.shp')
World_Ports_gdf = gpd.read_file(World_Ports_fp)
World_Ports_gdf.set_index("PORT_NAME", drop=False, inplace=True)

# Routeplanner waypoints
rp_waypoints_fp = os.path.join(data_fp, 'data_20200327', 'routeplanner_output_waypoints.csv')
rp_waypoints_df = pd.read_csv(rp_waypoints_fp, delimiter=';')
rp_waypoints_df_sea = rp_waypoints_df[(rp_waypoints_df['WaypointLong'] >= bbox[0]) &
                                      (rp_waypoints_df['WaypointLong'] <= bbox[2]) &
                                      (rp_waypoints_df['WaypointLat'] >= bbox[1]) &
                                      (rp_waypoints_df['WaypointLat'] <= bbox[3])]

rp_waypoints_sea = rp_waypoints_df_sea[['WaypointLong', 'WaypointLat']]
rp_waypoints_sea = rp_waypoints_sea.to_numpy()
rp_waypoints_sea = [Point(xy) for xy in rp_waypoints_sea]
rp_waypoints_sea = gpd.GeoDataFrame(rp_waypoints_sea,
                                    columns=['geometry'],
                                    crs=World_Seas_gdf.crs)

# Get port locations
ports = [startPort, endPort]
portLocations_gdf = World_Ports_gdf.loc[ports, ['LONGITUDE', 'LATITUDE']]
portLocations = portLocations_gdf.to_numpy()
portLocations = [Point(xy) for xy in portLocations ]
portLocations = gpd.GeoDataFrame(portLocations,
                                 columns=['geometry'],
                                 crs=World_Seas_gdf.crs)

polys = [[vg.Point(0.0,1.0), vg.Point(3.0,1.0), vg.Point(1.5,4.0)],
         [vg.Point(4.0,4.0), vg.Point(7.0,4.0), vg.Point(5.5,8.0)]]

# polys = GSHHS_gdf['geometry']
# g = vg.VisGraph()
# g.build(polys)
# shortest = g.shortest_path(startPort, endPort)

# Plot
fig, ax = plt.subplots()

sea_gdf.plot(ax=ax)

# Add port locations
portLocations.plot(ax=ax,
                    color='springgreen',
                    marker='*',
                    markersize=45)

# Add route waypoints
rp_waypoints_sea.plot(ax=ax,
                      color='red',
                      marker='.',
                      markersize=10)

# Set the x and y axis labels
ax.set(xlabel="Longitude (Degrees)",
       ylabel="Latitude (Degrees)",
       title="North Sea")

# plt.show()
