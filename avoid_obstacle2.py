import fiona
import pickle
import solution
import pandas as pd
import time
from random import seed
from plot_edge import plot_edge

# with open('output/test_route', 'rb') as file:
#     route2 = pickle.load(file)
#
shorelines_shp_fp = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/l/GSHHS_l_L1.shp'
shorelines_polygons = fiona.open(shorelines_shp_fp)
speeds_FM = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name='Fairmaster')
vessel_speed = speeds_FM['Speed'][0]
#
# # Check polygon crossings
# polygon_crossings = []
# count = 0
# for edge in route2.edges:
#     count += 1
#     print(count)
#     polygon = edge.crosses_polygon(shorelines_polygons, return_polygon=True)  # 3.1 - 3.8 seconds
#     if polygon:
#         polygon_crossings.append([edge, polygon])
#
# for i in polygon_crossings:
#     print(i[0])

with open('output/polygon_crossings', 'rb') as file:
    polygon_crossings = pickle.load(file)



# Fix polygon crossings
# plot_edge(polygon_crossings[0][0])
for edge in polygon_crossings[0][:]:
    value = edge.avoid_obstacle(shorelines_polygons)
print(value)
