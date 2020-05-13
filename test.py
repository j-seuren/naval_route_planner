import fiona
import pickle
import solution
import pandas as pd
import time
from random import seed

seed(1)

shorelines_shp_fp = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/l/GSHHS_l_L1.shp'
shorelines_polygons = fiona.open(shorelines_shp_fp)
speeds_FM = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name='Fairmaster')
vessel_speed = speeds_FM['Speed'][0]

path_shp = ([pt for pt in fiona.open('output/shapefiles/path_shape.shp')])
with open('output\path_list_c', 'rb') as file:
    path_list = pickle.load(file)

# Create route
waypoints = []
for waypoint in path_list:
    wp = solution.Waypoint(waypoint.x, waypoint.y)
    waypoints.append(wp)

route = solution.Route(waypoints, vessel_speed)

# Check polygon crossings
polygon_crossings = []
count = 0
for edge in route.edges:
    count += 1
    print(count)
    polygon = edge.crosses_polygon(shorelines_polygons, return_polygon=True)  # 3.1 - 3.8 seconds
    if polygon:
        polygon_crossings.append([edge, polygon])

for i in polygon_crossings:
    print(i[0])

# Fix polygon crossings

# # Check if there are waypoints within a polygon
# route_inf = route
#
# for waypoint in route_inf.waypoints:
#     if waypoint.in_polygon(shorelines_polygons):
#         route = route.move_waypoint(waypoint, shorelines_polygons)
#
#
# # Move waypoint outside polygon
# if route is not route_inf:
#     for waypoint in route.waypoints:
#         if waypoint.in_polygon(shorelines_polygons):
#             print('Error: waypoint still in polygon')
#
# # Insert waypoint
# test_route = route
#
# for i in range(20):
#     test_route = test_route.insert_waypoint(0.9, shorelines_polygons)
#
# # # Delete waypoint
# # start_time = time.time()
# #
# # test_route = test_route.delete_waypoint(shorelines_polygons)
# #
# # print("--- %s seconds ---" % (time.time() - start_time))
#
with open('output/test_route', 'wb') as file:
    pickle.dump(route, file)

