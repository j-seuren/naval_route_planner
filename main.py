import fiona
import pickle
import solution
import pandas as pd
from random import seed
from operators import insert_waypoint, get_one_crossover_from_routes
from copy import deepcopy

seed(2)

shorelines_shp_fp = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp'
shorelines_polygons = fiona.open(shorelines_shp_fp)
speeds_FM = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name='Fairmaster')
vessel_speed = speeds_FM['Speed'][0]

path_shp = ([pt for pt in fiona.open('output/shapefiles/path_shape.shp')])
with open('output\path_list_c_ocean', 'rb') as file:
    path_list = pickle.load(file)

# Create route class
waypoints = []
for waypoint in path_list:
    wp = solution.Waypoint(waypoint.x, waypoint.y)
    waypoints.append(wp)

edges = []
v = waypoints[0]
for w in waypoints[1:]:
    edge = solution.Edge(v, w, vessel_speed)
    v = w
    edges.append(edge)

route1 = solution.Route(edges)

# with open('output/route_ocean', 'wb') as file:
#     pickle.dump(route1, file)

# initial_routes = []
# count = 0
# for i in range(5):
#     init_route = deepcopy(route1)
#     count += 1
#     print(count)
#     for j in range(10):
#         init_route = insert_waypoint(init_route, bisector_length_ratio=0.5, polygons=shorelines_polygons)
#     initial_routes.append(init_route)

with open('output/initial_routes', 'rb') as file:
    initial_routes = pickle.load(file)

crossover_route = get_one_crossover_from_routes(initial_routes, shorelines_polygons)

# with open('output/crossover_routes', 'wb') as file:
#     pickle.dump(crossover_routes, file)





# # Move waypoints from shore (polygon) lines
# for waypoint in route1.waypoints:
#     route2 = route2.waypoint_feasible(waypoint, shorelines_polygons, radius=0.1, check_polygon_crossing=True)

# # Check polygon crossings
# polygon_crossings = []
# count = 0
# for edge in route1.edges:
#     count += 1
#     print(count)
#     polygon = edge.crosses_polygon(shorelines_polygons, return_polygon=True)  # 3.1 - 3.8 seconds
#     if polygon:
#         polygon_crossings.append([edge, polygon])
#
# for i in polygon_crossings:
#     print(i[0])

# Fix polygon crossings

# # Move waypoint outside polygon
# if route is not route_inf:
#     for waypoint in route.waypoints:
#         if waypoint.in_polygon(shorelines_polygons):
#             print('Error: waypoint still in polygon')
#
# # Insert waypoint
# test_route = route
#

#
# # # Delete waypoint
# # start_time = time.time()
# #
# # test_route = test_route.delete_waypoint(shorelines_polygons)
# #
# # print("--- %s seconds ---" % (time.time() - start_time))
#


