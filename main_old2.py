from preprocess_shapefiles import split_large_polygon
from random import seed
import pandas as pd
import pyvisgraph as vg
import fiona
from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
import solution
import pickle
# from mutation import insertion, init_wp_deviation


seed(2)

# # Load data
data_fp = 'C:/dev/data'
# seca_areas = pd.read_csv(data_fp + '/seca_areas.csv').drop('index', axis=1)
speeds_FM = pd.read_excel(data_fp + '/speed_table.xlsx', sheet_name='Fairmaster')
vessel_speed = speeds_FM['Speed'][0]
fuel_rate = speeds_FM['Fuel'][0]
print('Vessel speed {} knots.'.format(vessel_speed))
print('Fuel rate {} tons/day.'.format(fuel_rate))

# Get shorelines shapefile
res = 'f'

shorelines_shp_fp = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/' + res + '/GSHHS_' + res + '_L1.shp'
shoreline_polygons = fiona.open(shorelines_shp_fp)

# Get path shapefile and list, created by visibility_path.py`
path_shp = ([pt for pt in fiona.open('output/shapefiles/path_shape.shp')])
with open('output\path_list_c', 'rb') as file:
    path_list = pickle.load(file)
    print('Path contains {} points.'.format(len(path_list)))

# Create list of Waypoints
waypoints = []
for point in path_list:
    wp = solution.Waypoint(point.x, point.y, )
    waypoints.append(wp)

# Create list of Edges
edges = []
v = waypoints[0]
for w in waypoints[1:]:
    edge = solution.Edge(v, w, vessel_speed)
    v = w
    edges.append(edge)

# for edge in edges:
#     print(edge.intersect_polygon(shorelines_shp_fp))

# Create Route object
route = solution.Route(edges)
r_travel_time = route.travel_time()
print('Travel time is {} hours.'.format(round(r_travel_time, 1)))
r_fuel_consumption = route.fuel(fuel_rate)
print('Total fuel consumption is {} tons'.format(round(r_fuel_consumption, 1)))

# # Check waypoint location feasibility
# points_in_polygon = []
# for waypoint in route.waypoints:
#     polygon = waypoint.in_polygon(shorelines_shp_fp, return_polygon=True)
#     if polygon:
#         points_in_polygon.append([waypoint, polygon])

# Check polygon crossings
polygon_crossings = []
count = 0
for edge in route.edges:
    count += 1
    print(count)
    start_time = time.time()
    polygon = edge.crosses_polygon(shorelines_shp_fp, return_polygon=True)  # 3.1 - 3.8 seconds
    print("--- %s seconds ---" % (time.time() - start_time))
    if polygon:
        polygon_crossings.append([edge, polygon])

# with open('output/route_c', 'wb') as file:
#     pickle.dump(route, file)


# with open('output/route2', 'wb') as file:
#     pickle.dump(route2, file)


# Check if points in path lie in polygon
# for point in waypoints:
#     print(point, point.in_polygon(shorelines_shp_fp))

