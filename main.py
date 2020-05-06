import get_data as get
import pyvisgraph as vg
import fiona
from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
import route
import pickle

from haversine import haversine

# # Load data
# seca_areas = get.seca()
speeds_FM = get.speed_table('Fairmaster')
# shore_areas = get.shore_areas('crude')
shorelines_shp_fp = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp'
path_shp = ([pt for pt in fiona.open('output/shapefiles/path_shape.shp')])
with open('output\path_list_l', 'rb') as file:
    path_list = pickle.load(file)
    print('Path contains {} points.'.format(len(path_list)))

vessel_speed = speeds_FM['Speed'][0]
fuel_rate = speeds_FM['Fuel'][0]
print('Vessel speed {} knots.'.format(vessel_speed))
print('Fuel rate {} tons/day.'.format(fuel_rate))

# Create list of Waypoints
waypoints = []
for point in path_list:
    wp = route.Waypoint(point.x, point.y, )
    waypoints.append(wp)

# Create list of Edges
edges = []
v = waypoints[0]
for w in waypoints[1:]:
    edge = route.Edge(v, w, vessel_speed)
    v = w
    edges.append(edge)

for edge in edges:
    print(edge.intersect_polygon(shorelines_shp_fp))

# Create Route object
route = route.Route(edges)
r_travel_time = route.travel_time()
print('Travel time is {} hours.'.format(round(r_travel_time, 1)))
r_fuel_consumption = route.fuel(fuel_rate)
print('Total fuel consumption is {} tons'.format(round(r_fuel_consumption, 1)))


def pts_shp_in_polygon(points, polygons_shp):  # Check if Fiona list of points lie within polygons shapefile
    multi_pol = fiona.open(polygons_shp)
    multi = multi_pol.next()
    points_in_polygon = []
    for idx, pt in enumerate(points):
        pt = shape(pt['geometry'])
        if pt.within(shape(multi['geometry'])):
            points_in_polygon.append([idx, shape(points[i]['geometry'])])
            print('Point {} is in polygon.'.format(idx), shape(points[idx]['geometry']))
    return points_in_polygon

# Check if points in path lie in polygon
# for point in waypoints:
#     print(point, point.in_polygon(shorelines_shp_fp))

