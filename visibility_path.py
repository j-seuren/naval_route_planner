import pyvisgraph as vg
import folium
from haversine import haversine
import pickle
import shapefile

# Example points
startLat = 52.1
startLong = 2.266667
endLat = 3.9985
endLong = 122.5130

start_point = vg.Point(startLong, startLat)
end_point = vg.Point(endLong, endLat)

# Load the visibility graph file
graph = vg.VisGraph()
graph.load('output\GSHHS_c_L1.graph')

# Calculate the shortest path
path_list = graph.shortest_path(start_point, end_point)

# Save path_list
with open('output\path_list_c', 'wb') as file:
    pickle.dump(path_list, file)

# Calculate the total distance of the initial path
path_distance = 0
prev_point = path_list[0]
for point in path_list[1:]:
    path_distance += haversine((prev_point.y, prev_point.x), (point.y, point.x))
    prev_point = point
path_distance = path_distance*0.539957  # km to nautical miles

print('Path distance in nm: {}'.format(path_distance))