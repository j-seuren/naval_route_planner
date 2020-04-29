import pyvisgraph as vg
import solution
import folium
from haversine import haversine
import pickle

# Example points
startLat = 52.1             # Scheveningen
startLong = 2.266667
endLat = 3.9985             # Kingston upon Hull
endLong = 122.5130

start_point = vg.Point(startLong, startLat)
end_point = vg.Point(endLong, endLat)

# Load the visibility graph file
graph = vg.VisGraph()
graph.load('output/GSHHS_c_L1.graph')

# Calculate the shortest path
shortest_path = graph.shortest_path(start_point, end_point)

# Save shortest path
with open('shortest_path', 'wb') as file:
    pickle.dump(shortest_path, file)

# Calculate the total distance of the shortest path in km
path_distance = 0
prev_point = shortest_path[0]
for point in shortest_path[1:]:
    path_distance += haversine((prev_point.y, prev_point.x), (point.y, point.x))
    prev_point = point
path_distance = path_distance*0.539957  # km to nautical miles

print('Shortest path distance: {}'.format(path_distance))

# Plot of the path using folium
geopath = [[point.y, point.x] for point in shortest_path]
geomap  = folium.Map([0, 0], zoom_start=2)
for point in geopath:
    folium.Marker(point, popup=str(point)).add_to(geomap)
folium.PolyLine(geopath).add_to(geomap)

# Add a Mark on the start and positions in a different color
folium.Marker(geopath[0], popup=str(start_point), icon=folium.Icon(color='red')).add_to(geomap)
folium.Marker(geopath[-1], popup=str(end_point), icon=folium.Icon(color='red')).add_to(geomap)

# Save the interactive plot as a map
output_name = 'example_shortest_path_plot.html'
geomap.save(output_name)
print('Output saved to: {}'.format(output_name))
print(shortest_path)