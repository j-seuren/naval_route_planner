import folium
import pickle

# Open shortest path
directory = 'output/'
file = 'route2'


with open(directory + file, 'rb') as openfile:
    route = pickle.load(openfile)

# Plot of the path using folium
path_points = [[waypoint.y, waypoint.x] for waypoint in route.waypoints]
geomap = folium.Map([0, 0], zoom_start=2)
for point in path_points:
    folium.Marker(point, popup=str(point)).add_to(geomap)
folium.PolyLine(path_points).add_to(geomap)

# Add a Mark on the start and positions in a different color
folium.Marker(path_points[0], popup=str(route.waypoints[0]), icon=folium.Icon(color='red')).add_to(geomap)
folium.Marker(path_points[-1], popup=str(route.waypoints[-1]), icon=folium.Icon(color='red')).add_to(geomap)

# Save the interactive plot as a map
output_name = directory + 'route2.html'
geomap.save(output_name)
print('Output saved to: {}'.format(output_name))