import folium
import pickle

# Open shortest path
with open('shortest_path', 'rb') as openfile:
    path = pickle.load(openfile)

# Plot of the path using folium
geopath = [[point.y, point.x] for point in path]
geomap  = folium.Map([0, 0], zoom_start=2)
for point in geopath:
    folium.Marker(point, popup=str(point)).add_to(geomap)
folium.PolyLine(geopath).add_to(geomap)

# Add a Mark on the start and positions in a different color
folium.Marker(geopath[0], popup=str(path[0]), icon=folium.Icon(color='red')).add_to(geomap)
folium.Marker(geopath[-1], popup=str(path[-1]), icon=folium.Icon(color='red')).add_to(geomap)

# Save the interactive plot as a map
output_name = 'example_path_plot.html'
geomap.save(output_name)
print('Output saved to: {}'.format(output_name))