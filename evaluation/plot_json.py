import folium
import json
import os
import numpy as np


class PlotFolium:
    def __init__(self):
        self.geo_map = folium.Map([0, 0], zoom_start=2)
        self.add_once = True

    def plot_interactive_route(self, wps):
        wps_arr = np.array([np.array([el['lat'], el['lng']]) for el in wps.values()])
        if self.add_once:
            start_point, end_point = wps_arr[0], wps_arr[-1]
            # Add a Mark on the start and positions in a different color
            folium.Marker(wps_arr[0], popup=str(start_point), icon=folium.Icon(color='red')).add_to(self.geo_map)
            folium.Marker(wps_arr[-1], popup=str(end_point), icon=folium.Icon(color='red')).add_to(self.geo_map)
            self.add_once = False

        # Plot of the path using folium
        for point in wps_arr[0:-1]:
            folium.Marker(point, popup=str(point)).add_to(self.geo_map)
        folium.PolyLine(wps_arr).add_to(self.geo_map)

    def save_map(self, name):
        # Save the interactive plot as a map
        output_name = 'output/{}.html'.format(name)
        self.geo_map.save(output_name)
        print('Output saved to: {}'.format(output_name))


path_to_load = os.path.abspath('../api/routes_OUT')
file_name = 'test.json'

with open(os.path.join(path_to_load, file_name)) as f:
    file_in = json.load(f)

plotter = PlotFolium()

for path in file_in['canal_paths']:
    for obj_dict in path.values():
        plotter.plot_interactive_route(obj_dict['waypoints'])

plotter.save_map(file_name)