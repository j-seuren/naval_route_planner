import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(9, 13),
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax


extent = [-60, 0, 44, 50]  # [left, right, bottom, top)

fig, ax = make_map(projection=ccrs.PlateCarree())
ax.set_extent(extent)

shp = shapereader.Reader('C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/l/GSHHS_l_L1.shp')
for record, geometry in zip(shp.records(), shp.geometries()):
    ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray',
                      edgecolor='black')

# # Open solution.route file
# with open('output/population01', 'rb') as file:
#     route = pickle.load(file)
#
# route = route[0]
#
# route_waypoint_list = [[point.y, point.x] for point in route.waypoints]
# prev_point = route_waypoint_list[0]
# for point in route_waypoint_list[1:]:
#     plt.plot([prev_point[1], point[1]], [prev_point[0], point[0]],
#              color='blue', linewidth=1, marker='o', markersize=3, transform=ccrs.PlateCarree())
#     prev_point = point

# Parents plotting
with open('../output/pareto_solutions01', 'rb') as file:
    parents = pickle.load(file)
parent = parents[0]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
          'burlywood', 'lime', 'lightsteelblue', 'slateblue', 'gainsboro', 'dimgray', 'lightsalmon']
for c, parent in enumerate(parents[:10]):
    parent_waypoint_list = [[point.y, point.x] for point in parent.waypoints]
    prev_point = parent_waypoint_list[0]

    for point in parent_waypoint_list[1:]:
        plt.plot([prev_point[1], point[1]], [prev_point[0], point[0]], color=colors[c], linewidth=1, marker='o',
                 markersize=3, transform=ccrs.PlateCarree())
        prev_point = point


# Plot visibility route
with open('../output/visibility_route', 'rb') as file:
    visibility_route = pickle.load(file)
vis_route_wps = [[point.y, point.x] for point in visibility_route.waypoints]
prev_point = vis_route_wps[0]

for point in vis_route_wps[1:]:
    plt.plot([prev_point[1], point[1]], [prev_point[0], point[0]], linewidth=1, marker='o',
             markersize=3, transform=ccrs.PlateCarree())
    prev_point = point

plt.show()