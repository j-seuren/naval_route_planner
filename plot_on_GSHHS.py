import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(9, 13),
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax


extent = [-11.5, 125, -6, 44]  # [left, right, bottom, top)

fig, ax = make_map(projection=ccrs.PlateCarree())
ax.set_extent(extent)

shp = shapereader.Reader('C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/l/GSHHS_l_L1.shp')
for record, geometry in zip(shp.records(), shp.geometries()):
    ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray',
                      edgecolor='black')

# Open solution.route file
with open('output/test_route', 'rb') as file:
    route = pickle.load(file)
route_waypoint_list = [[point.y, point.x] for point in route.waypoints]
prev_point = route_waypoint_list[0]
for point in route_waypoint_list[1:]:
    plt.plot([prev_point[1], point[1]], [prev_point[0], point[0]],
             color='blue', linewidth=1, marker='o', markersize=3, transform=ccrs.PlateCarree())
    prev_point = point
plt.show()
