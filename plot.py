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

# Plot of the path
with open('output\path_list_l', 'rb') as file:
    path_list = pickle.load(file)
    print('Path contains {} points.'.format(len(path_list)))
path = [[point.y, point.x] for point in path_list]
prev_point = path[0]
for point in path[1:]:
    print(prev_point)
    plt.plot([prev_point[1], point[1]], [prev_point[0], point[0]], color='blue', linewidth=1, marker='o', markersize=3, transform=ccrs.PlateCarree())
    prev_point = point
plt.show()
