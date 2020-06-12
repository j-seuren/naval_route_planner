import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from shapely.geometry import LineString


def plot_edge(edge):
    def make_map(projection=ccrs.PlateCarree()):
        fig_f, ax_f = plt.subplots(figsize=(9, 13),
                               subplot_kw=dict(projection=projection))
        gl = ax.gridlines(draw_labels=True)
        gl.xlabels_top = gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        return fig_f, ax_f

    extent = [min(edge.v.lon, edge.w.lon) - 1, max(edge.v.lon, edge.w.lon) + 1, min(edge.v.lat, edge.w.lat) - 1, max(edge.v.lat, edge.w.lat) + 1]  # [left, right, bottom, top)

    fig, ax = make_map(projection=ccrs.PlateCarree())
    ax.set_extent(extent)

    polygons = shapereader.Reader('C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/l/GSHHS_l_L1.shp')

    for record, geometry in zip(polygons.records(), polygons.geometries()):
        ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray', edgecolor='black')

    point1 = [edge.v.lat, edge.v.lon]
    point2 = [edge.w.lat, edge.w.lon]

    plt.plot([point1[1], point2[1]], [point1[0], point2[0]], color='blue', linewidth=1, marker='o', markersize=3, transform=ccrs.PlateCarree())

    edge_line = LineString([(edge.v.lon, edge.v.lat), (edge.w.lon, edge.w.lat)])
    area_radius = 0.5*edge_line.length
    print(area_radius)
    area_input = edge_line.centroid.buffer(area_radius)

    x_area, y_area = area_input.exterior.coords.xy
    ax.plot(x_area, y_area)

    plt.show()
