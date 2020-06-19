from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection, shape
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import fiona
import os
import pickle


def split_polygon(geometry, threshold, count=0):
    """Split a Polygon into two parts across it's shortest dimension"""
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if max(width, height) <= threshold or count == 250:
        # either the polygon is smaller than the threshold, or the maximum
        # number of recursions has been reached
        return [geometry]
    if height >= width:
        # split left to right
        a = box(bounds[0], bounds[1], bounds[2], bounds[1]+height/2)
        b = box(bounds[0], bounds[1]+height/2, bounds[2], bounds[3])
    else:
        # split top to bottom
        a = box(bounds[0], bounds[1], bounds[0]+width/2, bounds[3])
        b = box(bounds[0]+width/2, bounds[1], bounds[2], bounds[3])
    result = []
    for d in (a, b,):
        c = geometry.intersection(d)
        if not isinstance(c, GeometryCollection):
            c = [c]
        for e in c:
            if isinstance(e, (Polygon, MultiPolygon)):
                result.extend(split_polygon(e, threshold, count+1))
    if count > 0:
        return result
    # convert multipart into singlepart
    final_result = []
    for g in result:
        if isinstance(g, MultiPolygon):
            final_result.extend(g)
        else:
            final_result.append(g)
    return final_result


def split_polygons(geometries, threshold):
    split_polygons = []
    for polygon in geometries:
        split_polygons.extend(split_polygon(polygon, threshold))
    return split_polygons


def plot_geometries(geometries):
    fig, ax = plt.subplots(figsize=(9, 13),
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax.add_geometries(geometries, ccrs.PlateCarree(), facecolor='lightgray', edgecolor='black')


if __name__ == '__main__':
    resolution = 'c'
    max_poly_size = 9

    # Get polygons
    polygons = [shape(shoreline['geometry']) for shoreline in
                iter(fiona.open('C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/{0}/GSHHS_{0}_L1.shp'.format(resolution)))]

    # Compute split polygons
    result = split_polygons(polygons, threshold=max_poly_size)

    # Save result
    output_file_name = 'split_polygons/res_{0}_treshold_{1}'.format(resolution, max_poly_size)
    try:
        with open('output/' + output_file_name, 'wb') as file:
            pickle.dump(result, file)
    except FileNotFoundError:
        os.mkdir('output/split_polygons')
        with open('output/' + output_file_name, 'wb') as file:
            pickle.dump(result, file)
    print('Saved to: output/{}'.format(output_file_name))


    # Plot result
    # plot_geometries(result)
    # plt.show()