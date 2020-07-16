from shapely.geometry import (box, Polygon, MultiPolygon,
                              GeometryCollection, shape)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import fiona
import os
import pickle


def split_polygon(geo, threshold, cnt=0):
    """Split a Polygon into two parts across it's shortest dimension"""
    (minx, miny, maxx, maxy) = geo.bounds
    width = maxx - minx
    height = maxy - miny
    if max(width, height) <= threshold or cnt == 250:
        # either the polygon is smaller than the threshold, or the maximum
        # number of recursions has been reached
        return [geo]
    if height >= width:
        # split left to right
        a = box(minx, miny, maxx, miny + height/2)
        b = box(minx, miny + height / 2, maxx, maxy)
    else:
        # split top to bottom
        a = box(minx, miny, minx+width/2, maxy)
        b = box(minx + width / 2, miny, maxx, maxy)
    result = []
    for d in (a, b,):
        c = geo.intersection(d)
        if not isinstance(c, GeometryCollection):
            c = [c]
        for e in c:
            if isinstance(e, (Polygon, MultiPolygon)):
                result.extend(split_polygon(e, threshold, cnt + 1))
    if cnt > 0:
        return result
    # convert multi part into single part
    final_result = []
    for g in result:
        if isinstance(g, MultiPolygon):
            final_result.extend(g)
        else:
            final_result.append(g)
    return final_result


def split_polygons(geometries, threshold):
    split_polys = []
    for polygon in geometries:
        split_polys.extend(split_polygon(polygon, threshold))
    return split_polys


def plot_geometries(geometries):
    fig, ax = plt.subplots(figsize=(9, 13),
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax.add_geometries(geometries, ccrs.PlateCarree(), facecolor='lightgray',
                      edgecolor='black')


def get_split_polygons(res, threshold):
    gshhg_dir = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp'
    gshhg_fp = '{0}/GSHHS_{0}_L1.shp'.format(res)
    geos = [shape(shoreline['geometry']) for shoreline in
            iter(fiona.open(os.path.join(gshhg_dir, gshhg_fp)))]

    # Compute split polygons
    split_polys = split_polygons(geos, threshold)

    # Save result
    result_dir = 'output/split_polygons'
    result_fn = 'res_{0}_threshold_{1}'.format(res, threshold)
    result_fp = os.path.join(result_dir, result_fn)
    try:
        with open(result_fp, 'wb') as f:
            pickle.dump(split_polys, f)
    except FileNotFoundError:
        os.mkdir('output/split_polygons')
        with open(result_fp, 'wb') as f:
            pickle.dump(split_polys, f)
    print('Saved to: ', result_fp)

    return split_polys


if __name__ == '__main__':
    resolution = 'c'
    max_poly_size = 9

    # Get polygons
    _gshhg_dir = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp'
    _gshhg_fp = '{0}/GSHHS_{0}_L1.shp'.format(resolution)
    _geos = [shape(shoreline['geometry']) for shoreline in
             iter(fiona.open(os.path.join(_gshhg_dir, _gshhg_fp)))]
    # Compute split polygons
    _result = split_polygons(_geos, threshold=max_poly_size)

    # Save result
    _result_dir = 'output/split_polygons'
    _result_fn = 'res_{0}_threshold_{1}'.format(resolution, max_poly_size)
    _result_fp = os.path.join(_result_dir, _result_fn)
    try:
        with open(_result_fp, 'wb') as _f:
            pickle.dump(_result, _f)
    except FileNotFoundError:
        os.mkdir('output/split_polygons')
        with open(_result_fp, 'wb') as _f:
            pickle.dump(_result, _f)
    print('Saved to: ', _result_fp)

    # Plot result
    # plot_geometries(result)
    # plt.show()
