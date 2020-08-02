import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import fiona
import os
import pickle

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pathlib import Path
from shapely.geometry import (box, Polygon, MultiPolygon,
                              GeometryCollection, shape)

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


def get_split_polygons(res, thr):
    proj_dir = Path(os.getcwd()).parent
    gshhg_dir = Path(proj_dir / 'data/gshhg-shp-2.3.7/GSHHS_shp')
    gshhg_fp = gshhg_dir(gshhg_dir / '{0}/GSHHS_{0}_L1.shp'.format(res))
    geos = [shape(shoreline['geometry']) for shoreline in
            iter(fiona.open(gshhg_fp))]

    # Add Arctic and Antarctic circles as polygon
    arctic_circle = Polygon([(-180, 66), (180, 66), (180, 89), (-180, 89)])
    antarctic_circle = Polygon([(-180, -66), (180, -66), (180, -89), (-180, -89)])
    geos.extend([arctic_circle, antarctic_circle])

    # Compute split polygons
    split_polys = split_polygons(geos, thr)

    # Save result
    result_dir = Path(proj_dir / 'output/split_polygons')
    result_fp = Path(result_dir / 'res_{0}_threshold_{1}'.format(res, thr))
    with open(result_fp, 'wb') as f:
        pickle.dump(split_polys, f)
    print('Saved to: ', result_fp)

    return split_polys


if __name__ == '__main__':
    # Get polygons
    resolution = 'c'
    max_poly_size = 9
    _result = get_split_polygons(resolution, max_poly_size)

    # Plot result
    # plot_geometries(result)
    # plt.show()
