import shapefile
import pandas as pd
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection

data_fp = 'C:/dev/data'


def shore_areas():
    # Get the shoreline shapes from the shape file
    shapes = shapefile.Reader(data_fp + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1').shapes()
    print('The shoreline shapefile contains {} shapes.'.format(len(shapes)))

    # # Create a DataFrame of polygons
    areas = pd.DataFrame(columns=['polygon_index', 'longitude', 'latitude'])

    shape_count = 1
    for shape in shapes:
        for point in shape.points:
            areas = areas.append(
                {'polygon_index': shape_count, 'longitude': point[0], 'latitude': point[1]}, ignore_index=True)

        shape_count += 1
    return areas


def split_large_polygon(geometry, max_width_height, count=0):
    """Split a Polygon into two parts across it's shortest dimension"""
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    count = 0
    if max(width, height) <= max_width_height or count == 250:
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
                result.extend(split_large_polygon(e, max_width_height, count + 1))
    if count > 0:
        return result

    # convert multi part into single part
    final_result = []
    for g in result:
        if isinstance(g, MultiPolygon):
            final_result.extend(g)
        else:
            final_result.append(g)
    return final_result

