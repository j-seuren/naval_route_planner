import shapefile
import pandas as pd


data_fp = 'C:/dev/data'


def shoreline_shapes(resolution):
    if resolution == 'crude':
        return shapefile.Reader(data_fp + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1').shapes()
    elif resolution == 'low':
        return shapefile.Reader(data_fp + '/gshhg-shp-2.3.7/GSHHS_shp/l/GSHHS_l_L1').shapes()
    elif resolution == 'intermediate':
        return shapefile.Reader(data_fp + '/gshhg-shp-2.3.7/GSHHS_shp/i/GSHHS_i_L1').shapes()
    elif resolution == 'high':
        return shapefile.Reader(data_fp + '/gshhg-shp-2.3.7/GSHHS_shp/h/GSHHS_h_L1').shapes()
    elif resolution == 'full':
        return shapefile.Reader(data_fp + '/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1').shapes()


def shore_areas(resolution):
    # Get the shoreline shapes from the shape file
    shapes = shoreline_shapes(resolution=resolution)
    print('The shoreline shapefile contains {} shapes.'.format(len(shapes)))

    # # Create a DataFrame of polygons
    areas = pd.DataFrame(columns=['polygon_index', 'longitude', 'latitude'])
    # shore_areas.append({'polygon_index': 0, 'latitude': 1, 'longitude': 2}, ignore_index=True)

    shape_count = 1
    for shape in shapes:
        for point in shape.points:
            areas = areas.append(
                {'polygon_index': shape_count, 'longitude': point[0], 'latitude': point[1]}, ignore_index=True)

        shape_count += 1
    return areas


def seca():
    return pd.read_csv(data_fp + '/seca_areas.csv').drop('index', axis=1)


def speed_table(ship_name):
    return pd.read_excel(data_fp + '/speed_table.xlsx', sheet_name=ship_name)

