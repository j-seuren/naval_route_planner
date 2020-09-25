import fiona
import numpy as np
import pandas as pd
import pickle

from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon, shape


def create_from_csv(fp, save=True):
    df = pd.read_csv(fp)
    area_names = df.area.unique()
    secas = []
    for area_name in area_names:
        area_df = df.loc[df['area'] == area_name]
        points = [[row['longitude'], row['latitude']] for idx, row in area_df.iterrows()]
        secas.append(Polygon([pt for pt in points]))
    if save:
        fp = 'D:/data/seca_areas_csv'
        with open(fp, 'wb') as file:
            pickle.dump(secas, file)
        print('Saved secas to', fp)

    return secas


def create_from_shp(fp, save=True):
    secas = []
    shapely_objects = [shape(poly_shape['geometry']) for poly_shape in iter(fiona.open(fp))]
    for shapely_object in shapely_objects:
        if shapely_object.geom_type == 'MultiPolygon':
            secas.extend(list(shapely_object))
        elif shapely_object.geom_type == 'Polygon':
            secas.append(shapely_object)
        else:
            raise IOError('Shape is not a polygon.')
    if save:
        fp = 'D:/data/seca_areas_shp'
        with open(fp, 'wb') as file:
            pickle.dump(secas, file)
        print('Saved secas to', fp)

    return secas


def plot_from_list(secas):
    m = Basemap(projection='merc', resolution='l', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)
    m.drawparallels(np.arange(-90., 90., 10.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 180., 10.), labels=[0, 0, 0, 1], fontsize=10)
    m.drawcoastlines()
    m.fillcontinents()

    for poly in secas:
        lon, lat = poly.exterior.xy
        x, y = m(lon, lat)
        m.plot(x, y, 'o-', markersize=2, linewidth=1)


if __name__ == '__main__':
    csv_fp = 'D:/data/seca_areas.csv'
    shp_fp = 'D:/data/eca_reg14_sox_pm/eca_reg14_sox_pm.shp'

    shp_secas = create_from_shp(shp_fp, save=True)
    # csv_secas = create_from_csv(csv_fp, save=False)
