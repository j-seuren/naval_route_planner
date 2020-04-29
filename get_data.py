import shapefile
import pandas as pd


data_fp = 'C:\dev\data'


def shorelines(resolution):
    if resolution == 'crude':
        return shapefile.Reader(data_fp + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1')
    elif resolution == 'low':
        return shapefile.Reader(data_fp + '/gshhg-shp-2.3.7/GSHHS_shp/l/GSHHS_l_L1')
    elif resolution == 'intermediate':
        return shapefile.Reader(data_fp + '/gshhg-shp-2.3.7/GSHHS_shp/i/GSHHS_i_L1')
    elif resolution == 'high':
        return shapefile.Reader(data_fp + '/gshhg-shp-2.3.7/GSHHS_shp/h/GSHHS_h_L1')
    elif resolution == 'full':
        return shapefile.Reader(data_fp + '/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1')


def seca():
    return pd.read_csv(data_fp + '/seca_areas.csv').drop('index', axis=1)