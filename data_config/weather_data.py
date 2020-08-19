import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys
import xarray as xr

from datetime import datetime, timedelta
from pathlib import Path


class WeatherDataRetriever:
    def __init__(self,
                 tS=datetime.today() - timedelta(days=1)):
        # Time period
        self.tS = tS
        # start time of the forecast YYYYMMDDHH, note: HH should be 00 06 12 or 18
        self.DATE = self.tS.strftime('%Y%m%d00')

        # Dataset save directory
        self.dsSave = Path('C:/dev/projects/naval_route_planner/data/weather/netcdf_OUT/')
        f = '{}.nc'.format(self.tS.strftime('%Y%m%d'))
        self.dsFP = Path(self.dsSave / f)

        # Grib file save directory
        gribParent = Path('C:/dev/projects/naval_route_planner/data/weather/grib_IN')
        self.gribDir = gribParent / self.DATE  # directory in which to put the output

    def get_gfs_perl(self):
        """Calls Perl script described in
        https://www.cpc.ncep.noaa.gov/products/wesley/fast_downloading_grib.html"""

        if not os.path.exists(self.gribDir):
            os.makedirs(self.gribDir)

        EXEC = "data_config/get_gfs.pl"
        HR0 = '0'  # first forecast hour wanted
        HR1 = '384'  # last forecast hour wanted
        DHR = '3'  # forecast hour increment (forecast every 3, 6, 12, or 24 hours)
        VARS = 'UGRD:VGRD'  # list of variables
        LEVS = '10_m_above_ground'  # level
        print(os.getcwd())
        DIR = str(self.gribDir.as_posix())
        commandLine = ['perl', EXEC, 'data', self.DATE, HR0, HR1, DHR, VARS, LEVS, DIR]
        print(commandLine)
        pipe = subprocess.Popen(commandLine, stdin=subprocess.PIPE, stdout=sys.stdout)
        pipe.communicate()
        pipe.stdin.close()

    def grib_to_netcdf(self):
        files = os.listdir(self.gribDir)
        assert len(files) > 0, 'no files in {}'.format(self.gribDir)
        fps = [self.gribDir / f for f in files]
        print('Converting grib files to netcdf')
        with xr.open_mfdataset(fps, combine='nested', concat_dim='step', engine='cfgrib', preprocess=prep_weather) as ds:
            ds.to_netcdf(self.dsFP, encoding={'BN': {'dtype': 'int16'}})
            print('done')

    def get_ds(self):
        if not os.path.exists(self.dsFP):
            print('Cannot find dataset, try loading GRIB files.')
            if not os.path.exists(self.gribDir):
                print('Cannot find GRIB files, downloading GRIB files.')
                self.get_gfs_perl()
                print('Downloaded GRIB files')
            self.grib_to_netcdf()
            print('Conmbined GRIB files and saved to netCDF file')
        print('Loading dataset into memory:'.format(self.dsFP), end=' ')
        with xr.open_dataset(self.dsFP, engine='h5netcdf') as ds:
            ds = ds.compute()
            print('done')
            return ds


def wind_bft(u, v):
    """Convert wind from metres per second to Beaufort scale"""
    ms = np.sqrt(u * u + v * v)
    bft_bins = np.array([0.5,  1.5,  3.3,  5.5,  7.9, 10.7, 13.8, 17.1, 20.7,
                         24.4, 28.4, 32.6, 32.7])
    return xr.apply_ufunc(np.digitize, ms, bft_bins)


def wind_dir(u, v):
    # Direction in which wind blows
    ms = xr.ufuncs.sqrt(u * u + v * v)
    wind_dir_to_rad = xr.ufuncs.arctan2(u / ms, v / ms)
    wind_dir_to = xr.ufuncs.degrees(wind_dir_to_rad)
    wind_dir_from = wind_dir_to - 180
    return 90 - wind_dir_from


def prep_weather(ds):
    ds.load()
    ds.attrs = {}
    ds = ds.drop(['valid_time', 'heightAboveGround', 'time'])
    ds['BN'] = wind_bft(ds['u10'], ds['v10'])
    ds['windDir'] = wind_dir(ds['u10'], ds['v10'])
    ds = ds.drop(['u10', 'v10'])
    shifted_longitude = (ds.longitude + 180) % 360 - 180
    ds = ds.assign_coords(longitude=shifted_longitude).sortby('longitude')
    return ds


if __name__ == '__main__':
    os.chdir('..')
    retriever = WeatherDataRetriever()
    # retriever.grib_to_netcdf()
    dss = retriever.get_ds()

