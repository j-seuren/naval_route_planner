import numpy as np
import os
import requests
import subprocess
import sys
import xarray as xr

from datetime import datetime, timedelta
from pathlib import Path


class WindDataRetriever:
    def __init__(self, startDate, nDays=20, DIR=Path('D:/')):
        self.t0 = startDate
        self.t0str = self.t0.strftime('%Y%m%d00')  # Forecast start time; hours should be 00 06 12 or 18
        self.nDays = nDays

        # Dataset file path and grib directory
        f = '{}_nDays{}.npz'.format(self.t0.strftime('%Y%m%d'), self.nDays)
        self.dsFP = DIR / 'data/wind/netcdf_OUT/' / f
        self.gribDir = DIR / 'data/wind/grib_IN' / self.t0.strftime('%Y')  # directory in which to put the output

    def download_grib_files(self, forecast):
        if not os.path.exists(self.gribDir):
            os.makedirs(self.gribDir)

        if forecast:
            # Calls Perl script described in
            # https://www.cpc.ncep.noaa.gov/products/wesley/fast_downloading_grib.html
            execFP = "data_config/get_gfs.pl"
            hr0 = '0'  # first forecast hour wanted
            hr1 = '384'  # last forecast hour wanted
            dHr = '3'  # forecast hour increment (forecast every 3, 6, 12, or 24 hours)
            var = 'UGRD:VGRD'  # list of variables
            level = '10_m_above_ground'  # level
            saveDir = str(self.gribDir.as_posix())
            commandLine = ['perl', execFP, 'data', self.t0str, hr0, hr1, dHr, var, level, saveDir]
            pipe = subprocess.Popen(commandLine, stdin=subprocess.PIPE, stdout=sys.stdout)
            pipe.communicate()
            pipe.stdin.close()
        else:
            pathDict = self.request_save_paths()
            for url, files in pathDict.items():
                for f in files:
                    if not os.path.exists(self.gribDir / f):
                        r = requests.get(url + f)
                        savePath = self.gribDir / f
                        with open(savePath, 'wb') as file:
                            file.write(r.content)

    def request_save_paths(self):
        serverURL = 'https://www.ncei.noaa.gov/data/global-forecast-system/access/historical/analysis/'
        hours = ['0000', '0600', '1200', '1800']
        pathDict = {}
        for dayNr in range(self.nDays):
            thisDay = self.t0 + timedelta(days=dayNr)
            Y = thisDay.strftime('%Y')
            M = thisDay.strftime('%m')
            D = thisDay.strftime('%d')
            requestPath = serverURL + '{0}{1}/{0}{1}{2}/'.format(Y, M, D)
            pathDict[requestPath] = []
            for H in hours:
                f = 'gfsanl_4_{}{}{}_{}_000.grb2'.format(Y, M, D, H)
                pathDict[requestPath].append(f)
        return pathDict

    def grib_to_netcdf(self, forecast):
        # Get a list of all grib files in the grib directory
        pathDict = self.request_save_paths()
        fps = [self.gribDir / f for files in pathDict.values() for f in files]

        print('Converting grib files to netcdf')
        if forecast:
            prep = prep_wind_forecast
            kwargs = {}
        else:
            # If grib file is faulty, i.e., has small size, replace with previous grib file
            for i, fp in enumerate(fps):
                if os.stat(fp).st_size <= 300 and i > 0:
                    print('{} replaced with {}'.format(fps[i].stem, fps[i-1].stem))
                    fps[i] = fps[i-1]
            prep = prep_wind_historical
            kwargs = {'filter_by_keys': {'typeOfLevel': 'sigma'}}

        with xr.open_mfdataset(fps, combine='nested', concat_dim='step', engine='cfgrib', backend_kwargs=kwargs,
                               preprocess=prep) as ds:
            ds = ds.sortby('latitude')
            da = ds.to_array().data
            np.savez_compressed(self.dsFP, da)  # ds.to_netcdf(self.dsFP, encoding={'BN': {'dtype': 'int16'}})
            print('done')

    def get_data(self, forecast):
        if not os.path.exists(self.dsFP):
            print('{} does not exist, downloading GRIB files.'.format(self.dsFP))
            self.download_grib_files(forecast)
            print('Downloaded GRIB files')
            self.grib_to_netcdf(forecast=forecast)
            print('Combined GRIB files and saved to netCDF file')
        print('Loading dataset into memory:'.format(self.dsFP), end=' ')
        with np.load(self.dsFP) as npz:
            print('done')
            return npz['arr_0']


def wind_bft(u, v):
    """Convert wind from metres per second to Beaufort scale
    https://en.wikipedia.org/wiki/Beaufort_scale
    """
    ms = np.sqrt(u * u + v * v)
    bft_bins = np.array([0.5,  1.5,  3.3,  5.5,  7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6])
    return xr.apply_ufunc(np.digitize, ms, bft_bins)


def wind_dir(u, v):
    """ Get direction from which the wind blows, measured clockwise from North line in [0, 360)
    http://weatherclasses.com/uploads/3/6/2/3/36231461/computing_wind_direction_and_speed_from_u_and_v.pdf
    """
    wind_dir_to_rad = xr.ufuncs.arctan2(u, v)
    return 180 + xr.ufuncs.degrees(wind_dir_to_rad)


def prep_wind_historical(ds):
    ds.load()
    ds.attrs = {}
    ds = ds.drop(['valid_time', 'sigma', 'step'])
    ds['BN'] = wind_bft(ds['u'], ds['v'])
    ds['windDir'] = wind_dir(ds['u'], ds['v'])
    ds = ds.drop(['u', 'v', 't', 'pt', 'r', 'w'])
    shifted_longitude = (ds.longitude + 180) % 360 - 180
    return ds.assign_coords(longitude=shifted_longitude).sortby('longitude')


def prep_wind_forecast(ds):
    ds.attrs = {}
    ds = ds.drop(['valid_time', 'heightAboveGround', 'time'])
    ds['BN'] = wind_bft(ds['u10'], ds['v10'])
    ds['windDir'] = wind_dir(ds['u10'], ds['v10'])
    ds = ds.drop(['u10', 'v10'])
    shifted_longitude = (ds.longitude + 180) % 360 - 180
    return ds.assign_coords(longitude=shifted_longitude).sortby('longitude')


if __name__ == '__main__':
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt

    def plot_wind(da):
        # set up the figure
        plt.figure(figsize=(12, 8))
        lon = np.linspace(-180, 179.5, 720)
        lat = np.linspace(-90, 90, 361)
        BNarr = da[0, 0]  # variables, time steps, latitude, longitude

        m = Basemap(projection='merc', urcrnrlat=80, urcrnrlon=180, llcrnrlat=-80, llcrnrlon=-180, resolution='c')

        # convert the lat/lon values to x/y projections.
        x, y = m(*np.meshgrid(lon, lat))
        mappable = m.contourf(x, y, BNarr, cmap=plt.cm.jet)
        m.colorbar(mappable=mappable, location='right')

        # # Add a coastline and axis values.
        m.drawcoastlines()
        # m.fillcontinents()
        m.drawmapboundary()
        # m.drawparallels(np.arange(-90., 90., 30.), labels=[1, 0, 0, 0])
        # m.drawmeridians(np.arange(-180., 180., 60.), labels=[0, 0, 0, 1])
        plt.title('GFS wind speed BN')

    os.chdir('..')
    retriever = WindDataRetriever(nDays=1, startDate=datetime(2018, 10, 10))  #27 20
    _ds = retriever.get_data(forecast=False)
    plot_wind(_ds)
    plt.show()

