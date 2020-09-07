import numpy as np
import os
import requests
import subprocess
import sys
import xarray as xr

from datetime import datetime, timedelta
from pathlib import Path

# Plotting
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


class WeatherDataRetriever:
    def __init__(self,
                 startDate=datetime.today() - timedelta(days=1),
                 nDays=20):
        # Start day
        self.startDate = startDate
        # start time of the forecast YYYYMMDDHH, note: HH should be 00 06 12 or 18
        self.DATE = self.startDate.strftime('%Y%m%d00')
        self.nDays = nDays

        # Dataset save directory
        self.dsSave = Path('C:/dev/projects/naval_route_planner/data/weather/netcdf_OUT/')
        f = '{}_nDays{}.nc'.format(self.startDate.strftime('%Y%m%d'), nDays)
        self.dsFP = Path(self.dsSave / f)

        # Grib file save directory
        gribParent = Path('C:/dev/projects/naval_route_planner/data/weather/grib_IN')
        self.gribDir = gribParent / self.DATE  # directory in which to put the output

    def get_gfs_historical(self):
        if not os.path.exists(self.gribDir):
            os.makedirs(self.gribDir)
        serverURL = 'https://www.ncei.noaa.gov/data/global-forecast-system/access/historical/analysis/'

        hours = ['0000', '0600', '1200', '1800']
        for dayNr in range(self.nDays):
            thisDay = self.startDate + timedelta(days=dayNr)
            Y = thisDay.strftime('%Y')
            M = thisDay.strftime('%m')
            D = thisDay.strftime('%d')

            url = serverURL + '{0}{1}/{0}{1}{2}/'.format(Y, M, D)

            for H in hours:
                filePath = 'gfsanl_4_{}{}{}_{}_000.grb2'.format(Y, M, D, H)
                savePath = self.gribDir / filePath
                if os.path.exists(savePath):
                    continue
                r = requests.get(url + filePath)
                with open(savePath, 'wb') as f:
                    f.write(r.content)

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

    def grib_to_netcdf(self, historical):
        dirfiles = [f for f in os.listdir(self.gribDir) if '.idx' not in f]
        assert len(dirfiles) > 0, 'no files in {}'.format(self.gribDir)

        print('Converting grib files to netcdf')
        if historical:
            dateStrings = [(self.startDate + timedelta(days=dayNr)).strftime('%Y%m%d') for dayNr in range(self.nDays)]
            fps = [self.gribDir / f for f in dirfiles if any(date in f for date in dateStrings)]
            with xr.open_mfdataset(fps, combine='nested', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'sigma'}},
                                   concat_dim='step', engine='cfgrib', preprocess=prep_weather_historical
                                   ) as ds:
                ds.to_netcdf(self.dsFP, encoding={'BN': {'dtype': 'int16'}})

        else:
            fps = [self.gribDir / f for f in dirfiles]
            with xr.open_mfdataset(fps, combine='nested',
                                   concat_dim='step', engine='cfgrib', preprocess=prep_weather) as ds:
                ds.to_netcdf(self.dsFP, encoding={'BN': {'dtype': 'int16'}})
                print('done')

    def get_ds(self, historical):
        if not os.path.exists(self.dsFP):
            print('{} does not exist, try loading GRIB files.'.format(self.dsFP))
            if not os.path.exists(self.gribDir):
                print('{} does not exist, downloading GRIB files.'.format(self.gribDir))
                if historical:
                    self.get_gfs_historical()
                else:
                    self.get_gfs_perl()
                print('Downloaded GRIB files')
            self.grib_to_netcdf(historical=historical)
            print('Combined GRIB files and saved to netCDF file')
        print('Loading dataset into memory:'.format(self.dsFP), end=' ')
        with xr.open_dataset(self.dsFP, engine='h5netcdf') as ds:
            ds = ds.compute()
            print(ds)
            print('done')
            return ds


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


def prep_weather_historical(ds):
    ds.load()
    ds.attrs = {}
    ds = ds.drop(['valid_time', 'sigma', 'step'])
    ds['BN'] = wind_bft(ds['u'], ds['v'])
    ds['windDir'] = wind_dir(ds['u'], ds['v'])
    ds = ds.drop(['u', 'v', 't', 'pt', 'r', 'w'])
    shifted_longitude = (ds.longitude + 180) % 360 - 180
    return ds.assign_coords(longitude=shifted_longitude).sortby('longitude')


def prep_weather(ds):
    ds.attrs = {}
    ds = ds.drop(['valid_time', 'heightAboveGround', 'time'])
    ds['BN'] = wind_bft(ds['u10'], ds['v10'])
    ds['windDir'] = wind_dir(ds['u10'], ds['v10'])
    ds = ds.drop(['u10', 'v10'])
    shifted_longitude = (ds.longitude + 180) % 360 - 180
    return ds.assign_coords(longitude=shifted_longitude).sortby('longitude')


def plot_wind(ds):
    # set up the figure
    plt.figure(figsize=(12, 8))

    lat = ds['latitude'].data[1:-1]
    lon = ds['longitude'].data
    data = ds['BN'][1, 1:-1, :].data

    m = Basemap(projection='merc', urcrnrlat=80, urcrnrlon=180, llcrnrlat=-80, llcrnrlon=-180, resolution='c')

    # convert the lat/lon values to x/y projections.
    x, y = m(*np.meshgrid(lon, lat))

    # plot the field using the fast pcolormesh routine
    # set the colormap to jet.
    m.pcolormesh(x, y, data, shading='flat', cmap=plt.cm.jet)
    m.colorbar(location='right')

    # Add a coastline and axis values.
    m.drawcoastlines()
    m.fillcontinents()
    m.drawmapboundary()
    m.drawparallels(np.arange(-90., 90., 30.), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180., 180., 60.), labels=[0, 0, 0, 1])

    # Add a colorbar and title, and then show the plot.

    plt.title('GFS wind speed BN')


if __name__ == '__main__':
    os.chdir('..')
    retriever = WeatherDataRetriever(nDays=1, startDate=datetime(2011, 5, 30))
    # retriever.grib_to_netcdf()
    _ds = retriever.get_ds(historical=True)
    plot_wind(_ds)
    plt.show()

