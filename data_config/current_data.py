import datetime
import math
import numpy as np
import os
import pandas as pd
import xarray as xr

from ftplib import FTP
from pathlib import Path
from os import path
from scipy import interpolate
#
#
# def ncdump(nc_fid, verb=True):
#     """
#     ncdump outputs dimensions, variables and their attribute information.
#     The information is similar to that of NCAR's ncdump utility.
#     ncdump requires a valid instance of Dataset.
#
#     Parameters
#     ----------
#     nc_fid : netCDF4.Dataset
#         A netCDF4 dateset object
#     verb : Boolean
#         whether or not nc_attrs, nc_dims, and nc_vars are printed
#
#     Returns
#     -------
#     nc_attrs : list
#         A Python list of the NetCDF file global attributes
#     nc_dims : list
#         A Python list of the NetCDF file dimensions
#     nc_vars : list
#         A Python list of the NetCDF file variables
#     """
#     def print_ncattr(key):
#         """
#         Prints the NetCDF file attributes for a given key
#
#         Parameters
#         ----------
#         key : unicode
#             a valid netCDF4.Dataset.variables key
#         """
#         try:
#             print("\t\ttype:", repr(nc_fid.variables[key].dtype))
#             for ncattr in nc_fid.variables[key].ncattrs():
#                 print('\t\t%s:' % ncattr,
#                       repr(nc_fid.variables[key].getncattr(ncattr)))
#         except KeyError:
#             print("\t\tWARNING: %s does not contain variable attributes" % key)
#
#     # NetCDF global attributes
#     nc_attrs = nc_fid.ncattrs()
#     if verb:
#         print("NetCDF Global Attributes:")
#         for nc_attr in nc_attrs:
#             print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
#     nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
#     # Dimension shape information.
#     if verb:
#         print("NetCDF dimension information:")
#         for dim in nc_dims:
#             print("\tName:", dim)
#             print("\t\tsize:", len(nc_fid.dimensions[dim]))
#             print_ncattr(dim)
#     # Variable information.
#     nc_vars = [var for var in nc_fid.variables]  # list of nc variables
#     if verb:
#         print("NetCDF variable information:")
#         for var in nc_vars:
#             if var not in nc_dims:
#                 print('\tName:', var)
#                 print("\t\tdimensions:", nc_fid.variables[var].dimensions)
#                 print("\t\tsize:", nc_fid.variables[var].size)
#                 print_ncattr(var)
#     return nc_attrs, nc_dims, nc_vars
#


class CurrentDataRetriever:
    def __init__(self,
                 t_s,
                 nDays,
                 host='eftp.ifremer.fr',
                 user='gg1f3e8',
                 passwd='xG3jZhT9',
                 DIR=Path('D:/')
                 ):
        # Current data period
        self.nDays = nDays
        self.tStart = t_s

        # FTP login details
        self.host = host
        self.user = user
        self.passwd = passwd

        # Get dataset file path and download and save directories
        d = DIR / 'data/currents/netcdf_OUT/'
        f = 'Y%d_YDAY%03d_NDAYS%03d.nc' % (t_s.year, t_s.timetuple().tm_yday, nDays)
        self.dataFP = Path(d / f)
        self.dsFTP = Path('/data/globcurrent/v3.0/global_025_deg/total_hs')
        self.dataDir = DIR / 'data/currents/'
        self.dsSave = self.dataDir / 'netcdf_IN'

    def get_data(self):
        # First check if combined netCDF file exists
        if os.path.exists(self.dataFP):
            print('Loading current data into memory:'.format(self.dataFP), end=' ')
            with xr.open_dataset(self.dataFP) as ds:
                print('done')
                return ds.to_array().data
        else:  # Create combined netCDF file from separate (to-be) downloaded netCDF files
            # Download non-existent netCDF files
            saveFPs = []
            with FTP(host=self.host, user=self.user, passwd=self.passwd) as ftp:
                for day in range(self.nDays):
                    # Get path appendix
                    t = self.tStart + datetime.timedelta(days=day)
                    y, yday = t.year, t.timetuple().tm_yday
                    path_appendix = Path('%d/%03d' % (y, yday))

                    # Set FTP current working directory and save directory
                    ftp.cwd(Path(self.dsFTP / path_appendix).as_posix())
                    saveDir = Path(self.dsSave / path_appendix)
                    if not path.exists(saveDir):
                        os.makedirs(saveDir)

                    # Append files to file_list
                    files = [f for f in ftp.nlst() if '0000' in f]
                    _saveFPs = [Path(saveDir / f).as_posix() for f in files]
                    saveFPs.extend(_saveFPs)
                    for saveFP, loadFP in zip(_saveFPs, files):
                        if not path.isfile(saveFP):  # If files does not exist, download from FTP server
                            with open(saveFP, 'wb') as f:  # Download file to fp_save
                                ftp.retrbinary('RETR %s' % loadFP, f.write)

            # Open downloaded netCDF files, combine and store locally
            print('Combining %d netCDF files:' % (8 * self.nDays), end=' ')
            with xr.open_mfdataset(saveFPs, parallel=True, combine='by_coords', preprocess=ms_to_knots) as ds:
                print("done\nStoring data array to '{}' :".format(self.dataFP), end=' ')
                ds.to_netcdf(self.dataFP)
                print('done')
                return ds.to_array().data

    def get_kc_data(self):
        # computing the drift along each arc
        df = pd.read_csv(self.dataDir / 'caldepth0.csv')
        uDict, vDict, numDict = {}, {}, {}
        lons, lats = [], []
        for i in range(len(df)):
            lon = df['longitude_degree'].iloc[i] + df['longitude_minute'].iloc[i] / 60.0
            if lon not in lons:
                lons.append(lon)
            lat = df['latitude_degree'].iloc[i] + df['latitude_minute'].iloc[i] / 60.0
            if lat not in lats:
                lats.append(lat)

            k = (lon, lat)
            if k in uDict:
                uDict[k] += df['velocity'].iloc[i] * math.cos(math.pi * df['direction'].iloc[i] / 180.0)
                vDict[k] += df['velocity'].iloc[i] * math.sin(math.pi * df['direction'].iloc[i] / 180.0)
                numDict[k] += 1.0
            else:
                uDict[k] = df['velocity'].iloc[i] * math.cos(math.pi * df['direction'].iloc[i] / 180.0)
                vDict[k] = df['velocity'].iloc[i] * math.sin(math.pi * df['direction'].iloc[i] / 180.0)
                numDict[k] = 1.0

        lons, lats = np.array(sorted(lons)), np.array(sorted(lats))

        array = np.empty([2, len(lats), len(lons)])
        array[:] = np.nan
        for k, u in uDict.items():
            lon, lat = k
            lonIdx = np.where(lons == lon)
            latIdx = np.where(lats == lat)
            assert len(lonIdx) == 1 and len(latIdx) == 1
            array[0, latIdx, lonIdx] = u / numDict[k]
            uDict[k] = u / numDict[k]

        for k, v in vDict.items():
            lon, lat = k
            lonIdx = np.where(lons == lon)
            latIdx = np.where(lats == lat)
            assert len(lonIdx) == 1 and len(latIdx) == 1
            array[1, latIdx, lonIdx] = v / numDict[k]
            vDict[k] = v / numDict[k]

        print(np.min(array), np.max(array))
        print(np.min(array[0]), np.max(array[0]))
        print(np.min(array[1]), np.max(array[1]))

        array = np.ma.masked_invalid(array)
        for i in range(2):
            arr = array[i]
            x = np.arange(0, arr.shape[1])
            y = np.arange(0, arr.shape[0])
            xx, yy = np.meshgrid(x, y)
            # get only the valid values
            x1 = xx[~arr.mask]
            y1 = yy[~arr.mask]
            newArr = arr[~arr.mask]
            array[i] = interpolate.griddata((x1, y1), newArr.ravel(), (xx, yy), fill_value=0.)

        return array, lons, lats


def ms_to_knots(ds):
    ds.attrs = {}
    arr2d = np.float32(np.ones((720, 1440)) * 1.94384)
    ds['u_knot'] = arr2d * ds['eastward_eulerian_current_velocity']
    ds['v_knot'] = arr2d * ds['northward_eulerian_current_velocity']
    ds = ds.drop_vars(['eastward_eulerian_current_velocity',
                       'eastward_eulerian_current_velocity_error',
                       'northward_eulerian_current_velocity',
                       'northward_eulerian_current_velocity_error'])
    return ds


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from datetime import datetime
    from mpl_toolkits.basemap import Basemap
    from pathlib import Path
    DIR = Path('D:/')


    def currents(uin, vin, lons, lats, extent):
        m = Basemap(projection='merc', resolution='l', llcrnrlon=extent[0], llcrnrlat=extent[1], urcrnrlon=extent[2],
                    urcrnrlat=extent[3])
        m.drawmapboundary()
        m.drawcoastlines()
        m.fillcontinents()
        m.drawparallels(np.arange(24., 36, 2.), labels=[1, 0, 0, 0], fontsize=8)
        m.drawmeridians(np.arange(120., 142, 2.), labels=[0, 0, 0, 1], fontsize=8)

        vLat = int((max(lats) - min(lats)) * 4)
        vLon = int((max(lons) - min(lons)) * 4)
        uRot, vRot, x, y = m.transform_vector(uin, vin, lons, lats, vLon, vLat, returnxy=True)
        m.quiver(x, y, uRot, vRot, np.hypot(uRot, vRot), pivot='mid', width=0.002, headlength=4, cmap='PuBu', scale=90)

        plt.show()

    data, lons0, lats0 = CurrentDataRetriever(datetime(2014, 10, 28), nDays=6, DIR=DIR).get_kc_data()
    _uin, _vin = data[0], data[1]

    _extent = (120, 23, 142, 37)

    currents(_uin, _vin, lons0, lats0, _extent)
