import datetime
import numpy as np
import os
import pandas as pd
import pickle
from scipy import interpolate
import xarray as xr

from ftplib import FTP
from pathlib import Path
from os import path


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
                 t0,
                 nDays,
                 host='eftp.ifremer.fr',
                 user='gg1f3e8',
                 passwd='xG3jZhT9',
                 DIR=Path('D:/')
                 ):
        # Current data period
        self.nDays = nDays
        self.tStart = t0

        # FTP login details
        self.host = host
        self.user = user
        self.passwd = passwd

        # Get dataset file path and download and save directories
        d = DIR / 'data/currents/netcdf_OUT/'
        f = 'Y%d_YDAY%03d_NDAYS%03d.nc' % (t0.year, t0.timetuple().tm_yday, nDays)
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

    def get_kc_data(self, step=1/6, itp=True):
        appendix = '_interpolate' if itp else ''
        fp = self.dataDir / 'KC_processed{}'.format(appendix)

        lons = np.arange(120, 142, step)
        lats = np.arange(23, 37, step)

        if os.path.exists(fp):
            with open(fp, 'rb') as fh:
                array = pickle.load(fh)
            return array, lons, lats

        df = pd.read_csv(self.dataDir / 'caldepth0.csv')
        df['lons'] = df['longitude_degree'] + df['longitude_minute'] / 60.
        df['lats'] = df['latitude_degree'] + df['latitude_minute'] / 60.
        df['u'] = df['velocity'] * np.cos(np.deg2rad(df['direction']))
        df['v'] = df['velocity'] * np.sin(np.deg2rad(df['direction']))

        uDict, vDict, numDict = {}, {}, {}
        for index, row in df.iterrows():
            k = (row['lats'], row['lons'])

            if k in uDict:
                uDict[k] += row['u']
                vDict[k] += row['v']
                numDict[k] += 1.
            else:
                uDict[k] = row['u']
                vDict[k] = row['v']
                numDict[k] = 1.

        xx, yy = np.meshgrid(lons, lats)
        u = np.empty(xx.shape)
        v = np.empty(xx.shape)
        step2 = step / 2.

        print('binning ocean currents:')
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if j % 10 == 0:
                    print('\r', i, j, end='')
                points = {k: v for k, v in numDict.items() if abs(k[0] - lat) <= step2 and abs(k[1] - lon) <= step2}
                nPoints = sum(points.values())

                v[i, j] = np.sum([vDict[k] for k in points]) / nPoints if nPoints > 0 else 0.
                u[i, j] = np.sum([uDict[k] for k in points]) / nPoints if nPoints > 0 else 0.

        array = np.empty([2, len(lats), len(lons)])
        array[0], array[1] = u, v
        print('\r\r done')

        if itp:
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

        with open(fp, 'wb') as fh:
            pickle.dump(array, fh)

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
        m.drawparallels(np.arange(24., 38, 2.), labels=[1, 0, 0, 0], fontsize=8)
        m.drawmeridians(np.arange(120., 144, 2.), labels=[0, 0, 0, 1], fontsize=8)
        vLat = int((max(lats) - min(lats)) * 6)
        vLon = int((max(lons) - min(lons)) * 6)
        print(min(lons), min(lats), max(lons), max(lats))
        uRot, vRot, x, y = m.transform_vector(uin, vin, lons, lats, vLon, vLat, returnxy=True)
        C = m.contourf(x, y, np.hypot(uRot, vRot), cmap='Blues')
        cb = m.colorbar(C, size=0.2, pad=0.05, location='right')
        cb.set_label('Current velocity [kn]', rotation=270, labelpad=15)
        Q = m.quiver(x, y, uRot, vRot, np.hypot(uRot, vRot), cmap='Greys', pivot='mid', width=0.002, headlength=4,
                     scale=90)
        # plt.quiverkey(Q, 0.45, -0.1, 2, r'$2$ knots', labelpos='E')

        # plt.savefig(DIR / 'output/figures' / 'KC_data.pdf', bbox_inches='tight', pad_inches=0.3)
        plt.show()

    data, lons0, lats0 = CurrentDataRetriever(datetime(2014, 10, 28), nDays=6, DIR=DIR).get_kc_data(itp=False)
    _uin, _vin = data[0], data[1]
    lonSlice, latSlice = slice(12, 121), slice(12, 71)
    # lons = np.arange(122, 140, step)
    # lats = np.arange(25, 35, step)

    _extent = (120, 25, 142, 37)

    currents(_uin[latSlice, lonSlice], _vin[latSlice, lonSlice], lons0[lonSlice], lats0[latSlice], _extent)
    currents(_uin, _vin, lons0, lats0, _extent)
