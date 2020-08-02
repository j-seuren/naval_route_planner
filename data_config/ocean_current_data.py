import netCDF4
import datetime
import numpy as np
import os
import xarray as xr

from ftplib import FTP
from pathlib import Path
from os import path


def ncdump(nc_fid, verb=True):
    """
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    """
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


class CurrentDataRetriever:
    def __init__(self,
                 t_s,
                 n_days,
                 host='eftp.ifremer.fr',
                 user='gg1f3e8',
                 passwd='xG3jZhT9'
                 ):
        # Current data period
        self.n_days = n_days
        self.t_s = t_s

        # FTP login details
        self.host = host
        self.user = user
        self.passwd = passwd

        # Get dataset file path and download and save directories
        d = Path(os.getcwd() + '/data/currents/netcdf_OUT/')
        print(d)
        f = 'Y%d_YDAY%03d_NDAYS%03d.nc' % (t_s.year,
                                           t_s.timetuple().tm_yday,
                                           n_days)
        self.ds_fp = Path(d / f)
        self.d_ftp = Path('/data/globcurrent/v3.0/global_025_deg/total_hs')
        self.d_save = Path('data/currents/netcdf_IN')

    def store_combined_nc(self):
        # Get output dataset file path
        if path.isfile(self.ds_fp):
            print('Combined netCDF file exists locally')
            return

        file_list = self.get_file_list()

        # If not all files exist, download non-existent files from FTP server
        if not all([path.isfile(f) for f in file_list]):
            self.download_nc_files()

        # Open downloaded netCDF files, combine and store locally
        print('Opening %d netCDF files:' % (8 * self.n_days), end=' ')
        with xr.open_mfdataset(file_list,
                               parallel=True,
                               combine='by_coords',
                               preprocess=to_kts
                               ) as ds:
            print('done')
            print("Storing combined netCDF to '{}' :".format(self.ds_fp), end=' ')
            print(os.getcwd())
            print(self.ds_fp)
            ds.to_netcdf(self.ds_fp)
            print('done')

    def get_file_list(self):
        # Get path list
        file_list = []
        with FTP(host=self.host, user=self.user, passwd=self.passwd) as ftp:
            for day in range(self.n_days):
                # Get path appendix
                t = self.t_s + datetime.timedelta(days=day)
                y, yday = t.year, t.timetuple().tm_yday
                path_appendix = Path('%d/%03d' % (y, yday))

                # Set FTP current working directory and save directory
                ftp.cwd(Path(self.d_ftp / path_appendix).as_posix())
                d_save = Path(self.d_save / path_appendix)

                # Append files to file_list
                file_list.extend([d_save / f for f in ftp.nlst() if '0000' in f])

        return file_list

    def download_nc_files(self):
        print('Downloading ocean current data from FTP server:', end=' ')

        # Access FTP server
        with FTP(host=self.host, user=self.user, passwd=self.passwd) as ftp:
            for day in range(self.n_days):
                # Get path appendix
                t = self.t_s + datetime.timedelta(days=day)
                y, yday = t.year, t.timetuple().tm_yday
                path_appendix = Path('%d/%03d' % (y, yday))

                # Set FTP current working directory and save directory
                ftp.cwd(Path(self.d_ftp / path_appendix).as_posix())
                d_save = Path(self.d_save / path_appendix)
                if not path.exists(d_save):
                    os.mkdir(d_save)

                # Get files FTP cwd
                files = [f for f in ftp.nlst() if '0000' in f]
                for f in files:
                    fp_save = d_save / f
                    if path.isfile(fp_save):
                        continue

                    # Download file to fp_save
                    with open(fp_save, 'wb') as fh:
                        ftp.retrbinary('RETR %s' % f, fh.write)
        print('done')


def to_kts(ds):
    ds.attrs = {}
    arr2d = np.float32(np.ones((720, 1440)) * 1.94384)
    ds['u_knot'] = arr2d * ds['eastward_eulerian_current_velocity']
    ds['v_knot'] = arr2d * ds['northward_eulerian_current_velocity']
    ds = ds.drop_vars(['eastward_eulerian_current_velocity',
                       'eastward_eulerian_current_velocity_error',
                       'northward_eulerian_current_velocity',
                       'northward_eulerian_current_velocity_error'])
    return ds


if __name__ == "__main__":
    _start_date = datetime.datetime(2016, 1, 1)
    _n_days = 50
    store_combined_nc(_start_date, _n_days)

    # Inspect netCDF file
    _nc_dir = Path('currents/netcdf_IN/')
    _y, _yday = _start_date.year, '%03d' % _start_date.timetuple().tm_yday
    _m, _d = '%02d' % _start_date.month, '%02d' % _start_date.day
    _nc_path = Path(_nc_dir / '{}/{}'.format(_y, _yday))
    _nc_fn = Path('{}{}{}000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v03.0-fv01.0.nc'.format(_y, _m, _d))
    _nc_fp = _nc_path / _nc_fn
    _nc_fid = netCDF4.Dataset(_nc_fp, 'r')
    _nc_attrs, _nc_dims, _nc_vars = ncdump(_nc_fid)
