import netCDF4
import datetime
import ftplib
import numpy as np
import os
import xarray as xr


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


def load(t_start, n_days):
    t_start.replace(second=0, microsecond=0, minute=0, hour=0)
    t_y, t_yday = t_start.year, '%03d' % t_start.timetuple().tm_yday
    n_days_str = '%03d' % n_days
    nc_dir = 'current_netCDF/ftp_files/'
    ds_dir = 'current_netCDF/chunked_data/'
    ds_fn = 'Y{}_YDAY{}_NDAYS{}.nc'.format(t_y, t_yday, n_days_str)
    ds_fp = os.path.join(ds_dir, ds_fn)

    if os.path.exists(ds_fp):
        return xr.open_dataset(ds_fp, engine='h5netcdf')

    # Get path list
    local_paths = []
    ftp = ftplib.FTP(host='eftp.ifremer.fr', user='gg1f3e8', passwd='xG3jZhT9')
    for t in range(n_days):
        current_date = t_start + datetime.timedelta(days=t)
        yday = current_date.timetuple().tm_yday
        y_str, yday_str = str(current_date.year), '%03d' % yday
        local_path = 'current_netCDF/ftp_files/{0}/{1}/'.format(y_str, yday_str)
        ftp.cwd('/data/globcurrent/v3.0/global_025_deg/total_hs/{0}/{1}'.format(y_str, yday_str))
        if not os.path.exists(local_path):
            os.mkdir(local_path)
        local_paths.extend([local_path + filename
                            for filename in ftp.nlst() if '0000' in filename])

    # Try opening current data locally, otherwise download from FTP server
    try:
        print('Trying to load %d netCDF files locally... ' % (8*n_days), end='')
        ds = xr.open_mfdataset(local_paths, parallel=True,
                               combine='by_coords', preprocess=to_kts)
        print('success')
        ftp.close()
    except FileNotFoundError:
        print('failed')
        print('Downloading ocean current data from FTP server... ', end='')
        for t in range(n_days):
            current_date = t_start + datetime.timedelta(days=t)
            yday = current_date.timetuple().tm_yday
            y_str, yday_str = str(current_date.year), '%03d' % yday
            ftp.cwd('/data/globcurrent/v3.0/global_025_deg/total_hs/{0}/{1}'.format(y_str, yday_str))
            ftp_file_names = [filename for filename in ftp.nlst() if '0000' in filename]
            for ftp_fn in ftp_file_names:
                nc_path = os.path.join(nc_dir, '{}/{}'.format(y_str, yday_str))
                nc_fp = os.path.join(nc_path, ftp_fn)
                if os.path.exists(nc_fp):
                    continue
                with open(nc_fp, 'wb') as f:
                    ftp.retrbinary('RETR %s' % ftp_fn, f.write)
        ftp.close()
        ds = xr.open_mfdataset(local_paths, parallel=True,
                               combine='by_coords', preprocess=to_kts)
        print('done')
    ds.to_netcdf(ds_fp)
    return xr.open_dataset(ds_fp, engine='h5netcdf')


if __name__ == "__main__":
    _start_date = datetime.datetime(2016, 1, 1)
    _n_days = 50
    _ds = load(_start_date, _n_days)

    # Inspect netCDF file
    _nc_dir = 'current_netCDF/ftp_files/'
    _y, _yday = _start_date.year, '%03d' % _start_date.timetuple().tm_yday
    _m, _d = '%02d' % _start_date.month, '%02d' % _start_date.day
    _nc_path = os.path.join(_nc_dir, '{}/{}'.format(_y, _yday))
    _nc_fn = '{}{}{}000000-GLOBCURRENT-' \
             'L4-CUReul_hs-ALT_SUM-v03.0-fv01.0.nc'.format(_y, _m, _d)
    _nc_fp = os.path.join(_nc_path, _nc_fn)
    _nc_fid = netCDF4.Dataset(_nc_fp, 'r')
    _nc_attrs, _nc_dims, _nc_vars = ncdump(_nc_fid)
