import dask.array as da
import netCDF4
import numpy as np
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
                print('\t\t%s:' % ncattr, repr(nc_fid.variables[key].getncattr(ncattr)))
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


# Open data as read-only
fp = 'C:/dev/data/gebco_2020_netcdf/' + 'GEBCO_2020.nc'
fp_TID = 'C:/dev/data/gebco_2020_tid_netcdf/' + 'GEBCO_2020_TID.nc'
# TID = xr.open_dataset(fp_TID, chunks={'lon': 16000, 'lat': 8000})
ds = xr.open_dataset(fp, chunks={'lon': 11313, 'lat': 5657})
ds = da.from_array(ds.to_array())
print(ds)
print(ds[0, 20000, 40000].compute())

bins = np.int8([-5, -10, -20])

inds = np.digitize(ds, bins)

print(inds[0, 20000, 40000].compute())

# print(inds[1, 20000, 40000].compute())
#
# print(ds2.isel(lon=40000, lat=20000).compute())
# # print(ds2)
#
# # Get longitudes and latitudes
# lons = ds.variables['lon']
# lats = ds.variables['lat']
#
#
# # tid = TID.variables['tid']
# # elevation = data.variables['elevation'][:]
# # TID.close()