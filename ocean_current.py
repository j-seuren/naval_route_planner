import cartopy.crs as ccrs
import ftplib
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import operator
import os
import pandas as pd
import pickle
import scipy

from datetime import datetime
from deap import base, creator
from haversine import haversine
from math import copysign, sqrt
from mpl_toolkits import basemap


class Vessel:
    def __init__(self, name, _speeds, _fuel_rates):
        self.name = name
        self.speeds = _speeds
        self.fuel_rates = _fuel_rates


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
            print("\tName:", dim )
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


def plot_current_field(uin, vin, lons, lats):
    # Create map
    m = basemap.Basemap(projection='cyl', llcrnrlat=-90., urcrnrlat=90., resolution='c', llcrnrlon=-180., urcrnrlon=180.)
    m.drawcoastlines()
    m.drawparallels(np.arange(-90., 90., 30.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 180., 30.), labels=[0, 0, 0, 1], fontsize=10)

    # Transform vector and coordinate data
    vec_lon = uin.shape[1] // 10
    vec_lat = uin.shape[0] // 10
    u_rot, v_rot, x, y = m.transform_vector(uin, vin, lons, lats, vec_lon, vec_lat, returnxy=True)

    # Create vector plot on map
    vec_plot = m.quiver(x, y, u_rot, v_rot, scale=50)
    plt.quiverkey(vec_plot, 0.2, -0.2, 1, '1 knot', labelpos='W')  # Position and reference label


def read_netcdf(date):
    # Convert date to day of year
    date = datetime.strptime(date, "%Y%m%d")
    Y, m, d, yday = date.year, '%02d' % date.month, '%02d' % date.day, '%03d' % date.timetuple().tm_yday

    # Try reading data from directory. Except if file path does not exist, read from file from FTP server
    dir_path = 'globcurrent_global_025_deg_total_hs/'
    try:
        # Open data as read-only
        fp = dir_path + 'netCDF/{0}/{1}/{0}{1}{2}-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v03.0-fv01.0.nc'.format(Y, m, d)
        data = netCDF4.Dataset(fp, mode='r')
    except FileNotFoundError:
        ftp = ftplib.FTP('eftp.ifremer.fr')
        ftp.login(user='gg1f3e8', passwd='xG3jZhT9')
        ftp.cwd('/data/globcurrent/v3.0/global_025_deg/total_hs/{}/{}/'.format(Y, yday))
        try:
            with open(
                    dir_path +
                    'netCDF/{0}/{1}/{0}{1}{2}-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v03.0-fv01.0.nc'.format(Y, m, d),
                    'wb') as fp:
                ftp.retrbinary('RETR {0}{1}{2}-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v03.0-fv01.0.nc'.format(Y, m, d),
                               fp.write)
        except FileNotFoundError:
            os.makedirs(dir_path + 'netCDF/{}/{}'.format(Y, m))
            with open(dir_path +
                      'netCDF/{0}/{1}/{0}{1}{2}-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v03.0-fv01.0.nc'.format(Y, m, d),
                      'wb') as fp:
                ftp.retrbinary('RETR {0}{1}{2}-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v03.0-fv01.0.nc'.format(Y, m, d),
                               fp.write)
        ftp.close()

        # Open data as read-only
        fp = dir_path + 'netCDF/{0}/{1}/{0}{1}{2}-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v03.0-fv01.0.nc'.format(Y, m, d)
        data = netCDF4.Dataset(fp, mode='r')

    # Get longitudes and latitudes
    lons = data.variables['lon'][:]
    lats = data.variables['lat'][:]

    # Get eastward (u) and northward (v) Euler velocities in knots
    u = 1.94384 * data.variables['eastward_eulerian_current_velocity'][0, :, :]
    v = 1.94384 * data.variables['northward_eulerian_current_velocity'][0, :, :]

    # ncdump(data)
    data.close()

    # Add cyclic column
    u_cyc, _ = basemap.addcyclic(u, lons)
    v_cyc, lons = basemap.addcyclic(v, lons)

    # Create mesh of lons and lats
    lons_mesh, lats_mesh = np.meshgrid(lons, lats)

    # Transform u and v to mercator projection
    src_projection, tgt_projection = ccrs.PlateCarree(), ccrs.Mercator()
    u_proj, v_proj = tgt_projection.transform_vectors(src_projection, lons_mesh, lats_mesh, u_cyc, v_cyc)

    return u_proj, v_proj, np.array(lons.tolist()), np.array(lats.tolist())


def bilinear_interpolation(x, y, z, xi, yi):
    a = np.array([x[1] - xi, xi - x[0]])
    b = np.array([y[1] - yi, yi - y[0]])

    return a.dot(z.dot(b)) / ((x[1] - x[0]) * (y[1] - y[0]))


def speed_over_ground(p, q, c_u, c_v, boat_speed):
    """
    Determine speed over ground (SOG) between points P and Q:
    SOG = boat speed + current speed (vectors)
    SOG direction must be the direction of PQ, hence
    SOG vector is the intersection of line PQ with the circle
    centered at the vector of current (u, v) with radius |boat_speed|
    """

    # Get equation for line PQ
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    try:
        alpha = dy / (dx + 0.0)
    except ZeroDivisionError:
        alpha = copysign(99999999999999999, dy)

    # Intersection of circle; (x - u)^2 + (y - v)^2 = boat_speed^2,
    # and line PQ; y = slope * x,
    # gives quadratic equation; ax^2 + bx + c = 0, with
    a = 1 + alpha ** 2
    b = -2 * (c_u + alpha * c_v)
    c = c_u ** 2 + c_v ** 2 - boat_speed ** 2
    d = b ** 2 - 4 * a * c  # discriminant

    assert d >= 0, "There exist no real solutions between points {} and {}".format(p, q)

    if d == 0:
        x = (-b + sqrt(d)) / (2 * a)
        y = alpha * x
        SOG = sqrt(x ** 2 + y ** 2)
    else:
        rt = sqrt(d)
        root1 = (-b + rt) / (2 * a)
        root2 = (-b - rt) / (2 * a)
        if copysign(1, root1) == copysign(1, dx) and copysign(1, root2) == copysign(1, dx):
            # If both roots return resultant vector in right direction,
            # use resultant vector with greatest length
            y1, y2 = alpha * root1, alpha * root2
            v1, v2 = sqrt(root1 ** 2 + y1 ** 2), sqrt(root2 ** 2 + y2 ** 2)
            if v1 > v2:
                SOG = v1
            else:
                SOG = v2
        else:
            if copysign(1, root2) == copysign(1, dx):
                x = root2
            else:
                x = root1
            y = alpha * x
            SOG = sqrt(x ** 2 + y ** 2)

    return SOG


def get_edge_travel_time(p, q, boat_speed, distance, uin, vin, lons, lats):
    # Middle point of edge
    x_m, y_m = (item / 2 for item in map(operator.add, p, q))

    # Get coordinates of nearby grid points
    lon2, lat2 = np.searchsorted(lons, x_m), np.searchsorted(lats, y_m)
    lon_idx, lat_idx = [lon2-1, lon2], [lat2-1, lat2]
    x = np.array([lons[lon_idx[0]], lons[lon_idx[1]]])
    y = np.array([lats[lat_idx[0]], lats[lat_idx[1]]])

    # Create array of u,v values at grid points
    u, v = np.empty([2, 2]), np.empty([2, 2])
    for i in range(2):
        for j in range(2):
            u[i, j] = uin[lat_idx[j], lon_idx[i]]
            v[i, j] = vin[lat_idx[j], lon_idx[i]]

    # Bilinear interpolation of u and v at point m
    u_m = bilinear_interpolation(x, y, u, x_m, y_m)
    v_m = bilinear_interpolation(x, y, v, x_m, y_m)

    # If u, v value is masked or nan, set ocean current to 0
    if np.ma.is_masked(u_m) or np.isnan(u_m) or np.ma.is_masked(v_m) or np.isnan(v_m):
        u_m = v_m = 0

    # Calculate speed over ground
    SOG = speed_over_ground(p, q, u_m, v_m, boat_speed)
    return distance / SOG


if __name__ == '__main__':
    # Read netCDF
    u_read, v_read, lons_read, lats_read = read_netcdf('20160101')

    # Plot current field
    plot_current_field(u_read, v_read, lons_read, lats_read)

    plt.show()
