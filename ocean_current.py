import datetime
import ftplib
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr

from mpl_toolkits import basemap


def round_to_3h(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    t_out = t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + datetime.timedelta(hours=t.minute // 30)
    if t_out.hour % 3 == 1:
        return t_out - datetime.timedelta(hours=1)
    elif t_out.hour % 3 == 2:
        return t_out + datetime.timedelta(hours=1)
    return t_out


class CurrentData:
    def __init__(self):
        self.period = None
        self.ds = None

    def get_currents(self, date_in):
        date_rounded = round_to_3h(date_in)

        # Convert date_time to day of year
        Y = str(date_rounded.year)
        current_yday = date_rounded.timetuple().tm_yday
        period_length = 5
        period = current_yday // period_length

        if self.period == period:
            ds = self.ds
        else:
            if self.ds:
                self.ds.close()
            p = 'current_netCDF/chunked_data/'
            ds = xr.open_dataset(p + '{0}_period{1}_length{2}.nc'.format(Y, period, period_length))
            self.ds = ds
            self.period = period

        day_idx = current_yday % period_length
        hr_idx = date_rounded.hour // 3
        ds = ds.isel(time=(8 * day_idx + hr_idx))

        # Get eastward (u) and northward (v) Euler velocities in knots
        u = ds.variables['u_knot']
        v = ds.variables['v_knot']
        lons = ds.variables['lon']
        lats = ds.variables['lat']

        return u, v, lons, lats


class Vessel:
    def __init__(self, name, _speeds, _fuel_rates):
        self.name = name
        self.speeds = _speeds
        self.fuel_rates = _fuel_rates


def plot_current_field(uin, vin, lons, lats):
    # Create map
    m = basemap.Basemap(projection='cyl', resolution='c',
                        llcrnrlat=-90., urcrnrlat=90., llcrnrlon=-180., urcrnrlon=180.)
    m.drawcoastlines()
    m.drawparallels(np.arange(-90., 90., 30.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 180., 30.), labels=[0, 0, 0, 1], fontsize=10)

    # Transform vector and coordinate data
    vec_lon = uin.shape[1] // 10
    vec_lat = uin.shape[0] // 10
    u_rot, v_rot, x, y = m.transform_vector(uin, vin, lons, lats, vec_lon, vec_lat, returnxy=True)

    # Create vector plot on map
    vec_plot = m.quiver(x, y, u_rot, v_rot, scale=50, width=0.002)
    plt.quiverkey(vec_plot, 0.2, -0.2, 1, '1 knot', labelpos='W')  # Position and reference label


def bilinear_interpolation(x, y, z, xi, yi):
    a = np.array([x[1] - xi, xi - x[0]])
    b = np.array([y[1] - yi, yi - y[0]])

    return a.dot(z.dot(b)) / ((x[1] - x[0]) * (y[1] - y[0]))


if __name__ == '__main__':
    # Initialize "CurrentData"
    _start_date = datetime.datetime(2016, 1, 1)
    current_data = CurrentData()

    u_out, v_out, lons_out, lats_out = current_data.get_currents(_start_date)
