import collections
import datetime
import download_currents
import functools
import math
import matplotlib.pyplot as plt
import numpy as np

from dask.cache import Cache
from mpl_toolkits import basemap


class Memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable: better to not cache than blow up.
            print(args, 'not hashable')
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


def round_to_3h(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    t_out = t.replace(second=0, microsecond=0, minute=0, hour=t.hour) \
            + datetime.timedelta(hours=t.minute // 30)
    if t_out.hour % 3 == 1:
        return t_out - datetime.timedelta(hours=1)
    elif t_out.hour % 3 == 2:
        return t_out + datetime.timedelta(hours=1)
    return t_out


class CurrentData:
    def __init__(self, start_date, n_days):
        self.start_date = round_to_3h(start_date)
        self.n_days = n_days
        self.ds = download_currents.load(self.start_date, self.n_days).compute()
        print('Loaded data into memory')
        cache = Cache(2e9)  # Leverage two gigabytes of memory
        cache.register()  # Turn cache on globally

    @Memoized
    def get_grid_pt_current(self, date_in, lon_idx, lat_idx):
        delta = date_in - self.start_date
        if delta.days < self.n_days:
            day_idx = delta.seconds // 3600 // 3
            vals = self.ds.isel(time=day_idx, lat=lat_idx, lon=lon_idx).load()
            u_pt = float(vals['u_knot'])
            v_pt = float(vals['v_knot'])

            if math.isnan(u_pt) or math.isnan(v_pt):
                u_pt = v_pt = 0.0
        else:
            u_pt = v_pt = 0.0

        return u_pt, v_pt


class Vessel:
    def __init__(self, name, _speeds, _fuel_rates):
        self.name = name
        self.speeds = _speeds
        self.fuel_rates = _fuel_rates


def plot_current_field(uin, vin, lons, lats):
    # Create map
    m = basemap.Basemap(projection='cyl', resolution='c',
                        llcrnrlat=-90., urcrnrlat=90.,
                        llcrnrlon=-180., urcrnrlon=180.)
    m.drawcoastlines()
    m.drawparallels(np.arange(-90., 90., 30.),
                    labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 180., 30.),
                    labels=[0, 0, 0, 1], fontsize=10)

    # Transform vector and coordinate data
    vec_lon = uin.shape[1] // 10
    vec_lat = uin.shape[0] // 10
    u_rot, v_rot, x, y = m.transform_vector(uin, vin, lons, lats,
                                            vec_lon, vec_lat, returnxy=True)

    # Create vector plot on map
    vec_plot = m.quiver(x, y, u_rot, v_rot, scale=50, width=0.002)
    plt.quiverkey(vec_plot, 0.2, -0.2, 1, '1 knot', labelpos='W')


def bilinear_interpolation(x, y, z, xi, yi):
    a = np.array([x[1] - xi, xi - x[0]])
    b = np.array([y[1] - yi, yi - y[0]])

    return a.dot(z.dot(b)) / ((x[1] - x[0]) * (y[1] - y[0]))


if __name__ == '__main__':
    print(None)
