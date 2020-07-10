import collections
import ocean_current_data
import functools
import math
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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


class CurrentOperator:
    def __init__(self, t_s, n_days):
        self.t_s = t_s.replace(second=0, microsecond=0, minute=0, hour=0)
        self.n_days = n_days

        # Initialize CurrentDataRetriever class instance
        downloader = ocean_current_data.CurrentDataRetriever(self.t_s, self.n_days)
        downloader.store_combined_nc()

        fp = downloader.ds_fp
        print('Opening and loading combined netCDF into memory:'.format(fp), end=' ')
        with xr.open_dataset(fp, engine='h5netcdf') as ds:
            self.ds = ds.compute()
        print('done')

        cache = Cache(2e9)  # Leverage two gigabytes of memory
        cache.register()  # Turn cache on globally

    @Memoized
    def get_grid_pt_current(self, date_in, lon_idx, lat_idx):
        delta = date_in - self.t_s
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


if __name__ == '__main__':
    print(None)
