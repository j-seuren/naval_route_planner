from data_config import ocean_current_data, weather_data
import math
import matplotlib.pyplot as plt
import numpy as np
import support

from dask.cache import Cache
from functools import lru_cache
from mpl_toolkits import basemap


class CurrentOperator:
    def __init__(self, t_s, n_days):
        self.t_s = t_s.replace(second=0, microsecond=0, minute=0, hour=0)
        self.n_days = n_days

        # Initialize CurrentDataRetriever class instance
        retriever = ocean_current_data.CurrentDataRetriever(self.t_s, self.n_days)
        self.ds = retriever.get_ds()

        cache = Cache(2e9)  # Leverage two gigabytes of memory
        cache.register()  # Turn cache on globally

    @lru_cache(maxsize=None)
    def get_grid_pt_current(self, date_in, lon, lat):
        lon_idx = int(round((lon + 179.875) / 0.25))
        lat_idx = int(round((lat + 89.875) / 0.25))
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
    plt.subplot()
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


class WeatherOperator:
    def __init__(self, startDate, nDays):
        self.startDate = startDate.replace(second=0, microsecond=0, minute=0, hour=0)
        assert nDays * 8 < 384, 'Estimated travel days exceeds weather forecast period'

        # Initialize CurrentDataRetriever class instance
        retriever = weather_data.WeatherDataRetriever(startDate=self.startDate, nDays=nDays)
        self.da = retriever.get_ds(historical=True).to_array().data

        cache = Cache(2e9)  # Leverage two gigabytes of memory
        cache.register()  # Turn cache on globally
        self.lons = np.linspace(-180, 179.5, 720)
        self.lats = np.linspace(90, -90, 360)

    @lru_cache(maxsize=None)
    def get_grid_pt_wind(self, time, lon, lat):
        lon_idx = int(round((lon + 180) / 0.5))
        lat_idx = int(round(-(lat - 90) / 0.5))
        step_idx = (time - self.startDate).seconds // 3600 // 3
        vals = self.da[:, step_idx, lat_idx, lon_idx]
        BN = vals[0]
        windDir = vals[1]

        if math.isnan(BN) or math.isnan(windDir):
            BN, windDir = 0, 0.0

        return BN, windDir


if __name__ == '__main__':
    from datetime import datetime, timedelta
    t_start = datetime.today() - timedelta(days=1)
    nr_days = 7
    windOp = WeatherOperator(t_start, nr_days)
    print(windOp.get_grid_pt_wind(t_start, 54.4, 2.3))