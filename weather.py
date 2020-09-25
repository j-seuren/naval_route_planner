from data_config import current_data, wind_data
import math
import matplotlib.pyplot as plt
import numpy as np

from dask.cache import Cache
# from functools import lru_cache
from mpl_toolkits import basemap


class CurrentOperator:
    def __init__(self, t0, nDays):
        self.t0 = t0.replace(second=0, microsecond=0, minute=0, hour=0)
        self.nDays = nDays

        # Initialize CurrentDataRetriever class instance
        retriever = current_data.CurrentDataRetriever(self.t0, self.nDays)
        self.da = retriever.get_da()

        cache = Cache(2e9)  # Leverage two gigabytes of memory
        cache.register()  # Turn cache on globally

    # @lru_cache(maxsize=None)
    def get_grid_pt_current(self, date_in, lon, lat):
        lonIdx = int(round((lon + 179.875) / 0.25))
        latIdx = int(round((lat + 89.875) / 0.25))
        delta = date_in - self.t0
        if delta.days < self.nDays:
            dayIdx = delta.seconds // 3600 // 3
            vals = self.da[:, dayIdx, latIdx, lonIdx]
            u_pt, v_pt = vals[0], vals[1]

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


class WindOperator:
    def __init__(self, t0, nDays):
        self.t0 = t0.replace(second=0, microsecond=0, minute=0, hour=0)
        assert nDays * 8 < 384, 'Estimated travel days exceeds wind forecast period'

        # Initialize CurrentDataRetriever class instance
        retriever = wind_data.WindDataRetriever(startDate=self.t0, nDays=nDays)
        self.da = retriever.get_da(forecast=False)

        cache = Cache(2e9)  # Leverage two gigabytes of memory
        cache.register()  # Turn cache on globally

    # @lru_cache(maxsize=None)
    def get_grid_pt_wind(self, time, lon, lat):
        resolution = 0.5
        # Get indices of data array
        lon_idx = int(round((lon + 180) / resolution))
        lon_idx = 0 if lon_idx == 720 else lon_idx
        lat_idx = int(round((lat + 90) / resolution))
        step_idx = (time - self.t0).seconds // 3600 // 3
        vals = self.da[:, step_idx, lat_idx, lon_idx]
        BN = vals[0]
        windDir = vals[1]

        if math.isnan(BN) or math.isnan(windDir):
            BN, windDir = 0, 0.0

        return BN, windDir


if __name__ == '__main__':
    from datetime import datetime, timedelta
    startDate = datetime(2018, 10, 10)
    nr_days = 1
    windOp = WindOperator(startDate, nr_days)
    print(windOp.get_grid_pt_wind(startDate, -86.707108, 27.572103))

