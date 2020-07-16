import pyproj

from functools import lru_cache, wraps


def np_cache(function):
    """
    Cache decorator for numpy arrays
    """
    @lru_cache(maxsize=None)
    def cached_wrapper(gc, p1, p2):
        return function(gc, p1, p2)

    @wraps(function)
    def wrapper(gc, p1, p2):
        if p1 is not tuple and p2 is not tuple:
            p1 = tuple(p1)
            p2 = tuple(p2)
        return cached_wrapper(gc, p1, p2)

    return wrapper


class GreatCircle:
    def __init__(self):
        self.geod = pyproj.Geod(a=3443.918467, f=0.0033528106647475126)

    @np_cache
    def distance(self, p1, p2):
        """
            Get great circle distance from the longitude-latitude
            pair ``lon1,lat1`` to ``lon2,lat2``
            Returns great circle distance
            """
        (lon1, lat1), (lon2, lat2) = p1, p2
        az12, az21, dist = self.geod.inv(lon1, lat1, lon2, lat2)

        return dist

    @lru_cache(maxsize=None)
    def points(self, p1, p2, dist, del_s):
        """
        Get great circle points from the longitude-latitude
        pair ``lon1,lat1`` to ``lon2,lat2``
        .. tabularcolumns:: |l|L|
        ==============   =======================================================
        Keyword          Description
        ==============   =======================================================
        del_s            points on great circle computed every del_s kilometers
                         (default 100).
        ==============   =======================================================
        Returns two lists of lons, lats points on great circle
        """
        (lon1, lat1), (lon2, lat2) = p1, p2
        n_points = int((dist + 0.5 * del_s) / del_s)
        lon_lats = self.geod.npts(lon1, lat1, lon2, lat2, n_points)
        lons, lats = [lon1], [lat1]
        for lon, lat in lon_lats:
            lons.append(lon)
            lats.append(lat)
        lons.append(lon2)
        lats.append(lat2)

        return lons, lats
