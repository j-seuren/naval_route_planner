import haversine
import math
import numpy as np
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


class Geodesic:
    def __init__(self, dist_calc='ellipsoidal'):
        if dist_calc == 'ellipsoidal':
            self.ref_sys = pyproj.Geod(ellps='WGS84')
            self.distance = self.ellipsoidal
        elif dist_calc == 'rhumb_line':
            self.distance = self.rhumb_line
        elif dist_calc == 'great_circle':
            self.distance = self.great_circle
        else:
            raise ValueError(
                "No distance calculation method specified."
                " Choose 'ellipsoidal', 'rhumb_line', or 'great_circle'.")

    @np_cache
    def ellipsoidal(self, p1, p2):
        """
            Get ellipsoidal distance from the longitude-latitude
            pair ``lon1,lat1`` to ``lon2,lat2``
            Returns ellipsoidal distance
            """
        (lon1, lat1), (lon2, lat2) = p1, p2
        az12, az21, dist = self.ref_sys.inv(lon1, lat1, lon2, lat2)

        return dist / 1852.0  # To nautical miles

    @np_cache
    def rhumb_line(self, p1, p2):
        """
            Get rhumb line distance from the longitude-latitude
            pair ``lon1,lat1`` to ``lon2,lat2``
            Returns rhumb line distance
            """
        (lam1, phi1), (lam2, phi2) = np.radians(p1), np.radians(p2)
        dLam = lam2 - lam1
        dPhi = phi2 - phi1

        dPsi = math.log(math.tan(math.pi / 4 + phi2 / 2) / math.tan(math.pi / 4 + phi1 / 2))
        if abs(dPsi) > 10e-12:
            q = dPhi / dPsi
        else:
            q = math.cos(phi1)

        if abs(dLam) > math.pi:
            if dLam > 0:
                dLam = -(2 * math.pi - dLam)
            else:
                dLam = 2 * math.pi + dLam

        return math.sqrt(dPhi * dPhi + q * q * dLam * dLam) * 3440.1  # nautical miles

    @np_cache
    def great_circle(self, p1, p2):
        """
            Get great circle distance from the longitude-latitude
            pair ``lon1,lat1`` to ``lon2,lat2``
            Returns great circle distance
            """

        return haversine.haversine(p1, p2, unit='nmi')  # nautical miles

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
        ==============   =======================================================
        Returns two lists of lons, lats points on great circle
        """
        (lon1, lat1), (lon2, lat2) = p1, p2
        n_points = int((dist + 0.5 * del_s) / del_s)
        lon_lats = self.ref_sys.npts(lon1, lat1, lon2, lat2, n_points)
        lons, lats = [lon1], [lat1]
        for lon, lat in lon_lats:
            lons.append(lon)
            lats.append(lat)
        lons.append(lon2)
        lats.append(lat2)

        return lons, lats
