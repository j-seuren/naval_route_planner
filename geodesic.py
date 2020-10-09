import haversine
import math
import numpy as np
import pyproj

from functools import lru_cache, wraps


def np_cache(function):
    """
    Cache decorator for numpy arrays
    """
    @lru_cache(maxsize=int(1e+6))
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
        self.ref_sys = pyproj.Geod(ellps='WGS84')

        if dist_calc == 'ellipsoidal':
            self.distance = self.ellipsoidal
        elif dist_calc == 'rhumb_line':
            self.distance = self.rhumb_line
        elif dist_calc == 'great_circle':
            self.distance = self.great_circle
        else:
            raise ValueError(
                "No distance calculation method specified."
                " Choose 'ellipsoidal', 'rhumb_line', or 'great_circle'.")

    def total_distance(self, ind):
        return sum([self.distance(wp1[0], wp2[0]) for wp1, wp2 in zip(ind[:-1], ind[1:])])

    @lru_cache(maxsize=None)
    def ellipsoidal(self, p1, p2, bearing=False):
        """
            Get ellipsoidal distance from the longitude-latitude
            pair ``lon1,lat1`` to ``lon2,lat2``
            Returns ellipsoidal distance
            """
        (lon1, lat1), (lon2, lat2) = p1, p2
        bearingDeg, az21, dist = self.ref_sys.inv(lon1, lat1, lon2, lat2)

        if bearing:
            return dist / 1852.0, bearingDeg
        else:
            return dist / 1852.0  # To nautical miles

    @lru_cache(maxsize=None)
    def euclidean(self, p1, p2):
        """
            Get ellipsoidal distance from the longitude-latitude
            pair ``lon1,lat1`` to ``lon2,lat2``
            Returns ellipsoidal distance
            """
        p1, p2 = np.asarray(p1), np.asarray(p2)
        return np.linalg.norm(p2 - p1)

    @lru_cache(maxsize=None)
    def rhumb_line(self, p1, p2, bearing=False):
        """
            Get rhumb line distance from the longitude-latitude
            pair ``lon1,lat1`` to ``lon2,lat2``
            Returns rhumb line distance
            """
        [(lam1, phi1), (lam2, phi2)] = np.radians([p1, p2])

        dPsi = math.log(math.tan(math.pi / 4 + phi2 / 2) / math.tan(math.pi / 4 + phi1 / 2))
        dPhi = phi2 - phi1
        if abs(dPsi) > 10e-12:
            q = dPhi / dPsi
        else:
            q = math.cos(phi1)

        dLam = lam2 - lam1  # Longitude difference
        if abs(dLam) > math.pi:  # take shortest route: dLam < PI
            dLam = dLam - math.copysign(2 * math.pi, dLam)

        if bearing:
            return math.sqrt(dPhi * dPhi + q * q * dLam * dLam) * 3440.1, math.atan2(dLam, dPsi)
        else:
            return math.sqrt(dPhi * dPhi + q * q * dLam * dLam) * 3440.1  # nautical miles

    @np_cache
    def great_circle(self, p1, p2, bearing=False):
        """
            Get great circle distance from the longitude-latitude
            pair ``lon1,lat1`` to ``lon2,lat2``
            Returns great circle distance
            """

        if bearing:
            return haversine.haversine(reversed(p1), reversed(p2), unit='nmi'), calc_bearing(p1, p2)
        else:
            return haversine.haversine(reversed(p1), reversed(p2), unit='nmi')  # nautical miles

    @lru_cache(maxsize=None)
    def points(self, p1, p2, dist, delS):
        """
        Get great circle points from the longitude-latitude
        pair ``lon1,lat1`` to ``lon2,lat2``
        .. tabularcolumns:: |l|L|
        ==============   =======================================================
        Keyword          Description
        ==============   =======================================================
        del_s            points on great circle computed every del_s nautical miles
        ==============   =======================================================
        Returns two lists of lons, lats points on great circle
        """
        (lon1, lat1), (lon2, lat2) = p1, p2
        nPoints = int(dist / delS + 0.5)
        lonLats = self.ref_sys.npts(lon1, lat1, lon2, lat2, nPoints)
        lons, lats = [lon1], [lat1]
        for lon, lat in lonLats:
            lons.append(lon)
            lats.append(lat)
        lons.append(lon2)
        lats.append(lat2)

        return lons, lats


def calc_bearing(p1, p2):
    """ Calculate bearing in degrees"""
    # Convert degrees to radians
    [(lam1, phi1), (lam2, phi2)] = np.radians([p1, p2])

    # Latitude difference projected on Mercator projection
    dPsi = math.log(math.tan(math.pi / 4 + phi2 / 2) / math.tan(math.pi / 4 + phi1 / 2))

    dLam = lam2 - lam1  # Longitude difference
    if abs(dLam) > math.pi:  # take shortest route: dLam < PI
        dLam = dLam - math.copysign(2 * math.pi, dLam)

    return math.degrees(math.atan2(dLam, dPsi))


if __name__ == '__main__':
    import timeit
    from math import radians

    n = 300000
    r = 100

    t = timeit.Timer(stmt="pyproj.Geod(ellps='WGS84').inv(-54, 60, 54, 8)", setup='import pyproj')
    print(1e+6 * np.min((t.repeat(r, n))) / n)
    t = timeit.Timer(stmt="DistanceMetric.get_metric('haversine').pairwise(np.radians([[lat1, lon1], [lat2, lon2]]))",
                     setup='from sklearn.neighbors import DistanceMetric; lon1, lat1, lon2, lat2 = -54, 60, 54, 8; import numpy as np')
    print(1e+6 * np.min((t.repeat(r, n))) / n)
