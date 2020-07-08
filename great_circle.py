import collections
import functools
import pyproj


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
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
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


class GreatCircle:
    def __init__(self):
        self.geod = pyproj.Geod(a=3443.918467, f=0.0033528106647475126)

    @Memoized
    def distance(self, p1, p2):
        """
            Get great circle distance from the longitude-latitude
            pair ``lon1,lat1`` to ``lon2,lat2``
            Returns great circle distance
            """
        (lon1, lat1), (lon2, lat2) = p1, p2
        az12, az21, dist = self.geod.inv(lon1, lat1, lon2, lat2)

        return dist

    @Memoized
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
