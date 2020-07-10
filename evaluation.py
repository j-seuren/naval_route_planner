import collections
import functools
import numpy as np
import operator
import pickle
import rtree

from datetime import timedelta
from math import copysign, sqrt
from shapely.geometry import LineString


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


class Evaluator:
    def __init__(self,
                 vessel,
                 prep_polys,
                 rtree_idx,
                 ecas,
                 rtree_idx_eca,
                 eca_f,
                 start_date,
                 gc,
                 incl_curr=True,
                 del_sf=100,
                 del_sc=15, ):
        self.vessel = vessel            # Vessel class instance
        self.prep_polys = prep_polys    # Prepared land polygons
        self.rtree_idx = rtree_idx      # prep_polys' R-tree spatial index
        self.ecas = ecas                # ECA polygons
        self.rtree_idx_eca = rtree_idx_eca  # ecas' R-tree spatial index
        self.eca_f = eca_f              # Multiplication factor for ECA fuel
        self.start_date = start_date    # Start date of voyage
        self.del_sf = del_sf            # Max segment length (for feasibility)
        self.del_sc = del_sc            # Max segment length (for currents)
        self.incl_curr = incl_curr      # Boolean for including currents
        self.gc = gc                    # Geod class instance
        self.current_data = None

    def evaluate(self, ind):
        # Initialize variables
        tt = fc = 0.0

        for e in range(len(ind) - 1):
            # Get edge waypoints and edge boat speed
            p1, p2 = sorted((ind[e][0], ind[e + 1][0]))
            boat_speed = ind[e][1]

            # Compute travel time over edge
            e_tt = self.e_tt(p1, p2, tt, boat_speed)

            # Compute fuel consumption over edge
            e_fc = self.vessel.fuel_rates[boat_speed] * e_tt  # Tons

            # If edge intersects SECA increase fuel consumption by eca_f
            if edge_x_geos(p1, p2, self.rtree_idx_eca, self.ecas):
                e_fc *= self.eca_f

            # Increment objective values
            tt += e_tt
            fc += e_fc

        return tt / 24, fc

    def feasible(self, ind):
        for i in range(len(ind)-1):
            p1, p2 = sorted((ind[i][0], ind[i+1][0]))
            if not self.e_feasible(p1, p2):
                return False
        return True

    @Memoized
    def e_feasible(self, p1, p2):
        dist = self.gc.distance(p1, p2)
        lons, lats = self.gc.points(p1, p2, dist, self.del_sf)
        vertices = np.stack([lons, lats]).T

        # since we know the difference between any two points, we can use this to find wrap arounds on the plot
        max_dist = self.del_sf * 10 / 60

        # calculate distances and compare with max allowable distance
        dists = np.abs(np.diff(lons))
        cuts = np.where(dists > max_dist)[0]

        # if there are any cut points, cut them and begin again at the next point
        for i, cut in enumerate(cuts):
            # create new vertices with a nan inbetween and set those as the path's vertices
            verts = np.concatenate([vertices[:cut+1, :],
                                    [[np.nan, np.nan]],
                                    vertices[cut+1:, :]]
                                   )
            vertices = verts
        for i in range(len(vertices)-1):
            if not np.isnan(np.sum(vertices[i:i+2, :])):
                q1, q2 = tuple(vertices[i, :]), tuple(vertices[i+1, :])
                if edge_x_geos(q1, q2, self.rtree_idx, self.prep_polys):
                    return False
        return True

    def e_tt(self, p1, p2, tt, boat_speed):
        dist = self.gc.distance(p1, p2)
        if self.incl_curr:
            # Split edge in segments (seg) of del_sc in nautical miles
            lons, lats = self.gc.points(p1, p2, dist, self.del_sc)
            e_tt = 0.0
            for i in range(len(lons) - 1):
                q1, q2 = sorted(((lons[i], lats[i]), (lons[i+1], lats[i+1])))
                tot_hours = tt + e_tt
                seg_tt = self.seg_tt(q1, q2, boat_speed, tot_hours)
                e_tt += seg_tt
            return e_tt
        else:
            return dist / boat_speed

    def seg_tt(self, p1, p2, boat_speed, delta_hours):
        now = self.start_date + timedelta(hours=delta_hours)
        dist = self.gc.distance(p1, p2)

        # Get current of nearest grid point to middle point of edge
        x_m, y_m = (item / 2 for item in map(operator.add, p1, p2))
        lon_idx = int(round((x_m + 179.875) / 0.25))
        lat_idx = int(round((y_m + 89.875) / 0.25))
        u_m, v_m = self.current_data.get_grid_pt_current(now, lon_idx, lat_idx)

        # Calculate speed over ground
        sog = calc_sog(p1, p2, u_m, v_m, boat_speed)
        return dist / sog


def calc_sog(p1, p2, c_u, c_v, boat_speed):
    """
    Determine speed over ground (SOG) between points P and Q:
    SOG = boat speed + current speed (vectors)
    SOG direction must be the direction of PQ, hence
    SOG vector is the intersection of line PQ with the circle
    centered at the vector of current (u, v) with radius |boat_speed|
    """

    # Get equation for line PQ
    dx, dy = np.subtract(p2, p1)
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

    assert d >= 0, "No real solutions between pts {} and {}".format(p1, p2)

    if d == 0:
        x = (-b + sqrt(d)) / (2 * a)
        y = alpha * x
        sog = sqrt(x ** 2 + y ** 2)
    else:
        sqrt_d = sqrt(d)
        r1 = (-b + sqrt_d) / (2 * a)
        r2 = (-b - sqrt_d) / (2 * a)
        if copysign(1, r1) == copysign(1, dx) \
                and copysign(1, r2) == copysign(1, dx):
            # If both roots return resultant vector in right direction,
            # use resultant vector with greatest length
            y1, y2 = alpha * r1, alpha * r2
            v1, v2 = sqrt(r1 ** 2 + y1 ** 2), sqrt(r2 ** 2 + y2 ** 2)
            if v1 > v2:
                sog = v1
            else:
                sog = v2
        else:
            if copysign(1, r2) == copysign(1, dx):
                x = r2
            else:
                x = r1
            y = alpha * x
            sog = sqrt(x ** 2 + y ** 2)

    return sog


def edge_x_geos(p1, p2, rtree_idx, geos):
    line = LineString([p1, p2])
    # Returns the geometry indices of the minimum bounding rectangles
    # of polygons that intersect the edge bounds
    mbr_intersections = list(rtree_idx.intersection(line.bounds))
    if mbr_intersections:
        # For every mbr intersection
        # check if its polygon is actually intersect by the edge
        for idx in mbr_intersections:
            if geos[idx].intersects(line):
                return True
    return False
