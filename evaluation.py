import great_circle
import ocean_current
import operator
import pickle
import rtree

from datetime import datetime, timedelta
from math import copysign, sqrt, isnan
from shapely.geometry import LineString


class Evaluator:
    def __init__(self,
                 vessel,
                 prepared_polys,
                 rtree_idx,
                 geod,
                 seca_factor,
                 start_date=datetime(2016, 1, 1),
                 del_s=100,
                 del_sc=15,
                 include_currents=True):
        self.vessel = vessel
        self.prepared_polys = prepared_polys
        self.rtree_idx = rtree_idx
        self.geod = geod
        self.seca_factor = seca_factor
        self.start_date = start_date                                    # Start date for initializing ocean currents
        self.del_s = del_s                                              # Maximum segment length (nautical miles)
        self.del_sc = del_sc                                            # Maximum segment length for current calculation
        self.include_currents = include_currents
        self.dist_cache = dict()
        self.feas_cache = dict()
        self.points_cache = dict()
        self.travel_time_cache = dict()

        with open('C:/dev/data/seca_areas_csv', 'rb') as f:
            self.secas = pickle.load(f)

        self.rtree_idx_seca = rtree.index.Index()
        for idx, seca in enumerate(self.secas):
            self.rtree_idx_seca.insert(idx, seca.bounds)

        # Initialize "CurrentData"
        self.current_data = ocean_current.CurrentData()

    def evaluate(self, individual):
        # Initialize variables
        travel_time = fuel_consumption = 0.0

        for e in range(len(individual) - 1):
            p1, p2, boat_speed = individual[e][0], individual[e + 1][0], individual[e][1]
            k = tuple(sorted([p1, p2]))

            e_dist = self.dist_cache.get(k, False)
            if not e_dist:  # Never steps in IF-statement
                print('computes distance')
                e_dist = great_circle.distance(p1[0], p1[1], p2[0], p2[1], self.geod)
                self.dist_cache[k] = e_dist

            if self.include_currents:
                k2 = k + (travel_time, boat_speed)
                e_travel_time = self.travel_time_cache.get(k2, False)
                if not e_travel_time:
                    # Split edge in segments (seg) of max seg_length in km
                    points = self.points_cache.get(k, False)  # Evaluated in "Feasible" decorator
                    lons, lats = points[0], points[1]
                    e_travel_time = 0.0
                    for i in range(len(lons) - 1):
                        p1, p2 = (lons[i], lats[i]), (lons[i+1], lats[i+1])
                        seg_dist = great_circle.distance(p1[0], p1[1], p2[0], p2[1], self.geod)
                        now = self.start_date + timedelta(hours=travel_time + e_travel_time)
                        seg_travel_time = self.get_seg_travel_time(p1, p2, boat_speed, seg_dist, now)
                        e_travel_time += seg_travel_time
                    self.travel_time_cache[k2] = e_travel_time
            else:
                e_travel_time = e_dist / boat_speed

            # If edge intersects SECA increase fuel consumption by seca_factor
            if edge_x_geos(p1, p2, self.rtree_idx_seca, self.secas):
                seca_factor = self.seca_factor
            else:
                seca_factor = 1
            edge_fuel_consumption = self.vessel.fuel_rates[boat_speed] * e_travel_time * seca_factor  # Tons

            # Increment objective values
            travel_time += e_travel_time
            fuel_consumption += edge_fuel_consumption

        return travel_time, fuel_consumption

    def feasible(self, individual):
        for i in range(len(individual) - 1):
            p1, p2 = individual[i][0], individual[i + 1][0]
            if not self.edge_feasible(p1, p2):
                return False
        return True

    def edge_feasible(self, p1, p2):
        # First check if feasibility check is already performed
        k = tuple(sorted([p1, p2]))
        feasible = self.feas_cache.get(k, None)
        if feasible == 1:
            return True
        elif feasible == 0:
            return False

        dist = self.dist_cache.get(k, False)
        if not dist:
            dist = great_circle.distance(p1[0], p1[1], p2[0], p2[1], self.geod)
            self.dist_cache[k] = dist

        points = self.points_cache.get(k, False)
        if not points:
            points = great_circle.points(p1[0], p1[1], p2[0], p2[1], dist, self.geod, self.del_s)
            self.points_cache[k] = points
        lons, lats = points[0], points[1]
        for i in range(len(lons) - 1):
            # Compute line bounds
            q1, q2 = (lons[i], lats[i]), (lons[i + 1], lats[i + 1])

            if edge_x_geos(q1, q2, self.rtree_idx, self.prepared_polys):
                self.feas_cache[k] = 0
                return False
        self.feas_cache[k] = 1
        return True

    def get_seg_travel_time(self, p1, p2, boat_speed, distance, date_time):
        # Middle point of edge
        x_m, y_m = (item / 2 for item in map(operator.add, p1, p2))

        u, v, _, _ = self.current_data.get_currents(date_time)

        # Get coordinates of nearest grid point
        lon_idx, lat_idx = int(round((x_m + 179.875) / 0.25)), int(round((y_m + 89.875) / 0.25))

        u_m = float(u[lat_idx, lon_idx].compute())
        v_m = float(v[lat_idx, lon_idx].compute())

        # If u, v value is nan, set ocean current to 0
        if isnan(u_m) or isnan(v_m):
            u_m = v_m = 0

        # Calculate speed over ground
        SOG = speed_over_ground(p1, p2, u_m, v_m, boat_speed)
        return distance / SOG


def speed_over_ground(p, q, c_u, c_v, boat_speed):
    """
    Determine speed over ground (SOG) between points P and Q:
    SOG = boat speed + current speed (vectors)
    SOG direction must be the direction of PQ, hence
    SOG vector is the intersection of line PQ with the circle
    centered at the vector of current (u, v) with radius |boat_speed|
    """

    # Get equation for line PQ
    dx = q[0] - p[0]
    dy = q[1] - p[1]
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

    assert d >= 0, "There exist no real solutions between points {} and {}".format(p, q)

    if d == 0:
        x = (-b + sqrt(d)) / (2 * a)
        y = alpha * x
        SOG = sqrt(x ** 2 + y ** 2)
    else:
        rt = sqrt(d)
        root1 = (-b + rt) / (2 * a)
        root2 = (-b - rt) / (2 * a)
        if copysign(1, root1) == copysign(1, dx) and copysign(1, root2) == copysign(1, dx):
            # If both roots return resultant vector in right direction,
            # use resultant vector with greatest length
            y1, y2 = alpha * root1, alpha * root2
            v1, v2 = sqrt(root1 ** 2 + y1 ** 2), sqrt(root2 ** 2 + y2 ** 2)
            if v1 > v2:
                SOG = v1
            else:
                SOG = v2
        else:
            if copysign(1, root2) == copysign(1, dx):
                x = root2
            else:
                x = root1
            y = alpha * x
            SOG = sqrt(x ** 2 + y ** 2)

    return SOG


def edge_x_geos(p1, p2, rtree_idx, geos):
    line_bounds = (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))

    # Returns the geometry indices of the minimum bounding rectangles of polygons that intersect the edge bounds
    mbr_intersections = list(rtree_idx.intersection(line_bounds))
    if mbr_intersections:
        # Create LineString if there is at least one minimum bounding rectangle intersection
        line_string = LineString([(p1[0], p1[1]), (p2[0], p2[1])])

        # For every mbr intersection check if its polygon is actually intersect by the edge
        for idx in mbr_intersections:
            if geos[idx].intersects(line_string):
                return True
    return False
