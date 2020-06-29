import great_circle
import ocean_current

from shapely.geometry import LineString


class Evaluator:
    def __init__(self,
                 vessel,
                 prepared_polys,
                 rtree_idx,
                 geod,
                 start_date='20160101',
                 del_s=100,
                 del_sc=15,
                 include_currents=True):
        self.vessel = vessel
        self.prepared_polys = prepared_polys
        self.rtree_idx = rtree_idx
        self.geod = geod
        self.date = start_date                                          # Start date for initializing ocean currents
        self.del_s = del_s                                              # Maximum segment length (nautical miles)
        self.del_sc = del_sc                                            # Maximum segment length for current calculation
        self.include_currents = include_currents
        if include_currents:
            self.u, self.v, _, _ = ocean_current.read_netcdf(self.date)
        else:
            self.u, self.v = None, None
        self.dist_cache = dict()
        self.feas_cache = dict()
        self.points_cache = dict()

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
                # Split edge in segments (seg) of max seg_length in km
                points = self.points_cache.get(k, False)
                if not points:  # Never steps in IF-statement
                    print('computes points')
                    points = great_circle.points(p1[0], p1[1], p2[0], p2[1], e_dist, self.geod, self.del_s)
                    self.points_cache[k] = points
                lons, lats = points[0], points[1]
                e_travel_time = 0.0
                for i in range(len(lons) - 1):
                    p1, p2 = (lons[i], lats[i]), (lons[i + 1], lats[i + 1])
                    seg_dist = great_circle.distance(p1[0], p1[1], p2[0], p2[1], self.geod)
                    seg_travel_time = ocean_current.get_edge_travel_time(p1, p2, boat_speed, seg_dist, self.u, self.v)
                    e_travel_time += seg_travel_time
            else:
                e_travel_time = e_dist / boat_speed
            edge_fuel_consumption = self.vessel.fuel_rates[boat_speed] * e_travel_time  # Tons

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
            q1_x, q1_y = lons[i], lats[i]
            q2_x, q2_y = lons[i + 1], lats[i + 1]
            line_bounds = (min(q1_x, q2_x), min(q1_y, q2_y), max(q1_x, q2_x), max(q1_y, q2_y))

            # Returns the geometry indices of the minimum bounding rectangles of polygons that intersect the edge bounds
            mbr_intersections = list(self.rtree_idx.intersection(line_bounds))
            if mbr_intersections:
                # Create LineString if there is at least one minimum bounding rectangle intersection
                line_string = LineString([(q1_x, q1_y), (q2_x, q2_y)])

                # For every mbr intersection check if its polygon is actually intersect by the edge
                for idx in mbr_intersections:
                    if self.prepared_polys[idx].intersects(line_string):
                        self.feas_cache[k] = 0
                        return False
        self.feas_cache[k] = 1
        return True
