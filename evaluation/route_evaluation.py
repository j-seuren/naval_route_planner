import numpy as np
import operator

from datetime import timedelta
from functools import lru_cache
from math import cos, sin, tan, atan2, sqrt, log, pi, copysign
from shapely.geometry import LineString


class Evaluator:
    def __init__(self,
                 vessel,
                 tree,
                 eca_tree,
                 eca_f,
                 vlsfo_price,
                 start_date,
                 gc,
                 incl_curr=True,
                 del_sf=100,
                 del_sc=15):
        self.vessel = vessel            # Vessel class instance
        self.tree = tree                # R-tree spatial index for shorelines
        self.eca_tree = eca_tree        # R-tree spatial index for ECAs
        self.eca_f = eca_f              # Multiplication factor for ECA fuel
        self.start_date = start_date    # Start date of voyage
        self.del_sf = del_sf            # Max segment length (for feasibility)
        self.del_sc = del_sc            # Max segment length (for currents)
        self.incl_curr = incl_curr      # Boolean for including currents
        self.gc = gc                    # Geod class instance
        self.current_data = None
        self.vlsfo_price = vlsfo_price

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
            e_fc = self.vessel.fuel_rates[boat_speed] * e_tt * self.vlsfo_price

            # If edge intersects SECA increase fuel consumption by eca_f
            if edge_x_geos(p1, p2, self.eca_tree):
                e_fc *= self.eca_f

            # Increment objective values
            tt += e_tt
            fc += e_fc

        return tt, fc

    def feasible(self, ind):
        for i in range(len(ind)-1):
            p1, p2 = sorted((ind[i][0], ind[i+1][0]))
            if not self.e_feasible(p1, p2):
                return False
        return True

    @lru_cache(maxsize=None)
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
            # create new vertices with a nan in between and set those as the path's vertices
            verts = np.concatenate([vertices[:cut+1, :],
                                    [[np.nan, np.nan]],
                                    vertices[cut+1:, :]]
                                   )
            vertices = verts
        for i in range(len(vertices)-1):
            if not np.isnan(np.sum(vertices[i:i+2, :])):
                q1, q2 = tuple(vertices[i, :]), tuple(vertices[i+1, :])
                if edge_x_geos(q1, q2, self.tree):
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
        else:
            e_tt = dist / boat_speed
        return e_tt / 24.0  # Travel time in days

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


def calc_sog(p1, p2, Se, Sn, V):
    """
    Determine speed over ground (sog) between points p1 and p2 in knots.
    see thesis section Ship Speed
    """
    # Convert degrees to radians
    [(lam1, phi1), (lam2, phi2)] = np.radians([p1, p2])

    # Latitude difference projected on Mercator projection
    dPsi = log(tan(pi / 4 + phi2 / 2) / tan(pi / 4 + phi1 / 2))

    dLam = lam2 - lam1  # Longitude difference
    if abs(dLam) > pi:  # take shortest route: dLam < PI
        dLam = dLam - copysign(2 * pi, dLam)

    b = atan2(dLam, dPsi)  # Bearing

    # Calculate speed over ground (sog)
    sinB, cosB = sin(b), cos(b)
    sog = Se * sinB + Sn * cosB + sqrt(V * V - (Se * cosB - Sn * sinB) ** 2)
    return sog


def edge_x_geos(p1, p2, tree):
    line = LineString([p1, p2])

    # Return a list of all geometries in the R-tree whose extents
    # intersect the extent of geom
    extent_intersections = tree.query(line)
    if extent_intersections:
        # Check if any geometry in extent_intersections actually intersects line
        for geom in extent_intersections:
            if geom.intersects(line):
                return True
    return False


if __name__ == '__main__':
    _p1, _p2 = (0.5, 4), (4.5, 1)
    _Sx, _Sy = -1, -0.5
    _V = 2.015564437

    sog2 = calc_sog(_p1, _p2, _Sx, _Sy, _V)
    print(sog2)
