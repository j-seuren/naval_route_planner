import numpy as np
import operator
import weather

from datetime import timedelta
from functools import lru_cache
from math import cos, sin, tan, atan2, sqrt, log, pi, copysign
from shapely.geometry import LineString


class Evaluator:
    def __init__(self,
                 vessel,
                 tree,
                 ecaTree,
                 ecaFactor,
                 fuelPrice,
                 geod,
                 startDate=None,
                 segLengthF=100,
                 segLengthC=15):
        self.vessel = vessel            # Vessel class instance
        self.tree = tree                # R-tree spatial index for shorelines
        self.ecaTree = ecaTree          # R-tree spatial index for ECAs
        self.ecaFactor = ecaFactor      # Multiplication factor for ECA fuel
        self.startDate = startDate      # Start date of voyage
        self.segLengthF = segLengthF    # Max segment length (for feasibility)
        self.segLengthC = segLengthC    # Max segment length (for currents)
        self.geod = geod                # Geod class instance
        self.currentOperator = None
        self.weatherOperator = None
        self.fuelPrice = fuelPrice
        self.inclWeather = None
        self.inclCurrent = None

    def set_classes(self, inclCurr, inclWeather, startDate, nDays):
        self.inclWeather = inclWeather
        self.inclCurrent = inclCurr
        if inclCurr:
            self.currentOperator = weather.CurrentOperator(startDate, nDays)
        if inclWeather:
            self.weatherOperator = weather.WeatherOperator(startDate, nDays)

    def evaluate(self, ind):
        # Initialize variables
        TT = FC = 0.0

        for e in range(len(ind) - 1):
            # Get edge waypoints and edge boat speed
            p1, p2 = sorted((ind[e][0], ind[e + 1][0]))
            boatSpeed = ind[e][1]

            # Compute travel time over edge
            edgeTT = self.e_tt(p1, p2, TT, boatSpeed)

            # Compute fuel consumption over edge
            edgeFC = self.vessel.fuel_rates[boatSpeed] * edgeTT * self.fuelPrice

            # If edge intersects SECA increase fuel consumption by ecaFactor
            if edge_x_geos(p1, p2, self.ecaTree):
                edgeFC *= self.ecaFactor

            # Increment objective values
            TT += edgeTT
            FC += edgeFC

        return TT, FC

    def feasible(self, ind):
        for i in range(len(ind)-1):
            p1, p2 = sorted((ind[i][0], ind[i+1][0]))
            if not self.e_feasible(p1, p2):
                return False
        return True

    @lru_cache(maxsize=None)
    def e_feasible(self, p1, p2):
        dist = self.geod.distance(p1, p2)
        lons, lats = self.geod.points(p1, p2, dist, self.segLengthF)
        vertices = np.stack([lons, lats]).T

        # since we know the difference between any two points, we can use this to find wrap arounds on the plot
        maxDist = self.segLengthF * 10 / 60

        # calculate distances and compare with max allowable distance
        dists = np.abs(np.diff(lons))
        cuts = np.where(dists > maxDist)[0]

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

    def e_tt(self, p1, p2, tt, boatSpeed):
        dist = self.geod.distance(p1, p2)
        if self.inclCurrent or self.inclWeather:
            # Split edge in segments (seg) of del_sc in nautical miles
            lons, lats = self.geod.points(p1, p2, dist, self.segLengthC)
            edgeTT = 0.0
            for i in range(len(lons) - 1):
                q1, q2 = sorted(((lons[i], lats[i]), (lons[i+1], lats[i+1])))
                currentTT = tt + edgeTT
                segmentTT = self.seg_tt(q1, q2, boatSpeed, currentTT)
                edgeTT += segmentTT
        else:
            edgeTT = dist / boatSpeed
        return edgeTT / 24.0  # Travel time in days

    def seg_tt(self, p1, p2, boatSpeed, currentTT):
        now = self.startDate + timedelta(hours=currentTT)
        dist = self.geod.distance(p1, p2)

        # Coordinates of middle point of edge
        lon, lat = (item / 2 for item in map(operator.add, p1, p2))

        bearing = calc_bearing(p1, p2)

        if self.inclWeather:
            # Beaufort number (BN) and true wind direction (TWD) at (lon, lat)
            BN, TWD = self.weatherOperator.get_grid_pt_wind(now, lon, lat)
            heading = bearing
            boatSpeed = self.vessel.reduced_speed(boatSpeed, BN, TWD, heading)
        if self.inclCurrent:
            # Easting and Northing currents at (lon, lat)
            u, v = self.currentOperator.get_grid_pt_current(now, lon, lat)

            # Calculate speed over ground
            sog = calc_sog(bearing, u, v, boatSpeed)
        else:
            sog = boatSpeed
        return dist / sog


def calc_bearing(p1, p2):
    # Convert degrees to radians
    [(lam1, phi1), (lam2, phi2)] = np.radians([p1, p2])

    # Latitude difference projected on Mercator projection
    dPsi = log(tan(pi / 4 + phi2 / 2) / tan(pi / 4 + phi1 / 2))

    dLam = lam2 - lam1  # Longitude difference
    if abs(dLam) > pi:  # take shortest route: dLam < PI
        dLam = dLam - copysign(2 * pi, dLam)

    return atan2(dLam, dPsi)


def calc_sog(bearing, Se, Sn, V):
    """
    Determine speed over ground (sog) between points p1 and p2 in knots.
    see thesis section Ship Speed
    """

    # Calculate speed over ground (sog)
    sinB, cosB = sin(bearing), cos(bearing)
    return Se * sinB + Sn * cosB + sqrt(V * V - (Se * cosB - Sn * sinB) ** 2)


def edge_x_geos(p1, p2, tree, xExterior=False):
    line = LineString([p1, p2])

    # Return a list of all geometries in the R-tree whose extents
    # intersect the extent of geom
    extent_intersections = tree.query(line)
    if extent_intersections:
        # Check if any geometry in extent_intersections actually intersects line
        for geom in extent_intersections:
            if xExterior and geom.exterior.intersects(line):
                return True
            elif geom.intersects(line):
                return True
    return False
