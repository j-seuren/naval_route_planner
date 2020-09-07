import numpy as np
import operator
import os
import pandas as pd
import support
import weather

from datetime import timedelta
from functools import lru_cache
from math import cos, sin, tan, atan2, sqrt, log, pi, copysign, degrees, pow
from shapely.geometry import LineString, Point


class Evaluator:
    def __init__(self,
                 vessel,
                 treeDict,
                 ecaTreeDict,
                 ecaFactor,
                 fuelPrice,
                 geod,
                 revertOutput,
                 startDate=None,
                 segLengthF=100,
                 segLengthC=15):
        self.vessel = vessel            # Vessel class instance
        self.treeDict = treeDict        # R-tree spatial index dictionary for shorelines
        self.ecaTreeDict = ecaTreeDict  # R-tree spatial index dictionary for ECAs
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
        self.revertOutput = revertOutput

    def set_classes(self, inclCurr, inclWeather, startDate, nDays):
        self.inclWeather = inclWeather
        self.inclCurrent = inclCurr
        if inclCurr:
            self.currentOperator = weather.CurrentOperator(startDate, nDays)
        if inclWeather:
            self.weatherOperator = weather.WeatherOperator(startDate, nDays)

    def evaluate(self, ind, revert=None):
        if revert is None:
            revert = self.revertOutput

        if not self.feasible(ind):
            return 1e+20, 1e+20

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
            if geo_x_geos(self.ecaTreeDict, p1, p2):
                edgeFC *= self.ecaFactor

            # Increment objective values
            TT += edgeTT
            FC += edgeFC

        if revert:
            return FC, TT
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
                if geo_x_geos(self.treeDict, q1, q2):
                    return False
        return True

    def e_tt(self, p1, p2, tt, boatSpeed):
        dist = self.geod.distance(p1, p2)
        if self.inclCurrent or self.inclWeather:
            # Split edge in segments (seg) of del_sc in nautical miles
            lons, lats = self.geod.points(p1, p2, dist, self.segLengthC)
            edgeTT = 0.0
            for i in range(len(lons) - 1):
                q1, q2 = (lons[i], lats[i]), (lons[i+1], lats[i+1])
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

        bearing_rad = calc_bearing(p1, p2)

        if self.inclWeather:
            # Beaufort number (BN) and true wind direction (TWD) at (lon, lat)
            BN, windDir = self.weatherOperator.get_grid_pt_wind(now, lon, lat)
            heading = degrees(bearing_rad)
            boatSpeed = self.vessel.reduced_speed(windDir, heading, BN, boatSpeed)
        if self.inclCurrent:
            # Easting and Northing currents at (lon, lat)
            u, v = self.currentOperator.get_grid_pt_current(now, lon, lat)

            # Calculate speed over ground
            sog = calc_sog(bearing_rad, u, v, boatSpeed)
        else:
            sog = boatSpeed
        return dist / sog


class Vessel:
    def __init__(self, name='Fairmaster', shipLoading='normal'):
        self.name = name
        vesselTableFP = os.path.abspath('data/speed_table.xlsx')
        vesselTable = pd.read_excel(vesselTableFP, sheet_name=self.name)
        self.speeds = [round(speed, 1) for speed in vesselTable['Speed']]
        self.fuel_rates = {speed: round(vesselTable['Fuel'][i], 1) for i, speed in enumerate(self.speeds)}

        # Set parameters for ship speed reduction calculations
        Lpp = 320  # Ship length between perpendiculars [m]
        B = 58  # Ship breadth [m]
        D = 20.8  # Ship draft [m]
        vol = 312622  # Displaced volume [m^3]
        blockCoefficient = vol / (Lpp * B * D)  # Block coefficient
        self.speed_reduction = SemiEmpiricalSpeedReduction(blockCoefficient, shipLoading, Lpp, vol)

    def reduced_speed(self, windDir, heading, BN, boatSpeed):
        return self.speed_reduction.reduced_speed(windDir, heading, BN, boatSpeed)


class SemiEmpiricalSpeedReduction:
    """Based on Kwon's method for calculating the reduction of ship speed as a function of wind direction and speed.
    Kwon, Y.J., 2008. Speed loss due to added resistance in wind and waves """

    def __init__(self, block, shipLoading, Lpp, volume):
        self.g = 9.81  # gravitational acceleration [m/s^2]
        self.Lpp = Lpp  # Length between perpendiculars [m]
        self.vol = volume  # Displaced volume [m^3]

        coefficientTableFP = 'data/kwons_method_coefficient_tables.xlsx'
        # Weather direction reduction table
        df = pd.read_excel(coefficientTableFP, sheet_name='direction_reduction_coefficient')
        self.directionDF = df[['a', 'b', 'c']].to_numpy()
        self.windAngleBins = [30, 60, 150, 180.1]

        # Speed reduction formula coefficients: a, b, c
        if shipLoading == 'ballast':
            blockBins = np.asarray([0.75, 0.8, 0.85])
        else:
            blockBins = np.asarray([0.6, 0.65, 0.7, 0.75, 0.8, 0.85])

        roundedBlock = blockBins[support.find_closest(blockBins, block)]
        df = pd.read_excel(coefficientTableFP, sheet_name='speed_reduction_coefficient')
        abc = df.loc[(df['block_coefficient'] == roundedBlock) & (df['ship_loading'] == shipLoading)]
        self.aB, self.bB, self.cB = float(abc['a']), float(abc['b']), float(abc['c'])

        # Ship coefficient formula coefficients: a, b
        df = pd.read_excel(coefficientTableFP, sheet_name='ship_form_coefficient')
        ab = df.loc[(df['ship_type'] == 'all') & (df['ship_loading'] == shipLoading)]
        self.aU, self.bU = float(ab['a']), float(ab['b'])

    def speed_reduction_coefficient(self, Fn):
        return self.aB + self.bB * Fn + self.cB * Fn ** 2

    def direction_reduction_coefficient(self, BN, windAngle):
        windAngleIdx = int(np.digitize(abs(windAngle), self.windAngleBins))
        abc = self.directionDF[windAngleIdx]
        a, b, c = abc[0], abc[1], abc[2]

        return (a + b * pow(BN + c, 2)) / 2

    def ship_form_coefficient(self, BN):
        return self.aU * BN + pow(BN, 6.5) / (self.bU * pow(self.vol, (2 / 3)))

    def reduced_speed(self, windDir, heading, BN, designSpeed):
        """ The weather effect, presented as speed loss, compares the
        speed of the ship in varying actual sea conditions to the ship's
        expected speed in still water conditions.
        formC is the
        directionC is the ,
        speedC is is the  """

        # General speed loss in head weather condition
        formC = self.ship_form_coefficient(BN)

        # Weather direction reduction factor
        windAngle = (windDir - heading + 180) % 360 - 180  # [-180, 180] degrees
        directionC = self.direction_reduction_coefficient(BN, windAngle)

        # Correction factor for block coefficient and Froude number
        Fn = (designSpeed * 0.514444) / sqrt(self.Lpp * self.g)
        speedC = self.speed_reduction_coefficient(Fn)
        relativeSpeedLoss = max(min((directionC * speedC * formC) / 100, 0.99), -0.3)

        actualSpeed = designSpeed * (1 - relativeSpeedLoss)
        return actualSpeed


def calc_bearing(p1, p2):
    """ Calculate bearing in degrees"""
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


def geo_x_geos(treeDict, p1, p2=None):
    if p2 is None:
        geo = Point(p1)
    else:
        geo = LineString([p1, p2])

    # Return a list of all geometries in the R-tree whose extents
    # intersect the extent of geom
    extent_intersections = treeDict['tree'].query(geo)
    if extent_intersections:
        # Check if any geometry in extent_intersections actually intersects line
        for geom in extent_intersections:
            geomIdx = treeDict['indexByID'][id(geom)]
            prepGeom = treeDict['polys'][geomIdx]
            if p2 is None and prepGeom.contains(geo):
                return True
            elif prepGeom.intersects(geo):
                return True

    return False


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pprint

    os.chdir('..')

    _heading = 0
    _vessel = Vessel()
    _speed = 16.8  # knots
    windDirs = np.linspace(0, 180, 181)
    BNs = np.linspace(0, 12, 13)
    newSpeeds = np.zeros([len(windDirs), len(BNs)])
    for ii, _windDir in enumerate(windDirs):
        for jj, _BN in enumerate(BNs):
            newSpeeds[ii, jj] = _vessel.reduced_speed(_speed, _BN, _windDir, _heading)

    # Print
    pp = pprint.PrettyPrinter()
    pp.pprint(windDirs)
    pp.pprint(BNs)
    pp.pprint(newSpeeds)

    # Plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(BNs, windDirs)
    ax.plot_surface(X, Y, newSpeeds)
    plt.show()
