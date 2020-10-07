import datetime
import numpy as np
import os
import pandas as pd
import support
import weather

from functools import lru_cache, wraps
from math import cos, sin, sqrt, radians, pow
from pathlib import Path
from shapely.geometry import LineString, Point
# from case_studies.demos import create_currents


def delta_penalty(func):
    @wraps(func)
    def wrapper(self, individual, *args, **kwargs):
        return func(self, individual, *args, **kwargs) if self.feasible(individual) else self.delta
    return wrapper


class Evaluator:
    def __init__(self,
                 vessel,
                 landRtree,
                 ecaRtree,
                 bathRtree,
                 ecaFactor,
                 geod,
                 criteria,
                 parameters,
                 DIR,
                 startDate=None):
        self.vessel = vessel            # Vessel class instance
        self.landRtree = landRtree      # R-tree spatial index dictionary for shorelines
        self.ecaRtree = ecaRtree        # R-tree spatial index dictionary for ECAs
        self.bathRtree = bathRtree      # R-tree spatial index dictionary for bathymetry
        self.ecaFactor = ecaFactor      # Multiplication factor for ECA fuel
        self.startDate = startDate      # Start date of voyage
        self.segLengthF = parameters['segLengthF']    # Max segment length (for feasibility)
        self.segLengthC = parameters['segLengthC']    # Max segment length (for currents)
        self.geod = geod                # Geod class instance
        self.currentOp = None
        self.weatherOp = None
        self.inclWeather = None
        self.inclCurrent = None
        self.bathymetry = False if bathRtree is None else True
        self.criteria = criteria
        self.revertOutput = not criteria['minimalTime']
        self.penaltyValue = parameters['penaltyValue']
        self.includePenalty = False
        self.delta = (1e+20,) * len(criteria)
        self.DIR = DIR

    def set_classes(self, inclCurr, inclWeather, startDate, nDays):
        self.startDate = startDate
        self.inclWeather = inclWeather
        self.inclCurrent = inclCurr
        if inclCurr:
            assert isinstance(startDate, datetime.date), 'Set start date'
            self.currentOp = weather.CurrentOperator(startDate, nDays, DIR=self.DIR, KC=self.vessel.name == 'Tanaka')
            # self.currentOperator.da = create_currents(nDays)
        if inclWeather:
            assert isinstance(startDate, datetime.date), 'Set start date'
            self.weatherOp = weather.WindOperator(startDate, nDays, DIR=self.DIR)

    @delta_penalty
    def evaluate(self, ind, revert=None, includePenalty=None):
        self.includePenalty = self.bathymetry if includePenalty is None else includePenalty
        revert = self.revertOutput if revert is None else revert
        hours = cost = 0.

        for e in range(len(ind) - 1):
            # Leg endpoints and boat speed
            p1, p2 = sorted((ind[e][0], ind[e+1][0]))
            speedKnots = ind[e][1]

            # Leg travel time and fuel cost
            legHours = self.leg_hours(p1, p2, hours, speedKnots)
            legCost = self.vessel.fuelCostPerDay[speedKnots] * legHours / 24.  # x1000 EUR or USD

            # If leg intersects ECA increase fuel consumption by ecaFactor
            if geo_x_geos(self.ecaRtree, p1, p2):
                legCost *= self.ecaFactor

            if self.includePenalty:
                timePenalty, costPenalty = self.e_feasible(p1, p2)
                legHours += timePenalty
                legCost += costPenalty

            # Increment objective values
            hours += legHours
            cost += legCost



        days = hours / 24.
        if revert:
            return cost, days
        return days, cost

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
        lineSegs = np.stack([lons, lats]).T

        # since we know the difference between any two points, we can use this to find wrap arounds
        dLonMax = self.segLengthF * 10. / 60.

        # calculate distances and compare with max allowable distance
        dLons = np.abs(np.diff(lons))
        cuts = np.where(dLons > dLonMax)[0]

        # if there are any cut points, cut them and begin again at the next point
        for i, cut in enumerate(cuts):
            # create new vertices with a nan in between and set those as the path's vertices
            segs = np.concatenate([lineSegs[:cut+1, :],
                                   [[np.nan, np.nan]],
                                   lineSegs[cut+1:, :]]
                                  )
            lineSegs = segs

        # Check feasibility of each segment, and penalize sailing through shallow water
        timePenalty = costPenalty = 0.
        for q1, q2 in zip(lineSegs[:-1, :], lineSegs[1:, :]):
            if not np.isnan(np.sum([q1, q2])):  # Skip segments crossing datum line
                if geo_x_geos(self.landRtree, q1, q2):  # If segment crosses land obstacle
                    return False
                if self.includePenalty and not geo_x_geos(self.bathRtree, q1, q2):
                    timePenalty += self.penaltyValue['time']
                    costPenalty += self.penaltyValue['cost']
        return timePenalty, costPenalty

    @lru_cache(maxsize=None)
    def leg_hours(self, p1, p2, startHours, speedKnots):
        nauticalMiles = self.geod.distance(p1, p2)
        if self.inclCurrent or self.inclWeather:
            # Split leg in segments (seg) of segLengthC in nautical miles
            lons, lats = self.geod.points(p1, p2, nauticalMiles, self.segLengthC)
            segPoints = list(zip(lons, lats))
            legHours = 0.
            for segment in zip(segPoints[:-1], segPoints[1:]):
                q1, q2 = segment
                segmentTT = self.calc_seg_hours(q1, q2, speedKnots, startHours + legHours)
                legHours += segmentTT
        else:
            legHours = nauticalMiles / speedKnots
        return legHours  # Travel time in days

    def calc_seg_hours(self, p1, p2, speedKnots, currentHours):
        try:
            now = self.startDate + datetime.timedelta(hours=currentHours)
        except TypeError:
            raise TypeError("Date {} ; currHours {}".format(self.startDate, currentHours))
        nauticalMiles, bearingDeg = self.geod.distance(p1, p2, bearing=True)

        # Coordinates of middle point of leg
        # If segment crosses datum line (-180 degrees), choose p1 as middle point
        lon, lat = p1 if abs(p1[0] - p2[0]) > 300 else ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        if self.inclWeather:
            # Beaufort number (BN) and true wind direction (TWD) at (lon, lat)
            BN, windDeg = self.weatherOp.get_grid_pt_wind(now, lon, lat)
            headingDeg = bearingDeg
            speedKnots = self.vessel.reduced_speed(windDeg, headingDeg, BN, speedKnots)

        if self.inclCurrent:
            # Eastward and northward current velocities at (lon, lat)
            uKnots, vKnots = self.currentOp.get_grid_pt_current(now, lon, lat)

            # Calculate speed over ground (actualSpeed)
            actualSpeedKnots = calc_sog(radians(bearingDeg), uKnots, vKnots, speedKnots)
        else:
            actualSpeedKnots = speedKnots

        return nauticalMiles / actualSpeedKnots


class Vessel:
    def __init__(self, fuelPrice, name='Fairmaster_2', shipLoading='normal', DIR=Path('D:/'), speeds=None):
        self.name = name
        vesselTableFP = DIR / 'data/speed_table.xlsx'

        if os.path.exists(vesselTableFP):
            df = pd.read_excel(vesselTableFP, sheet_name=self.name)
            df = df[df['Loading'] == shipLoading].round(1)
            self.fuelCostPerDay = pd.Series(df.Fuel.values * fuelPrice / 1000., index=df.Speed).to_dict()
            if speeds is None:
                self.speeds = list(self.fuelCostPerDay.keys())
            elif isinstance(speeds, float) or isinstance(speeds, int):
                self.speeds = [speeds]
            elif isinstance(speeds, list):
                self.speeds = speeds
            else:
                raise ValueError('Provide speed profile as list, float or integer')

        else:  # Use empirical formula for fuel consumption
            print('Approximate fuel consumption with nonlinear function')

            def fc(v):
                """Approximate fuel consumption per day nonlinear function from (Psaraftis and Kontovas, 2013)
                    Arguments:
                        v: Vessel speed in knots
                    Returns:
                        Fuel consumption in tonnes per day
                  """
                return 5.466e-4 * v**3 * 24

            if isinstance(speeds, float) or isinstance(speeds, int):
                self.speeds = [speeds]
            elif isinstance(speeds, list):
                self.speeds = speeds
            else:
                raise ValueError('Vessel speed profile not provided')
            self.fuelCostPerDay = {speed: fc(speed) for speed in self.speeds}

        # Set parameters for ship speed reduction calculations
        Lpp = 152.9  # Ship length between perpendiculars [m]
        vol = 27150  # Displaced volume [m^3]
        blockCoefficient = 0.8
        self.speed_reduction = SemiEmpiricalSpeedReduction(blockCoefficient, shipLoading, Lpp, vol, DIR)
        self.reduced_speed = self.speed_reduction.reduced_speed


class SemiEmpiricalSpeedReduction:
    """Based on Kwon's method for calculating the reduction of ship speed as a function of wind direction and speed.
    Kwon, Y.J., 2008. Speed loss due to added resistance in wind and waves """

    def __init__(self, block, shipLoading, Lpp, volume, DIR):

        coefficientTableFP = DIR / 'data/kwons_method_coefficient_tables.xlsx'
        # Weather direction reduction table
        df = pd.read_excel(coefficientTableFP, sheet_name='direction_reduction_coefficient')
        self.directionDF = df[['a', 'b', 'c']].to_numpy()
        # self.windAngleBins = [30, 60, 150, 180]

        # Speed reduction formula coefficients: a, b, c
        if shipLoading == 'ballast':
            blockBins = np.asarray([0.75, 0.8, 0.85])
        else:
            blockBins = np.asarray([0.6, 0.65, 0.7, 0.75, 0.8, 0.85])

        roundedBlock = blockBins[support.find_closest(blockBins, block)]
        df = pd.read_excel(coefficientTableFP, sheet_name='speed_reduction_coefficient')
        abc = df.loc[(df['block_coefficient'] == roundedBlock) & (df['ship_loading'] == shipLoading)]
        self.aB, self.bB, cB = float(abc['a']), float(abc['b']), float(abc['c'])
        g = 9.81  # gravitational acceleration [m/s^2]
        knotToMs = 0.514444
        self.FnConstant = knotToMs / sqrt(Lpp * g)
        self.cB = cB * self.FnConstant ** 2

        # Ship coefficient formula coefficients: a, b
        df = pd.read_excel(coefficientTableFP, sheet_name='ship_form_coefficient')
        ab = df.loc[(df['ship_type'] == 'all') & (df['ship_loading'] == shipLoading)]
        self.aU, bU = float(ab['a']), float(ab['b'])

        self.formDenominator = 1 / (bU * pow(volume, (2 / 3)))

    def reduced_speed(self, windDeg, headingDeg, BN, speedKnots):
        """ The wind effect, presented as speed loss, compares the
        speed of the ship in varying actual sea conditions to the ship's
        expected speed in still water conditions.
        formC is the
        directionC is the ,
        speedC is is the  """

        # General speed loss in head wind condition
        formC = self.aU * BN + pow(BN, 6.5) * self.formDenominator

        # Weather direction reduction factor
        windAngle = abs((windDeg - headingDeg + 180) % 360 - 180)  # [0, 180] degrees
        if windAngle < 30:
            abc = self.directionDF[0]
        elif windAngle < 60:
            abc = self.directionDF[1]
        elif windAngle < 150:
            abc = self.directionDF[2]
        else:
            abc = self.directionDF[3]
        a, b, c = abc[0], abc[1], abc[2]
        directionC = (a + b * pow(BN + c, 2)) / 2

        # Correction factor for block coefficient and Froude number
        Fn = speedKnots * self.FnConstant
        speedC = self.aB + self.bB * Fn + self.cB * pow(speedKnots, 2)
        speedLoss = max(min((directionC * speedC * formC) / 100, 0.99), -0.3)

        reducedSpeedKnots = speedKnots * (1 - speedLoss)
        return reducedSpeedKnots


def calc_sog(bearingRad, Se, Sn, V):
    """
    Determine speed over ground (sog) between points p1 and p2 in knots.
    see thesis section Ship Speed
    """

    # Calculate speed over ground (sog)
    sinB, cosB = sin(bearingRad), cos(bearingRad)
    try:
        result = Se * sinB + Sn * cosB + sqrt(V * V - (Se * cosB - Sn * sinB) ** 2)
    except ValueError:
        result = V
    return result


def geo_x_geos(treeDict, p1, p2=None):
    p2 = p1 if p2 is None else p2
    minx, maxx = (p1[0], p2[0]) if p1[0] < p2[0] else (p2[0], p1[0])
    miny, maxy = (p1[1], p2[1]) if p1[1] < p2[1] else (p2[1], p1[1])
    bounds = (minx, miny, maxx, maxy)

    # Return the geometry indices whose bounds intersect the query bounds
    indices = treeDict['rtree'].intersection(bounds)
    if indices:
        shapelyObject = Point(tuple(p1)) if p2 is None else LineString([tuple(p1), tuple(p2)])
        for idx in indices:
            geometry = treeDict['geometries'][idx]
            if p2 is None and geometry.contains(shapelyObject):
                return True
            elif geometry.intersects(shapelyObject):
                return True
    return False


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pprint

    from geodesic import Geodesic
    from data_config.navigable_area import NavigableAreaGenerator

    os.chdir('...')

    def test_kwon():
        _heading = 0
        _vessel = Vessel(300)
        _speed = 15.2  # knots
        windDegs = np.linspace(0, 180, 181)
        BNs = np.linspace(0, 12, 13)
        newSpeeds = np.zeros([len(windDegs), len(BNs)])
        for ii, _windDeg in enumerate(windDegs):
            for jj, _BN in enumerate(BNs):
                newSpeeds[ii, jj] = _vessel.reduced_speed(_windDeg, _heading, _BN, _speed)

        # Print
        pp = pprint.PrettyPrinter()
        pp.pprint(windDegs)
        pp.pprint(BNs)
        pp.pprint(newSpeeds)

        # Plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(BNs, windDegs)
        ax.plot_surface(X, Y, newSpeeds)
        plt.show()

    parameters = {
        # Navigation area parameters
        'avoidAntarctic': True, 'avoidArctic': True, 'res': 'i',  # Resolution of shorelines
        'penaltyValue': {'time': -5, 'cost': -1},
        'graphDens': 4,  # Recursion level graph
        'graphVarDens': 6,  # Variable recursion level graph
        'splits': 3,  # Threshold for split_polygon (val 3 yields best performance)

        # MOEA parameters
        'n': 322,  # Population size
        'nBar': 100,  # Local archive size (M-PAES)
        'cxpb': 0.81,  # Crossover probability (NSGAII, SPEA2)
        'mutpb': 0.28,  # Mutation probability (NSGAII, SPEA2)
        'nMutations': 9,  # Max. number of mutations per selected individual
        'cr_trials': 5,  # Max recombination trials (M-PAES)
        'l_fails': 3,  # Max fails (M-PAES)
        'l_opt': 5,  # Max moves (M-PAES)

        # Stopping parameters
        'maxEvaluations': None, 'gen': 100,  # Minimal number of generations
        'maxGDs': 33,  # Max length of generational distance list
        'minVar': 1e-5,  # Minimal variance of generational distance list

        # Mutation parameters
        'mutationOperators': ['speed', 'insert', 'move', 'delete'],  # Operators to be included
        'widthRatio': 1.5,  # 7.5e-4 obtained from hyp param tuning
        'radius': 0.4,  # 0.39 obtained from hyp param tuning
        'scaleFactor': 0.1,  # Scale factor for Exponential distribution
        'delFactor': 1.1,  # Factor of deletions
        'gauss': False,  # Use Gaussian mutation for insert and move operators

        # Evaluation parameters
        'segLengthF': 15,  # Length of linear approx. of great circle track for feasibility
        'segLengthC': 15  # same for ocean currents and wind along route
    }

    def test_evaluator(startDate):
        _dir = Path('D:/')
        vessel = Vessel(300)
        areaGenerator = NavigableAreaGenerator(parameters=parameters, DIR=_dir)
        landRtree = areaGenerator.get_shoreline_rtree()
        ecaRtree = areaGenerator.get_eca_rtree()
        bathRtree = areaGenerator.get_bathymetry_rtree()
        ecaFactor = 1.0
        geod = Geodesic()
        criteria = {'minimalTime': -5, 'minimalCost': -1}
        evaluator = Evaluator(vessel, landRtree, ecaRtree, bathRtree, ecaFactor, geod, criteria, parameters, _dir,
                              startDate=startDate)

        return evaluator

    # evl = test_evaluator(datetime.datetime(2015, 6, 21))
    # evl.calc_seg_hours((179, 0), (-179, 0), 9.9, 100)

    def test_calc_sog():
        _V = 100
        bearingRads = np.radians(np.linspace(-180, 170, 36))

        Se = -10
        Sn = 0
        output2 = [calc_sog(bearingRad, Se, Sn, _V) for bearingRad in bearingRads]

        plt.scatter(np.degrees(bearingRads), output2)

        plt.show()
