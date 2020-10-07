import matplotlib.pyplot as plt
import pprint

from haversine import haversine
from datetime import datetime, timedelta
from case_studies.plot_results import RoutePlotter
from support import locations
from main import RoutePlanner
from pathlib import Path

DIR = Path('D:/')

departure = datetime(2013, 9, 24)
waypoints = [{'lat': 50.328661,
               'lon': -4.162378,
               'speed': 15.0},
              {'lat': 50.25905886460962,
               'lon': -4.215019271105965,
               'speed': 14.3},
              {'lat': 49.880591590859,
               'lon': -5.078427082130161,
               'speed': 14.3},
              {'lat': 49.975738527210815,
               'lon': -5.777764566382415,
               'speed': 9.9},
              {'lat': 51.044753201846795,
               'lon': -10.13693723364505,
               'speed': 14.1},
              {'lat': 45.014872002471066,
               'lon': -51.80389925597136,
               'speed': 8.8},
              {'lat': 25.823790111178006,
               'lon': -80.04629470453106,
               'speed': 8.8},
              {'lat': 25.39113069047549,
               'lon': -80.18735634750871,
               'speed': 8.8},
              {'lat': 25.211163811622214,
               'lon': -80.31471674021553,
               'speed': 8.8},
              {'lat': 25.100433277168744,
               'lon': -80.39617543015429,
               'speed': 8.8},
              {'lat': 25.024645008590365,
               'lon': -80.45304532762857,
               'speed': 8.8},
              {'lat': 24.792054269686936,
               'lon': -80.69663598073879,
               'speed': 8.8},
              {'lat': 24.78826660396925,
               'lon': -80.73014311635906,
               'speed': 14.1},
              {'lat': 24.23048635190675,
               'lon': -81.29130791491666,
               'speed': 9.9},
              {'lat': 23.870840459330967,
               'lon': -81.69297793539558,
               'speed': 9.9},
              {'lat': 23.205105,
               'lon': -82.39594,
               'speed': None}]
# waypoints = [(wp['lon'], wp['lat']) for wp in waypoints]
def get_dates(plot=False):
    days = 0
    totDistance = 0
    dates = []
    for i, (wp1, wp2) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        distance = haversine(reversed((wp1['lon'], wp1['lat'])), reversed((wp2['lon'], wp2['lat'])), unit='nmi')
        days += distance / wp1['speed'] / 24.0
        totDistance += distance
        date = departure + timedelta(days=days)
        dates.append(date)
        print('waypoint', i+1, 'nautical miles', totDistance, 'days', days, 'date', date)

    if plot:
        calculate_waypoints(dates)


def calculate_waypoints(dates=None):
    _startEnd = (locations['Plymouth'], locations['Havana'])
    startDate = datetime(2013, 9, 24)
    kwargsPlanner = {'inputParameters': {}, 'ecaFactor': 1.0, 'criteria': {'minimalTime': True, 'minimalCost': True}}
    kwargsCompute = {'startEnd': _startEnd, 'startDate': startDate, 'recompute': False, 'current': False,
                     'weather': True, 'seed': 1, 'algorithm': 'NSGA2'}

    planner = RoutePlanner(**kwargsPlanner)
    rawResults = planner.compute(**kwargsCompute)

    procResults, rawResults = planner.post_process(rawResults)
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(procResults)

    if dates is not None:
        routePlotter = RoutePlotter(DIR, procResults, rawResults=rawResults, vessel=planner.vessel)
        for date in dates:
            fig, ax = plt.subplots()
            fig.suptitle('{}-{}'.format(date.month, date.day))
            routePlotter.results(ax, bathymetry=True, weatherDate=date, initial=True, ecas=False, colorbar=True)

            plt.show()


get_dates(plot=True)
