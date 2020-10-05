import main
import os

from datetime import datetime
from pathlib import Path
from support import locations
from test_general import multiple_experiments

DIR = Path('D:/')
parameters = {'DIR': DIR,
              'bathymetry': False,
              'current': False,
              'ecaFactor': 1.0,
              'exp': 'weather',
              'iterations': 1,
              'MOEA': 'NSGA2',
              }

inputWeather = {'instance': 'WTH', 'input': {'from': [
                                                      # ('Ny', locations['New York']),  # Kuhlemann
                                                      # ('K', locations['Keelung']),  # Lin2013
                                                      ('No', locations['Normandy']),  # Shao2012
                                                      # ('No', locations['Normandy']),  # Vettor2016
                                                      # ('V', locations['Valencia'])  # Vettor2016
                                                      ],
                                             'to': [
                                                    # ('P', locations['Paramaribo']),
                                                    # ('S', locations['San Francisco']),
                                                    ('Ny', locations['New York']),
                                                    # ('Mi', locations['Miami']),
                                                    # ('Ma', locations['Malta'])
                                                    ],
                                             'departureDates': [
                                                                # datetime(2017, 9, 4),
                                                                # datetime(2011, 5, 28),  # DEP 0000Z 28 May 2011, ETA 0000Z 11 June 2011
                                                                datetime(2011, 1, 25),  # DEP 03:00 p.m. ETA: 00:30 p.m. 30/01/2011
                                                                # datetime(2015, 6, 21),  # June 21, 2015 at 00:00
                                                                # datetime(2015, 6, 21)  # June 21, 2015 at 00:00
                                                                ]}
                }


inputDict = inputWeather
criteria = {'minimalTime': True, 'minimalCost': True}

for weather in [True, False]:
    for speed in [
                  # 'constant',
                  'var'
                 ]:
        speedOps = ['insert', 'move', 'delete'] if speed == 'constant' else ['speed', 'insert', 'move', 'delete']
        par = {'mutationOperators': speedOps}
        planner = main.RoutePlanner(inputParameters=par,
                                    bathymetry=parameters['bathymetry'],
                                    ecaFactor=parameters['ecaFactor'],
                                    criteria=criteria)
        parameters['weather'] = weather
        # Create directories
        _dir = DIR / 'output' / parameters['exp'] / inputDict['instance']
        genDir = _dir / '{}_{}SP_B{}_ECA{}/{}'.format(parameters['MOEA'],
                                                      speed,
                                                      parameters['bathymetry'],
                                                      parameters['ecaFactor'],
                                                      parameters['iterations'])
        parameters['ref'] = 'R_' if not weather else ''
        createDirs = [genDir / 'tables/csv', genDir / 'figures', genDir / 'raw']
        [os.makedirs(directory) for directory in createDirs if not os.path.exists(directory)]
        multiple_experiments(inputDict, planner, parameters, genDir=genDir)
