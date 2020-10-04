import main
import os

from datetime import datetime
from pathlib import Path
from support import locations
from test_general import multiple_experiments

DIR = Path('D:/')
parameters = {'DIR': DIR,
              'bathymetry': False,
              'weather': False,
              'ecaFactor': 1.0,
              'exp': 'current',
              'iterations': 5,
              'MOEA': 'NSGA2',
              }


eastLocations = [(-51., 39.6), (-52., 41.2), (-53., 42.8), (-54., 44.4)]
westLocations = [(-72.4, 33.4), (-72.8, 34.8), (-73.2, 36.2), (-73.6, 37.6)]
inputGulf = {'instance': 'Gulf', 'input': {'from': [], 'to': []}}
for i, west in enumerate(westLocations):
    for j, east in enumerate(eastLocations):
        inputGulf['input']['from'].append(('{}'.format(i + 1), west))
        inputGulf['input']['to'].append(('{}'.format(j + 1), east))

inputGulf['input']['departureDates'] = [datetime(2014, 10, 28), datetime(2014, 11, 11), datetime(2014, 11, 25),
                                        datetime(2014, 4, 20), datetime(2015, 5, 4), datetime(2015, 5, 18)]

inputKC = {'instance': 'KC', 'input': {'from': [('K', locations['KeelungC']), ('T', locations['Tokyo'])],
                                       'to': [('T', locations['Tokyo']), ('K', locations['KeelungC'])],
                                       'departureDates': [datetime(2014, 9, 15), datetime(2015, 3, 15)]}}

inputKC_2 = {'instance': 'KC0', 'input': {'from': [('K', locations['KeelungC']), ('T', locations['Tokyo'])],
                                       'to': [('T', locations['Tokyo']), ('K', locations['KeelungC'])],
                                       'departureDates': [datetime(2014, 9, 15)]}}

inputGulf0 = {'instance': 'Gulf0', 'input': {'from': [('3', (-73.2, 36.2))], 'to': [('1', (-51., 39.6))],
                                            'departureDates': [datetime(2014, 11, 25)]}}

inputDict = inputGulf
criteria = {'minimalTime': True, 'minimalCost': True}

for current in [True, False]:
    for speed in ['constant'  #, 'var'
                  ]:
        speedOps = ['insert', 'move', 'delete'] if speed == 'constant' else ['speed', 'insert', 'move', 'delete']
        par = {'mutationOperators': speedOps}

        nSpeeds = 12 if speed == 'constant' else 1
        for _ in range(1):
            speedIdx = 11
            planner = main.RoutePlanner(speedIdx=speedIdx,
                                        inputParameters=par,
                                        bathymetry=parameters['bathymetry'],
                                        ecaFactor=parameters['ecaFactor'],
                                        criteria=criteria)
            parameters['current'] = current
            speedStr = speed + str(speedIdx) if speed == 'constant' else speed
            # Create directories
            _dir = DIR / 'output' / parameters['exp'] / inputDict['instance']
            genDir = _dir / '{}_{}SP_B{}_ECA{}/{}'.format(parameters['MOEA'],
                                                          speedStr,
                                                          parameters['bathymetry'],
                                                          parameters['ecaFactor'],
                                                          parameters['iterations'])
            parameters['ref'] = 'R_' if not current else ''
            createDirs = [genDir / 'tables/csv', genDir / 'figures', genDir / 'raw']
            [os.makedirs(directory) for directory in createDirs if not os.path.exists(directory)]
            multiple_experiments(inputDict, planner, parameters, genDir=genDir)
