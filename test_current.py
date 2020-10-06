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

TEST_SPEED = True

eastLocations = [('E1', (-50.0, 38.0)), ('E2', (-52.5, 42.0)), ('E3', (-55.0, 46.0))]
westLocations = [('W1', (-72.0, 32.0)), ('W2', (-73.0, 35.5)), ('W3', (-74.0, 39.0))]
inputGulf = {'instance': 'Gulf', 'input': {'from': [], 'to': []}}
for west in westLocations:
    for east in eastLocations:
        inputGulf['input']['from'].append(west)
        inputGulf['input']['to'].append(east)
        inputGulf['input']['from'].append(east)
        inputGulf['input']['to'].append(west)

inputGulf['input']['departureDates'] = [datetime(2014, 11, 15), datetime(2015, 5, 15)]

inputKC = {'instance': 'KC', 'input': {'from': [('K', locations['KeelungC']),
                                                ('T', locations['Tokyo'])],
                                       'to': [('T', locations['Tokyo']),
                                              ('K', locations['KeelungC'])],
                                       'departureDates': [datetime(2014, 9, 15),
                                                          datetime(2015, 3, 15)]}}

inputKC_2 = {'instance': 'KC0', 'input': {'from': [('K', locations['KeelungC']),
                                                   ('T', locations['Tokyo'])],
                                          'to': [('T', locations['Tokyo']),
                                                 ('K', locations['KeelungC'])],
                                          'departureDates': [datetime(2014, 9, 15)]}}

inputGulf0 = {'instance': 'Gulf0', 'input': {'from': [('3', (-73.2, 36.2))],
                                             'to': [('1', (-51., 39.6))],
                                             'departureDates': [datetime(2014, 11, 25)]}}

inputDict = inputGulf
criteria = {'minimalTime': True, 'minimalCost': True}

for current in [True, False]:
    for speed in ['constant', 'var']:
        if not TEST_SPEED and speed == 'constant':
            continue
        speedOps = ['insert', 'move', 'delete'] if speed == 'constant' else ['speed', 'insert', 'move', 'delete']
        par = {'mutationOperators': speedOps, 'gen': 100}

        nSpeeds = [0, -1] if speed == 'constant' else [0]
        for speedIdx in range(nSpeeds):
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