import main
import os

from pathlib import Path
from support import locations
from test_general import multiple_experiments

DIR = Path('D:/')
parameters = {'DIR': DIR,
              'current': False,
              'bathymetry': False,
              'weather': False,
              'exp': 'eca',
              'iterations': 5,
              'MOEA': 'NSGA2',
              'speed': 'var'  # 'constant' or 'var'
              }


ecaStart = [
            # ('K', locations['Kristiansand']),
            # ('Fle', locations['Flekkefjord']),
            # ('St', locations['Stavanger']),
            # ('B', locations['Bergen']),
            ('Flo', locations['Floro'])]
ecaEnd = [('Sa', locations['Santander'])] * len(ecaStart)

inputDict = {'instance': 'ECA', 'input': {'from': ecaStart,
                                          'to': ecaEnd, 'departureDates': [None]}}
criteria = {'minimalTime': True, 'minimalCost': True}

speedOps = ['insert', 'move', 'delete'] if parameters['speed'] == 'constant' else ['speed', 'insert', 'move', 'delete']
par = {'mutationOperators': speedOps, 'gen': 300}

for ecaFactor in [1.5593, 1.0]:
    parameters['ecaFactor'] = ecaFactor
    planner = main.RoutePlanner(inputParameters=par,
                                bathymetry=parameters['bathymetry'],
                                ecaFactor=parameters['ecaFactor'],
                                criteria=criteria)
    # Create directories
    _dir = DIR / 'output' / parameters['exp'] / inputDict['instance']
    genDir = _dir / '{}_{}SP_B{}_ECA{}/{}'.format(parameters['MOEA'],
                                                  parameters['speed'],
                                                  parameters['bathymetry'],
                                                  parameters['ecaFactor'],
                                                  parameters['iterations'])
    parameters['ref'] = 'R_' if ecaFactor == 1.0 else ''
    createDirs = [genDir / 'tables/csv', genDir / 'figures', genDir / 'raw']
    [os.makedirs(directory) for directory in createDirs if not os.path.exists(directory)]
    multiple_experiments(inputDict, planner, parameters, genDir=genDir)
