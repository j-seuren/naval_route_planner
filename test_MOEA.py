import main
import os

from pathlib import Path
from test_current import inputKC
from test_general import multiple_experiments

DIR = Path('D:/')
parameters = {'DIR': DIR,
              'bathymetry': False,
              'weather': False,
              'current': True,
              'ecaFactor': 1.5593,
              'exp': 'MOEA',
              'iterations': 5,
              'speed': 'var',
              'ref': False
              }

criteria = {'minimalTime': True, 'minimalCost': True}
inputDict = inputKC

speedOps = ['insert', 'move', 'delete'] if parameters['speed'] == 'constant' else ['speed', 'insert', 'move', 'delete']
par = {'mutationOperators': speedOps, 'n': 100, 'maxEvaluations': 30000}
planner = main.RoutePlanner(inputParameters=par, bathymetry=parameters['bathymetry'],
                            ecaFactor=parameters['ecaFactor'], criteria=criteria)

for MOEA in ['NSGA2', 'SPEA2', 'MPAES']:
    parameters['MOEA'] = MOEA
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
    createDirs = [genDir / 'tables/csv', genDir / 'figures', genDir / 'raw']
    [os.makedirs(directory) for directory in createDirs if not os.path.exists(directory)]
    multiple_experiments(inputDict, planner, parameters, genDir=genDir)
