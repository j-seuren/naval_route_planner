import main
import os

from pathlib import Path
from support import inputKC
from test_general import multiple_experiments

DIR = Path('D:/')
parameters = {'DIR': DIR,
              'current': False,
              'weather': False,
              'ecaFactor': 1.5593,
              'exp': 'bathymetry',
              'iterations': 5,
              'MOEA': 'NSGA2',
              'speed': 'var'  # 'constant' or 'var'
              }

# inputDict = inputKC
criteria = {'minimalTime': True, 'minimalCost': True}

speedOps = ['insert', 'move', 'delete'] if parameters['speed'] == 'constant' else ['speed', 'insert', 'move', 'delete']
par = {'mutationOperators': speedOps}

for bathymetry in [True, False]:
    planner = main.RoutePlanner(inputParameters=par,
                                bathymetry=parameters['bathymetry'],
                                ecaFactor=parameters['ecaFactor'],
                                criteria=criteria)
    parameters['bathymetry'] = bathymetry
    # Create directories
    _dir = DIR / 'output' / parameters['exp'] / inputDict['instance']
    genDir = _dir / '{}_{}SP_B{}_ECA{}/{}'.format(parameters['MOEA'],
                                                  parameters['speed'],
                                                  parameters['bathymetry'],
                                                  parameters['ecaFactor'],
                                                  parameters['iterations'])
    parameters['ref'] = 'R_' if not bathymetry else ''
    createDirs = [genDir / 'tables/csv', genDir / 'figures', genDir / 'raw']
    [os.makedirs(directory) for directory in createDirs if not os.path.exists(directory)]
    multiple_experiments(inputDict, planner, parameters, genDir=genDir)
