import main
import os

from datetime import datetime
from pathlib import Path
from test_general import multiple_experiments
from support import locations

import pickle

DIR = Path('D:/')
parameters = {'DIR': DIR,
              'bathymetry': False,
              'weather': False,
              'current': True,
              'ecaFactor': 1.5593,
              'exp': 'MOEA',
              'iterations': 50,
              'speed': 'var',
              'ref': False
              }

criteria = {'minimalTime': True, 'minimalCost': True}

inputMOEA_w = {'instance': 'MOEA', 'input': {'from': [('Ply', locations['Plymouth'])
                                                      ],
                                             'to': [('Hav', locations['Havana'])
                                                    ],
                                             'departureDates': [datetime(2013, 9, 24)]}}

inputMOEA_b = {'instance': 'MOEA', 'input': {'from': [('Rot', locations['Rotterdam'])
                                                      ],
                                             'to': [('Hou', locations['Houston'])
                                                    ],
                                             'departureDates': [datetime(2016, 1, 1)]}}

inputMOEA_c = {'instance': 'MOEA', 'input': {'from': [('Sal', locations['Salvador'])
                                                      ],
                                             'to': [('Par', locations['Paramaribo'])
                                                    ],
                                             'departureDates': [datetime(2014, 11, 25)]}}

inputDict = inputMOEA_c

speedOps = ['insert', 'move', 'delete'] if parameters['speed'] == 'constant' else ['speed', 'insert', 'move', 'delete']
par = {'mutationOperators': speedOps, 'n': 100, 'maxEvaluations': 21000}

for MOEA in ['MPAES', 'NSGA2', 'SPEA2']:
    with open('D:/evals/{}'.format(MOEA), 'wb') as fh:
        pickle.dump(1, fh)
    parameters['MOEA'] = MOEA
    planner = main.RoutePlanner(inputParameters=par,
                                bathymetry=parameters['bathymetry'],
                                ecaFactor=parameters['ecaFactor'],
                                criteria=criteria,
                                seeds=range(parameters['iterations']))

    # Create directories
    _dir = DIR / 'output' / (parameters['exp'] + '_08_11') / inputDict['instance']
    genDir = _dir / '{}_{}SP_B{}_ECA{}/{}'.format(parameters['MOEA'],
                                                  parameters['speed'],
                                                  parameters['bathymetry'],
                                                  parameters['ecaFactor'],
                                                  parameters['iterations'])
    createDirs = [genDir / 'tables/csv', genDir / 'figures', genDir / 'raw']
    [os.makedirs(directory) for directory in createDirs if not os.path.exists(directory)]
    multiple_experiments(inputDict, planner, parameters, genDir=genDir, seed=True)