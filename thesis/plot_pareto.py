import case_studies.plot_results as plot_results
import main
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pprint

from datetime import datetime
from pathlib import Path


# KC_((123, 26), (139, 34))_depart2015_03_15_iters5_BFalse_ECA1_NSGA2
# KC_((139, 34), (123, 26))_depart2014_09_15_iters5_BFalse_ECA1_NSGA2

currentDir = Path('D:/output/current/rawListColab_1504/KC constant')
os.chdir(currentDir)

fileList = os.listdir()
fileList = [file for file in fileList if '1729' in file]

pp = pprint.PrettyPrinter()
pp.pprint(fileList)

DIR = Path('D:/')
SPEED = 'var'  # 'constant' or 'var'
speedOps = ['insert', 'move', 'delete'] if SPEED == 'constant' else ['insert', 'move', 'speed', 'delete']
par = {'mutationOperators': speedOps}
ECA_F = 1
DEPTH = False
PLANNER = main.RoutePlanner(inputParameters=par, bathymetry=DEPTH, ecaFactor=ECA_F,
                            criteria={'minimalTime': True, 'minimalCost': True})

exp = 'current'

for i, rawFile in enumerate(fileList):
    print(rawFile)
    depDate = datetime(2015, 3, 15) if '2015' in rawFile else datetime(2014, 9, 17)

    with open(rawFile, 'rb') as f:
        rawList = pickle.load(f)

    for j, raw in enumerate(rawList):
        proc, raw = PLANNER.post_process(raw, inclEnvironment={exp: depDate})

        statisticsPlotter = plot_results.StatisticsPlotter(raw, DIR=DIR)
        frontFig, _ = statisticsPlotter.plot_fronts()
        statsFig, _ = statisticsPlotter.plot_stats()

        if exp == 'current':
            cData = PLANNER.evaluator.currentOperator.data
            lons0 = np.linspace(-179.875, 179.875, 1440)
            lats0 = np.linspace(-89.875, 89.875, 720)
            currentDict = {'u': cData[0, 0], 'v': cData[1, 0], 'lons': lons0, 'lats': lats0}
        else:
            currentDict = None

        weatherDate = depDate if exp == 'weather' else None
        routePlotter = plot_results.RoutePlotter(DIR, proc, rawResults=raw, vessel=PLANNER.vessel)
        routeFig, _ = routePlotter.results(initial=False, ecas=False, bathymetry=DEPTH, nRoutes='all',
                                           weatherDate=weatherDate, current=currentDict, colorbar=True)
        # frontFig.savefig('front_{}_v{}.png'.format(i, j), dpi=300)
        # statsFig.savefig('stats_{}_v{}.png'.format(i, j), dpi=300)
        routeFig.savefig('route_{}_v{}.png'.format(i, j), dpi=300)

        plt.close('all')