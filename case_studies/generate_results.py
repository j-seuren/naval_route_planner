import case_studies.plot_results as plot_results
import matplotlib.pyplot as plt
import main
import numpy as np
import os
import pandas as pd
import pickle
import time

# from case_studies.demos import create_currents
from datetime import datetime
from pathlib import Path
from support import locations

os.chdir('..')

DIR = Path('D:/')
SPEED = True
ITERS = 1
parameters = {'mutpb': 0.61,
              'widthRatio': 7.5e-4,
              'radius': 0.39,
              'gauss': False,
              'mutationOperators': ['insert', 'move', 'delete'] if SPEED else ['insert', 'move', 'speed', 'delete'],
              'gen': 2,  # Min number of generations
              'n': 10}  # Population size
PLANNER = main.RoutePlanner(inputParameters=parameters, criteria={'minimalTime': True, 'minimalCost': True})


def get_df(procList):
    L_time = [proc['routeResponse'][0]['distance'] for proc in procList]
    L_fuel = [proc['routeResponse'][1]['distance'] for proc in procList]
    C_time = [proc['routeResponse'][0]['fuelCost'] for proc in procList]
    C_fuel = [proc['routeResponse'][1]['fuelCost'] for proc in procList]
    T_time = [proc['routeResponse'][0]['travelTime'] for proc in procList]
    T_fuel = [proc['routeResponse'][1]['travelTime'] for proc in procList]
    compTime = [proc['computationTime'] for proc in procList]
    table = list(zip(compTime, T_fuel, T_time, C_fuel, C_time, L_fuel, L_time))
    df = pd.DataFrame(table, columns=['compTime', 'T_fuel', 'T_time', 'C_fuel', 'C_time', 'L_fuel', 'L_time']).T
    mean, std = df.mean(axis=1), df.std(axis=1)
    df['mean'], df['std'] = mean, std

    return df


def single_experiment(experiment, startEnd, depDate, depS, saveFig=True):
    current = True if experiment == 'current' else False
    weather = True if experiment == 'weather' else False
    ecas = True if experiment == 'ecas' else False

    rawList, procList = [], []
    for i in range(ITERS):
        t0 = time.time()
        raw = PLANNER.compute(startEnd, startDate=depDate, recompute=True, weather=weather, current=current)
        t1 = time.time() - t0
        proc, raw = PLANNER.post_process(raw)
        proc['computationTime'] = t1
        rawList.append(raw)
        procList.append(proc)

        if saveFig:
            statisticsPlotter = plot_results.StatisticsPlotter(raw)
            frontFig, _ = statisticsPlotter.plot_fronts()
            statsFig, _ = statisticsPlotter.plot_stats()

            if experiment == 'current':
                cData = PLANNER.evaluator.currentOperator.data
                lons0 = np.linspace(-179.875, 179.875, 1440)
                lats0 = np.linspace(-89.875, 89.875, 720)
                currentDict = {'u': cData[0, 0], 'v': cData[1, 0], 'lons': lons0, 'lats': lats0}
            else:
                currentDict = None

            weatherDate = depDate if experiment == 'weather' else None
            routePlotter = plot_results.RoutePlotter(proc, rawResults=raw, vessel=PLANNER.vessel)
            routeFig, _ = routePlotter.results(initial=False, ecas=ecas, nRoutes=4, weatherDate=weatherDate,
                                               current=currentDict, colorbar=True)

            for name, fig in {'front': frontFig, 'stats': statsFig, 'routes': routeFig}.items():
                fig.savefig(DIR / "output/{}/figures/{}_{}_iters{}.pdf".format(experiment, startEnd, name, i, depS))

            plt.close('all')

    with open(DIR / 'output/{}/raw/{}_{}_iters{}'.format(experiment, startEnd, ITERS, depS)) as f:
        pickle.dump(rawList, f)

    return get_df(procList)


def multiple_experiments(startEnds, experiment, depDates=None):
    depDates = [None] if depDates is None else depDates
    for d, depDate in enumerate(depDates):
        print('date {} of {}'.format(d, len(depDates)))
        depS = '' if depDate is None else 'depart' + depDate.strftime('%Y_%m_%d')
        speedS = 'constant' if SPEED else 'var'
        writer = pd.ExcelWriter(DIR / 'output/{}/{}_gulf_{}Speed.xlsx'.format(experiment, depS, speedS))
        dfSummary = pd.DataFrame(index=['compTime', 'T_fuel', 'T_time', 'C_fuel', 'C_time', 'L_fuel', 'L_time'])
        starts, ends = zip(*startEnds)

        for i, start in enumerate(starts):
            for j, end in enumerate(ends):
                print('start, end {}, {} of {}'.format(i, j, len(ends)))
                df = single_experiment(experiment, (start, end), depDate, depS, saveFig=True)
                locS = str((i, j))
                df.to_excel(writer, sheet_name=locS)
                df.to_csv(DIR / 'output/{}/single_files/{}_{}_gulf_{}Speed.xlsx'.format(experiment, depS, locS, speedS))
                dfSummary[locS + '_mean'] = df['mean']
                dfSummary[locS + '_std'] = df['std']

        dfSummary.to_excel(writer, sheet_name='summary')
        writer.save()
    print('DONE TESTING')


# Test current
currentDepartures = [datetime(2014, 10, 28),
                     datetime(2014, 11, 11),
                     datetime(2014, 11, 25),
                     datetime(2014, 4, 20),
                     datetime(2015, 5, 4),
                     datetime(2015, 5, 18)]
currentStartEnds = zip(locations['westLocations'], locations['eastLocations'])
# multiple_experiments(currentStartEnds, 'current', currentDepartures)

# Test ECA
ecaStartEnds = zip([locations['ECA1: Jacksonville']], [locations['ECA2: New York']])
# multiple_experiments(ecaStartEnds, 'ecas')

departure = datetime(2019, 3, 1)
departureS = 'depart' + departure.strftime('%Y_%m_%d')

single_experiment('weather', (locations['Caribbean Sea'], locations['North UK']), datetime(2019, 3, 1), departureS,
                  saveFig=True)
plt.show()

