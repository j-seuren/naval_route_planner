import case_studies.plot_results as plot_results
import datetime
import matplotlib.pyplot as plt
import main
import numpy as np
import os
import pandas as pd
import time

# from case_studies.demos import create_currents
from pathlib import Path
from support import locations

os.chdir('..')

DIR = Path('D:/')

PARAMETERS = {'mutpb': 0.61,
              'widthRatio': 7.5e-4,
              'radius': 0.39,
              'gauss': True,
              'mutationOperators': ['insert', 'move', 'delete'],  # Operators to be included
              'gen': 2,  # Number of generations
              'n': 300}  # Population size


START_END = (locations['Current1'], locations['Current2'])
START_DATE = datetime.datetime(2016, 1, 1)
START_DATE_STRING = START_DATE.strftime('%Y_%m_%d')
# START_DATE = datetime.datetime(2019, 3, 1)
ITERATIONS = 1


def eca_test(parameters, iterations):
    startEnd = (locations['ECA1: Jacksonville'], locations['ECA2: New York'])
    routePlanner = main.RoutePlanner(inputParameters=parameters, criteria={'minimalTime': True, 'minimalCost': True})

    startTime = time.time()
    rawList, procList = [], []
    for _ in range(iterations):
        raw = routePlanner.compute(startEnd,
                                   recompute=False,
                                   current=False,
                                   weather=False,
                                   seed=1)
        proc, raw = routePlanner.post_process(raw)

        rawList.append(raw)
        procList.append(proc)

    statisticsPlotter = plot_results.StatisticsPlotter(rawList[0])
    statisticsPlotter.plot_fronts()
    statisticsPlotter.plot_stats()

    routePlotter = plot_results.RoutePlotter(procList[0], vessel=routePlanner.vessel)
    routePlotter.results(initial=True, ecas=True, bathymetry=True)

    print("--- %s seconds ---" % (time.time() - startTime))

    return procList


def current_test(startEnd, startDate, parameters, iterations, saveFig=True):
    routePlanner = main.RoutePlanner(inputParameters=parameters, criteria={'minimalTime': True, 'minimalCost': True})

    rawList, procList = [], []
    for i in range(iterations):
        startTime = time.time()
        raw = routePlanner.compute(startEnd, startDate=startDate, recompute=True, current=True, seed=1)
        totTime = time.time() - startTime
        proc, raw = routePlanner.post_process(raw)
        proc['computationTime'] = totTime
        rawList.append(raw)
        procList.append(proc)

        statisticsPlotter = plot_results.StatisticsPlotter(rawList[0])
        statisticsPlotter.plot_fronts()
        statisticsPlotter.plot_stats()

        da = routePlanner.evaluator.currentOperator.data
        lons0 = np.linspace(-179.875, 179.875, 1440)
        lats0 = np.linspace(-89.875, 89.875, 720)
        currentDict = {'u': da[0, 0], 'v': da[1, 0], 'lons': lons0, 'lats': lats0}

        if saveFig:
            routePlotter = plot_results.RoutePlotter(procList[0], rawResults=rawList[0], vessel=routePlanner.vessel)
            fig, _ = routePlotter.results(initial=False, ecas=False, nRoutes=5, current=currentDict, colorbar=True)
            timestamp = datetime.datetime.now().strftime("%m%d_%H-%M-%S")
            fig.savefig(DIR / "output/figures/{}_current_demo_ITER{}.pdf".format(timestamp, i))
            plt.close('all')

    distancesTime = [proc['routeResponse'][0]['distance'] for proc in procList]
    distancesFuel = [proc['routeResponse'][1]['distance'] for proc in procList]
    costsTime = [proc['routeResponse'][0]['fuelCost'] for proc in procList]
    costsFuel = [proc['routeResponse'][1]['fuelCost'] for proc in procList]
    timesTime = [proc['routeResponse'][0]['travelTime'] for proc in procList]
    timesFuel = [proc['routeResponse'][1]['travelTime'] for proc in procList]
    compTimes = [proc['computationTime'] for proc in procList]

    zipped = list(zip(compTimes, timesFuel, timesTime, costsFuel, costsTime, distancesFuel, distancesTime))
    df = pd.DataFrame(zipped, columns=['compTimes', 'timesFuel', 'timesTime', 'costsFuel', 'costsTime',
                                          'distancesFuel', 'distancesTime']).T
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)

    return df


def general_test(startEnd, startDate, parameters, plot=True):
    current, weather = False, False

    startTime = time.time()
    routePlanner = main.RoutePlanner(inputParameters=parameters, criteria={'minimalTime': True, 'minimalCost': True})
    raw = routePlanner.compute(startEnd, recompute=True, startDate=startDate, current=current, weather=weather, seed=1)

    print("--- %s seconds ---" % (time.time() - startTime))

    if plot:
        weatherDate = startDate if weather else None
        proc, raw = routePlanner.post_process(raw)

        statisticsPlotter = plot_results.StatisticsPlotter(raw)
        statisticsPlotter.plot_fronts()
        statisticsPlotter.plot_stats()

        routePlotter = plot_results.RoutePlotter(proc, vessel=routePlanner.vessel)
        routePlotter.results(weatherDate=weatherDate,
                             initial=True,
                             bathymetry=True,
                             ecas=True)


writer = pd.ExcelWriter(DIR / 'output/currents/gulf_stream_routes_DATE{}_constantSpeed.xlsx'.format(START_DATE_STRING))
dfSummary = pd.DataFrame(index=['compTimes', 'timesFuel', 'timesTime', 'costsFuel', 'costsTime', 'distancesFuel',
                                'distancesTime'])

for i, west in enumerate(locations['westLocations']):
    for j, east in enumerate(locations['eastLocations']):
        _df = current_test((west, east), START_DATE, PARAMETERS, ITERATIONS, saveFig=False)
        string = str((i, j))
        _df.to_excel(writer, sheet_name=string)
        _df.to_csv(DIR / 'output/currents/single_files/{}_gulf_stream_routes_DATE{}_constantSpeed.xlsx'.format(string,
                                                                                                               START_DATE_STRING))

        dfSummary[string + '_mean'] = _df['mean']
        dfSummary[string + '_std'] = _df['std']


dfSummary.to_excel(writer, sheet_name='summary')

writer.save()

# df = pd.DataFrame.from_dict(procDict)
# df.to_excel(DIR / 'output/currents/gulf_stream_routes.xlsx')

# eca_test(PARAMETERS, ITERATIONS)
# general_test(START_END, START_DATE, PARAMETERS)

