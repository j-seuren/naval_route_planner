import case_studies.plot_results as plot_results
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from case_studies.demos import create_currents
from main import RoutePlanner
from support import locations

os.chdir('..')

PARAMETERS = {'mutpb': 0.61, 'widthRatio': 7.5e-4, 'radius': 0.39, 'gauss': True,
              'gen': 2000,  # Number of generations
              'n': 300}  # Population size


START_END = (locations['Current1'], locations['Current2'])
# START_DATE = datetime.datetime(2016, 1, 1)
START_DATE = datetime.datetime(2019, 3, 1)
ITERATIONS = 1


def eca_test(parameters, iterations):
    startEnd = (locations['ECA1: Jacksonville'], locations['ECA2: New York'])
    routePlanner = RoutePlanner(inputParameters=parameters, criteria={'minimalTime': True, 'minimalCost': True})

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


def current_test(startEnd, parameters, iterations, saveFig=True):
    routePlanner = RoutePlanner(inputParameters=parameters, criteria={'minimalTime': True, 'minimalCost': True})

    rawList, procList = [], []
    for i in range(iterations):
        startTime = time.time()
        raw = routePlanner.compute(startEnd, startDate=datetime.datetime(2016, 1, 1), recompute=True, current=True,
                                   seed=1)
        proc, raw = routePlanner.post_process(raw)
        rawList.append(raw)
        procList.append(proc)

        statisticsPlotter = plot_results.StatisticsPlotter(rawList[0])
        statisticsPlotter.plot_fronts()
        statisticsPlotter.plot_stats()

        da = routePlanner.evaluator.currentOperator.da
        lons0 = np.linspace(-179.875, 179.875, 1440)
        lats0 = np.linspace(-89.875, 89.875, 720)
        currentDict = {'u': da[0,0], 'v': da[1,0], 'lons': lons0, 'lats': lats0}

        routePlotter = plot_results.RoutePlotter(procList[0], rawResults=rawList[0], vessel=routePlanner.vessel)
        fig, ax = routePlotter.results(initial=False, ecas=False, nRoutes=5, current=currentDict, colorbar=True)

        print("--- %s seconds ---" % (time.time() - startTime))
        if saveFig:
            timestamp = datetime.datetime.now().strftime("%m%d_%H-%M-%S")
            fig.savefig("thesis/figures/{}_current_demo_ITER{}.pdf".format(timestamp, i))


def general_test(startEnd, startDate, parameters, plot=True):
    current, weather = False, False

    startTime = time.time()
    routePlanner = RoutePlanner(inputParameters=parameters, criteria={'minimalTime': True, 'minimalCost': True})
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


for west in locations['westLocations']:
    for east in locations['eastLocations']:
        current_test((west, east), PARAMETERS, ITERATIONS, saveFig=True)

plt.show()
# eca_test(PARAMETERS, ITERATIONS)
# general_test(START_END, START_DATE, PARAMETERS)

