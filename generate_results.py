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

# os.chdir('')

DIR = Path('D:/')
SPEED = 'constant'  # 'constant' or 'var'
ITERS = 5
parameters = {'mutationOperators': ['insert', 'move', 'delete'] if SPEED == 'constant' else ['insert', 'move', 'speed', 'delete']}
ECA_FACTOR = 1.2
BATHYMETRY = True
PLANNER = main.RoutePlanner(inputParameters=parameters, bathymetry=BATHYMETRY, ecaFactor=ECA_FACTOR,
                            criteria={'minimalTime': True, 'minimalCost': True})
CURRENT = True


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
    rawListFP = DIR / 'output/{}/raw/{}_{}_iters{}_B{}_ECA{}'.format(experiment, startEnd, ITERS, depS, BATHYMETRY, ECA_FACTOR)
    if os.path.exists(rawListFP):
        return None

    current = True if experiment == 'current' and CURRENT else False
    weather = True if experiment == 'weather' else False
    ecas = True if ECA_FACTOR != 1 else False

    rawList, procList = [], []
    for i in range(ITERS):
        print('ITERATION {} of {}'.format(i+1, ITERS))
        t0 = time.time()
        raw = PLANNER.compute(startEnd, startDate=depDate, recompute=True, weather=weather, current=current)
        t1 = time.time() - t0
        proc, raw = PLANNER.post_process(raw)
        proc['computationTime'] = t1
        rawList.append(raw)
        procList.append(proc)

        if saveFig:
            statisticsPlotter = plot_results.StatisticsPlotter(raw, DIR=DIR)
            frontFig, _ = statisticsPlotter.plot_fronts()
            statsFig, _ = statisticsPlotter.plot_stats()

            if current:
                cData = PLANNER.evaluator.currentOperator.data
                lons0 = np.linspace(-179.875, 179.875, 1440)
                lats0 = np.linspace(-89.875, 89.875, 720)
                currentDict = {'u': cData[0, 0], 'v': cData[1, 0], 'lons': lons0, 'lats': lats0}
            else:
                currentDict = None

            weatherDate = depDate if experiment == 'weather' else None
            routePlotter = plot_results.RoutePlotter(DIR, proc, rawResults=raw, vessel=PLANNER.vessel)
            routeFig, _ = routePlotter.results(initial=False, ecas=ecas, bathymetry=BATHYMETRY, nRoutes=4,
                                               weatherDate=weatherDate, current=currentDict, colorbar=True)

            for name, fig in {'front': frontFig, 'stats': statsFig, 'routes': routeFig}.items():
                fig.savefig(DIR / "output/{}/figures/{}_{}_iters{}_B{}_ECA{}.pdf".format(experiment, startEnd, name, i, depS, BATHYMETRY, ECA_FACTOR))
            plt.close('all')

    with open(DIR / 'output/{}/raw/{}_{}_iters{}_B{}_ECA{}'.format(experiment, startEnd, ITERS, depS, BATHYMETRY, ECA_FACTOR), 'wb') as f:
        pickle.dump(rawList, f)

    return get_df(procList)


def multiple_experiments(startEnds, experiment, depDates=None):

    def init_experiment(WRITER, DF_SUMMARY, DEP_DATE, DEP_S, LOC_S, START, END):
        fp = DIR / 'output/{}/single_files/{}_{}_gulf_{}Speed.csv'.format(experiment, DEP_S, LOC_S, SPEED)
        DF = single_experiment(experiment, (START, END), DEP_DATE, DEP_S, saveFig=True)
        if DF is None:
            DF = pd.read_csv(fp)
        else:
            DF.to_excel(WRITER, sheet_name=LOC_S)
            DF.to_csv(fp)
        DF_SUMMARY[LOC_S + '_mean'] = DF['mean']
        DF_SUMMARY[LOC_S + '_std'] = DF['std']
        return DF_SUMMARY.T

    depDates = [None] if depDates is None else depDates
    for d, depDate in enumerate(depDates):
        print('date {} of {}'.format(d+1, len(depDates)))
        depS = '' if depDate is None else 'depart' + depDate.strftime('%Y_%m_%d')
        writer = pd.ExcelWriter(DIR / 'output/{}/{}_gulf_{}Speed.xlsx'.format(experiment, depS, SPEED))
        dfSummary = pd.DataFrame(columns=['compTime', 'T_fuel', 'T_time', 'C_fuel', 'C_time', 'L_fuel', 'L_time'])

        if experiment == 'weather':
            locS = str(d)
            startEnds = list(startEnds)
            start, end = startEnds[d]
            dfSummary = init_experiment(writer, dfSummary, depDate, depS, locS, start, end)
        else:
            starts, ends = [startEnd[0] for startEnd in startEnds], [startEnd[1] for startEnd in startEnds]
            for i, start in enumerate(starts):
                for j, end in enumerate(ends):
                    print('start, end {}, {} of {}'.format(i+1, j+1, len(ends)))
                    locS = str((i, j))
                    dfSummary = init_experiment(writer, dfSummary, depDate, depS, locS, start, end)

        dfSummary.to_excel(writer, sheet_name='summary')
        writer.save()
    print('DONE TESTING')


def multiple_experiments2(inputDict, experiment):

    def init_experiment(WRITER, DF_SUMMARY, DEP_DATE, DEP_S, LOC_S, START, END):
        fp = DIR / 'output/{}/single_files/{}_{}_gulf_{}Speed_B{}_ECA{}.csv'.format(experiment, DEP_S, LOC_S, SPEED,
                                                                                    BATHYMETRY, ECA_FACTOR)
        DF = single_experiment(experiment, (START, END), DEP_DATE, DEP_S, saveFig=True)
        if DF is None:
            DF = pd.read_csv(fp)
        else:
            DF.to_excel(WRITER, sheet_name=LOC_S)
            DF.to_csv(fp)
        DF_SUMMARY[LOC_S + '_mean'] = DF['mean']
        DF_SUMMARY[LOC_S + '_std'] = DF['std']
        return DF_SUMMARY.T

    depDates = inputDict['departureDates']
    for d, depDate in enumerate(depDates):
        print('date {} of {}'.format(d+1, len(depDates)))
        depS = '' if depDate is None else 'depart' + depDate.strftime('%Y_%m_%d')
        writer = pd.ExcelWriter(DIR / 'output/{}/{}_gulf_{}Speed_B{}_ECA{}.xlsx'.format(experiment, depS, SPEED,
                                                                                        BATHYMETRY, ECA_FACTOR))
        dfSummary = pd.DataFrame(columns=['compTime', 'T_fuel', 'T_time', 'C_fuel', 'C_time', 'L_fuel', 'L_time'])

        for i, (startTup, endTup) in enumerate(zip(inputDict['from'], inputDict['to'])):
            startKey, start = startTup
            endKey, end = endTup
            print('location combination {} of {}'.format(i + 1, len(inputDict['from'])))
            locS = '{}{}'.format(startKey, endKey)
            dfSummary = init_experiment(writer, dfSummary, depDate, depS, locS, start, end)

        dfSummary.to_excel(writer, sheet_name='summary')
        writer.save()
    print('DONE TESTING')

# # Test weather
# # Weather locations
# weatherStarts = [
#     locations['New York'],  # to Paramaribo 2017, 9, 4
#     locations['Keelung'],   # Lin2013: departure 0000Z 28 May 2011, ETA 0000Z 11 June 2011
#     locations['Normandy'],  # Shao2012: Departure: 03:00 p.m. 25/01/2011 ETA: 00:30 p.m. 30/01/2011
#     locations['Normandy'],  # Vettor2016: June 21, 2015 at 00:00
#     locations['Valencia'],  # June 21, 2015 at 00:00
#     # locations['Thessaloniki']
#     ]
#
# weatherEnds = [
#     locations['Paramaribo'],
#     locations['San Francisco'],
#     locations['New York'],
#     locations['Miami'],
#     locations['Malta'],
#     # locations['Agios Nikolaos']
#     ]
#
# dateTimes = [
#     datetime(2017, 9,  4),
#     datetime(2011, 5, 28),
#     datetime(2011, 1, 25),
#     datetime(2015, 6, 21),
#     datetime(2015, 6, 21)
#     ]
#
# weatherStartEnds = zip(weatherStarts, weatherEnds)
# multiple_experiments(weatherStartEnds, 'weather', dateTimes)


# Test currents
currentDepartures = [datetime(2014, 10, 28),
                     datetime(2014, 11, 11),
                     datetime(2014, 11, 25),
                     datetime(2014, 4, 20),
                     datetime(2015, 5, 4),
                     datetime(2015, 5, 18)
                     ]

inputGulf = {'from': [],
             'to': [],
             'departureDates': []}
for date in currentDepartures:
    for west in locations['westLocations']:
        for east in locations['eastLocations']:
            inputGulf['from'].append(west)
            inputGulf['to'].append(east)
            inputGulf['departureDates'].append(date)


inputKC = {'from': [('K', locations['KeelungC']),
                    ('T', locations['Tokyo'])],
           'to': [('T', locations['Tokyo']),
                  ('K', locations['KeelungC'])],
           'departureDates': [datetime(2014, 9, 15),
                              datetime(2015, 3, 15)]}

multiple_experiments2(inputKC, 'current')  # TEST FOR [GC, CONSTANT] [CURRENT, CONSTANT] [CURRENT, VAR]

# # Test ECA
# ecaStartEnds = list(zip([locations['ECA1: Jacksonville']], [locations['ECA2: New York']]))
# multiple_experiments(ecaStartEnds, 'ecas')
