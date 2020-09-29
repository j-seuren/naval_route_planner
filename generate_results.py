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
SPEED = 'var'  # 'constant' or 'var'
ITERS = 5
speedOps = ['insert', 'move', 'delete'] if SPEED == 'constant' else ['insert', 'move', 'speed', 'delete']
par = {'mutationOperators': speedOps}
ECA_F = 1
DEPTH = False
PLANNER = main.RoutePlanner(inputParameters=par, bathymetry=DEPTH, ecaFactor=ECA_F,
                            criteria={'minimalTime': True, 'minimalCost': True})
CURRENT = False
BL = 'BLANK_' if not CURRENT else ''


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


def single_experiment(exp, inst, startEnd, depDate, locS, depS, saveFig=True):
    rawListFP = DIR / 'output/{}/raw/{}{}_{}_{}_iters{}_B{}_ECA{}'.format(exp,
                                                                          BL,
                                                                          inst,
                                                                          locS,
                                                                          depS,
                                                                          ITERS,
                                                                          DEPTH,
                                                                          ECA_F)
    if os.path.exists(rawListFP):
        return None

    current = True if exp == 'current' and CURRENT else False
    weather = True if exp == 'weather' else False
    ecas = True if ECA_F != 1 else False

    rawList, procList = [], []
    for itr in range(ITERS):
        print('ITERATION {} of {}'.format(itr+1, ITERS))
        t0 = time.time()
        raw = PLANNER.compute(startEnd, startDate=depDate, recompute=True, weather=weather, current=current)
        t1 = time.time() - t0
        proc, raw = PLANNER.post_process(raw, inclEnvironment={exp: depDate})
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

            weatherDate = depDate if exp == 'weather' else None
            routePlotter = plot_results.RoutePlotter(DIR, proc, rawResults=raw, vessel=PLANNER.vessel)
            routeFig, _ = routePlotter.results(initial=False, ecas=ecas, bathymetry=DEPTH, nRoutes=4,
                                               weatherDate=weatherDate, current=currentDict, colorbar=True)

            for name, fig in {'front': frontFig, 'stats': statsFig, 'routes': routeFig}.items():
                fig.savefig(DIR / "output/{}/figures/{}{}_{}_{}_iter{}_B{}_ECA{}_{}.pdf".format(exp,
                                                                                                BL,
                                                                                                inst,
                                                                                                startEnd,
                                                                                                depS,
                                                                                                itr,
                                                                                                DEPTH,
                                                                                                ECA_F,
                                                                                                name))
            plt.close('all')

    with open(DIR / 'output/{}/raw/{}{}_{}_{}_iters{}_B{}_ECA{}'.format(exp,
                                                                        BL,
                                                                        inst,
                                                                        startEnd,
                                                                        depS,
                                                                        ITERS,
                                                                        DEPTH,
                                                                        ECA_F), 'wb') as f:
        pickle.dump(rawList, f)

    return get_df(procList)


def init_experiment(writer, experiment, inst, dfSummary, depDate, depS, locS, start, end):
    fp = DIR / 'output/{}/single_files/{}{}_{}_{}_{}_B{}_ECA{}.csv'.format(experiment,
                                                                           BL,
                                                                           inst,
                                                                           depS,
                                                                           locS,
                                                                           SPEED,
                                                                           DEPTH,
                                                                           ECA_F)
    df = single_experiment(experiment, inst, (start, end), depDate, locS, depS, saveFig=True)
    if df is None:
        df = pd.read_csv(fp)
    else:
        df.to_excel(writer, sheet_name=locS)
        df.to_csv(fp)
    dfSummary[locS + '_mean'] = df['mean']
    dfSummary[locS + '_std'] = df['std']
    return dfSummary.T


def multiple_experiments(inputDict, exp):
    inst = inputDict['instance']
    depDates = inputDict['input']['departureDates']
    for d, depDate in enumerate(depDates):
        print('date {} of {}'.format(d+1, len(depDates)))
        depS = '' if depDate is None else 'depart' + depDate.strftime('%Y_%m_%d')
        writer = pd.ExcelWriter(DIR / 'output/{}/{}{}_{}_{}_B{}_ECA{}.xlsx'.format(exp, BL, inst, depS, SPEED,
                                                                                   DEPTH, ECA_F))
        dfSummary = pd.DataFrame(columns=['compTime', 'T_fuel', 'T_time', 'C_fuel', 'C_time', 'L_fuel', 'L_time'])

        for routeIdx, (startTup, endTup) in enumerate(zip(inputDict['input']['from'], inputDict['input']['to'])):
            startKey, start = startTup
            endKey, end = endTup
            print('location combination {} of {}'.format(routeIdx + 1, len(inputDict['input']['from'])))
            locS = '{}{}'.format(startKey, endKey)
            dfSummary = init_experiment(writer, exp, inst, dfSummary, depDate, depS, locS, start, end)

        dfSummary.to_excel(writer, sheet_name='summary')
        writer.save()
    print('DONE TESTING')


# Test weather
# Weather locations
inputWeather = {'from': [('Ny', locations['New York']),  # Kuhlemann
                         ('K', locations['Keelung']),    # Lin2013
                         ('No', locations['Normandy']),  # Shao2012
                         ('No', locations['Normandy']),  # Vettor2016
                         ('V', locations['Valencia'])    # Vettor2016
                         ],
                'to': [('P', locations['Paramaribo']),
                       ('S', locations['San Francisco']),
                       ('Ny', locations['New York']),
                       ('Mi', locations['Miami']),
                       ('Ma', locations['Malta'])],
                'departureDates': [datetime(2017, 9,  4),
                                   datetime(2011, 5, 28),  # DEP 0000Z 28 May 2011, ETA 0000Z 11 June 2011
                                   datetime(2011, 1, 25),  # DEP 03:00 p.m. ETA: 00:30 p.m. 30/01/2011
                                   datetime(2015, 6, 21),  # June 21, 2015 at 00:00
                                   datetime(2015, 6, 21)   # June 21, 2015 at 00:00
                                   ]}

# locations['Thessaloniki']
# locations['Agios Nikolaos']

# Test currents
gulfDepartures = [datetime(2014, 10, 28),
                  datetime(2014, 11, 11),
                  datetime(2014, 11, 25),
                  datetime(2014, 4, 20),
                  datetime(2015, 5, 4),
                  datetime(2015, 5, 18)
                  ]

inputGulf = {'instance': 'Gulf',
             'input': {'from': [],
                       'to': [],
                       'departureDates': []}
             }
for date in gulfDepartures:
    for i, west in enumerate(locations['westLocations']):
        for j, east in enumerate(locations['eastLocations']):
            inputGulf['input']['from'].append(('{}'.format(i+1), west))
            inputGulf['input']['to'].append(('{}'.format(j+1), east))
            inputGulf['input']['departureDates'].append(date)


inputKC = {'instance': 'KC',
           'input': {'from': [('K', locations['KeelungC']),
                              ('T', locations['Tokyo'])],
                     'to': [('T', locations['Tokyo']),
                            ('K', locations['KeelungC'])],
                     'departureDates': [datetime(2014, 9, 15),
                                        datetime(2015, 3, 15)]}
           }

inputSalLim = {'instance': 'SalLim',
               'input': {'from': [('S', locations['Salvador'])],
                         'to': [('L', locations['Lima'])],
                         'departureDates': [datetime(2014, 11, 11)]}
               }

inputECA = {'instance': 'ECA',
            'input': {'from': [('ECA1', locations['ECA1: Jacksonville'])],
                      'to': [('ECA2', locations['ECA2: New York'])],
                      'departureDates': [None]}
            }

multiple_experiments(inputKC, exp='current')  # TEST FOR [GC, CONSTANT] [CURRENT, CONSTANT] [CURRENT, VAR]
# multiple_experiments(inputKC, 'current')
# multiple_experiments(inputGulf, 'current')
# multiple_experiments(inputWeather, 'weather')
# multiple_experiments(inputECA, 'ecas')
