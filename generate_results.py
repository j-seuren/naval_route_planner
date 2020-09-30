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

# INPUT PARAMETERS
ECA_F = 1
DEPTH = False
# SPEED = 'constant'  # 'constant' or 'var'
ITERS = 1
# CURRENT = True
criteria = {'minimalTime': True, 'minimalCost': True}
# -------------------------------------------------


#  Other parameters
T = datetime.now().strftime('%m%d-%H%M')
DIR = Path('D:/')


# Test weather
# Weather locations
inputWeather = {'from': [('Ny', locations['New York']),  # Kuhlemann
                         ('K', locations['Keelung']),  # Lin2013
                         ('No', locations['Normandy']),  # Shao2012
                         ('No', locations['Normandy']),  # Vettor2016
                         ('V', locations['Valencia'])  # Vettor2016
                         ],
                'to': [('P', locations['Paramaribo']), ('S', locations['San Francisco']), ('Ny', locations['New York']),
                       ('Mi', locations['Miami']), ('Ma', locations['Malta'])],
                'departureDates': [datetime(2017, 9, 4), datetime(2011, 5, 28),
                                   # DEP 0000Z 28 May 2011, ETA 0000Z 11 June 2011
                                   datetime(2011, 1, 25),  # DEP 03:00 p.m. ETA: 00:30 p.m. 30/01/2011
                                   datetime(2015, 6, 21),  # June 21, 2015 at 00:00
                                   datetime(2015, 6, 21)  # June 21, 2015 at 00:00
                                   ]}

# locations['Thessaloniki']
# locations['Agios Nikolaos']

# Test currents
gulfDepartures = [datetime(2014, 10, 28), datetime(2014, 11, 11), datetime(2014, 11, 25), datetime(2014, 4, 20),
                  datetime(2015, 5, 4), datetime(2015, 5, 18)]

inputGulf = {'instance': 'Gulf', 'input': {'from': [], 'to': [], 'departureDates': []}}
for date in gulfDepartures:
    for i, west in enumerate(locations['westLocations']):
        for j, east in enumerate(locations['eastLocations']):
            inputGulf['input']['from'].append(('{}'.format(i + 1), west))
            inputGulf['input']['to'].append(('{}'.format(j + 1), east))
            inputGulf['input']['departureDates'].append(date)

inputKC = {'instance': 'KC', 'input': {'from': [('K', locations['KeelungC']), ('T', locations['Tokyo'])],
                                       'to': [('T', locations['Tokyo']), ('K', locations['KeelungC'])],
                                       'departureDates': [datetime(2014, 9, 15), datetime(2015, 3, 15)]}}

inputSalLim = {'instance': 'SalLim', 'input': {'from': [('S', locations['Salvador'])], 'to': [('L', locations['Lima'])],
                                               'departureDates': [datetime(2014, 11, 11)]}}

inputECA = {'instance': 'ECA', 'input': {'from': [('ECA1', locations['ECA1: Jacksonville'])],
                                         'to': [('ECA2', locations['ECA2: New York'])], 'departureDates': [None]}}


def get_df(procList):
    LN, FC, TT = {}, {}, {}
    for idx, obj in enumerate(['time', 'fuel']):
        LN[obj] = [proc['routeResponse'][idx]['distance'] for proc in procList]
        FC[obj] = [proc['routeResponse'][idx]['fuelCost'] for proc in procList]
        TT[obj] = [proc['routeResponse'][idx]['travelTime'] for proc in procList]

    compTime = [proc['computationTime'] for proc in procList]
    table = list(zip(compTime, TT['fuel'], TT['time'], FC['fuel'], FC['time'], LN['fuel'], LN['time']))
    df = pd.DataFrame(table, columns=['compTime', 'T_fuel', 'T_time', 'C_fuel', 'C_time', 'L_fuel', 'L_time']).T
    mean, std = df.mean(axis=1), df.std(axis=1)
    df['mean'], df['std'] = mean, std

    return df


for SPEED in ['var', 'constant']:
    for CURRENT in [True, False]:
        speedOps = ['insert', 'move', 'delete'] if SPEED == 'constant' else ['insert', 'move', 'speed', 'delete']
        par = {'mutationOperators': speedOps}
        PLANNER = main.RoutePlanner(inputParameters=par, bathymetry=DEPTH, ecaFactor=ECA_F, criteria=criteria)

        BL = 'BLANK_' if not CURRENT else ''


        def single_experiment(exp, startEnd, depDate, fileString, algorithm, generalFP, saveFig=True):
            rawListFP = generalFP / 'raw/{}'.format(fileString)
            if os.path.exists(rawListFP):
                return None

            current = True if exp == 'current' and CURRENT else False
            weather = True if exp == 'weather' else False
            ecas = True if ECA_F != 1 else False

            rawList, procList = [], []
            for itr in range(ITERS):
                print('ITERATION {} of {}'.format(itr+1, ITERS))
                t0 = time.time()
                raw = PLANNER.compute(startEnd, startDate=depDate, recompute=True, weather=weather, current=current,
                                      algorithm=algorithm)
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
                    routeFig, routeAx = plt.subplots()
                    routeAx = routePlotter.results(routeAx, initial=False, ecas=ecas, bathymetry=DEPTH, nRoutes=4,
                                                   weatherDate=weatherDate, current=currentDict, colorbar=True)

                    for fig in [frontFig, statsFig, routeFig]:
                        fp = generalFP / 'figures/{}_{}.png'.format(fileString, itr)
                        fig.savefig(fp, dpi=300)
                        print('saved', fp)

                    plt.close('all')

            with open(rawListFP, 'wb') as f:
                pickle.dump(rawList, f)
                print('saved', rawListFP)

            return get_df(procList)


        def init_experiment(writer, exp, dfSummary, depDate, locS, fileString, start, end, algorithm, generalFP):
            fp = generalFP / 'tables/csv/{}.csv'.format(fileString)
            df = single_experiment(exp, (start, end), depDate, fileString, algorithm, generalFP, saveFig=True)
            if df is None:
                df = pd.read_csv(fp)
                print('read', fp)
            else:
                df.to_csv(fp)
                print('saved', fp)
            df.to_excel(writer, sheet_name=locS)
            dfSummary[locS + '_mean'] = df['mean']
            dfSummary[locS + '_std'] = df['std']
            return dfSummary.T


        def multiple_experiments(inputDict, exp, algorithm='NSGA2'):
            depDates = inputDict['input']['departureDates']

            # Create directories
            generalFP = DIR / 'output' / exp / inputDict['instance'] / '{}_{}SP_B{}_ECA{}'.format(algorithm,
                                                                                                  SPEED,
                                                                                                  DEPTH,
                                                                                                  ECA_F)
            testDirs = [generalFP / 'tables/csv', generalFP / 'figures', generalFP / 'raw']
            for directory in testDirs:
                if not os.path.exists(directory):
                    os.makedirs(directory)

            for d, depDate in enumerate(depDates):
                print('date {} of {}'.format(d+1, len(depDates)))
                depS = '' if depDate is None else 'depart' + depDate.strftime('%Y_%m_%d')
                writer = pd.ExcelWriter(generalFP / 'tables' / '{}departure_{}.xlsx'.format(BL, depS))
                dfSummary = pd.DataFrame(columns=['compTime', 'T_fuel', 'T_time', 'C_fuel', 'C_time', 'L_fuel', 'L_time'])

                for routeIdx, (startTup, endTup) in enumerate(zip(inputDict['input']['from'], inputDict['input']['to'])):
                    startKey, start = startTup
                    endKey, end = endTup
                    print('location combination {} of {}'.format(routeIdx + 1, len(inputDict['input']['from'])))
                    locS = '{}{}'.format(startKey, endKey)
                    fileString = '{}_{}location{}_departure{}'.format(T, BL, locS, depS)
                    dfSummary = init_experiment(writer, exp, dfSummary, depDate, locS, fileString, start, end, algorithm,
                                                generalFP)

                dfSummary.to_excel(writer, sheet_name='summary')
                writer.save()
                print('saved', writer)
            print('DONE TESTING')

        multiple_experiments(inputKC, exp='current')

# if __name__ == '__main__':
#     # for alg in ['NSGA2', 'SPEA2', 'MPAES']:
#     #     multiple_experiments(inputKC, exp='current', algorithm=alg)
#     #     # TEST FOR [GC, CONSTANT] [CURRENT, CONSTANT] [CURRENT, VAR]
#     multiple_experiments(inputKC, exp='current')
#
#     # multiple_experiments(inputKC, 'current')
#     # multiple_experiments(inputGulf, 'current')
#     # multiple_experiments(inputWeather, 'weather')
#     # multiple_experiments(inputECA, 'ecas')
