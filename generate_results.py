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
from support import inputKC_2, inputKC, inputWeather, inputECA, inputGulf, inputSalLim, locations


def get_df(procList):
    LN, FC, TT = {}, {}, {}
    for i, obj in enumerate(['time', 'fuel']):
        LN[obj] = [proc['routeResponse'][i]['distance'] for proc in procList]
        FC[obj] = [proc['routeResponse'][i]['fuelCost'] for proc in procList]
        TT[obj] = [proc['routeResponse'][i]['travelTime'] for proc in procList]

    compTime = [proc['computationTime'] for proc in procList]
    table = list(zip(compTime, TT['fuel'], TT['time'], FC['fuel'], FC['time'], LN['fuel'], LN['time']))
    df = pd.DataFrame(table, columns=['compTime', 'T_fuel', 'T_time', 'C_fuel', 'C_time', 'L_fuel', 'L_time']).T
    mean, std = df.mean(axis=1), df.std(axis=1)
    df['mean'], df['std'] = mean, std

    return df


for speedIdx in range(12):
    for CURRENT in [True, False]:
        # INPUT PARAMETERS
        # ECA_F = 1.5593
        ECA_F = 1.0
        DEPTH = False
        SPEED = 'constant'  # 'constant' or 'var'
        ITERS = 5
        # CURRENT = False
        criteria = {'minimalTime': True, 'minimalCost': True}
        # -------------------------------------------------

        #  Other parameters
        timestamp = datetime.now().strftime('%m%d-%H%M')
        DIR = Path('D:/')
        speedOps = ['insert', 'move', 'delete'] if SPEED == 'constant' else ['speed', 'insert', 'move', 'delete']
        par = {'mutationOperators': speedOps}
        PLANNER = main.RoutePlanner(inputParameters=par, bathymetry=DEPTH, ecaFactor=ECA_F, criteria=criteria)
        R = 'R_' if not CURRENT else ''


        def single_experiment(exp, startEnd, depDate, fileString, algorithm, generalFP, saveFig=True):
            rawListFP = generalFP / 'raw/{}'.format(fileString)
            if os.path.exists(rawListFP):
                return None

            current = True if exp == 'current' and CURRENT else False
            weather = True if exp == 'weather' else False
            ecas = True if ECA_F != 1 or exp == 'eca' else False

            rawList, procList = [], []
            for itr in range(ITERS):
                print('ITERATION {} of {}'.format(itr+1, ITERS))
                t0 = time.time()
                raw = PLANNER.compute(startEnd, startDate=depDate, recompute=True, weather=weather, current=current,
                                      algorithm=algorithm)
                t1 = time.time() - t0
                proc, raw = PLANNER.post_process(raw  # , inclEnvironment={exp: depDate}
                                                 )
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
                    routePlotter.results(routeAx, initial=False, ecas=ecas, bathymetry=DEPTH, nRoutes=4,
                                         weatherDate=weatherDate, current=currentDict, colorbar=True)

                    for figName, fig in {'front': frontFig, 'stats': statsFig, 'routes': routeFig}.items():
                        fp = generalFP / 'figures/{}_{}'.format(fileString, itr)
                        fig.savefig(fp.as_posix() + '_{}.png'.format(figName), dpi=300)
                        print('saved', fp)

                    plt.close('all')

            with open(rawListFP, 'wb') as f:
                pickle.dump(rawList, f)
                print('saved', rawListFP)

            return get_df(procList)


        def init_experiment(writer, exp, summary, depDate, fileString, start, end, algorithm, generalFP):
            fp = generalFP / 'tables/csv/{}.csv'.format(fileString)
            df = single_experiment(exp, (start, end), depDate, fileString, algorithm, generalFP, saveFig=True)
            if df is None:
                df = pd.read_csv(fp)
                print('read', fp)
            else:
                df.to_csv(fp)
                print('saved', fp)
            df.to_excel(writer, sheet_name=fileString)
            summary[fileString + '_mean'] = df['mean']
            summary[fileString + '_std'] = df['std']
            return summary


        def multiple_experiments(_input, exp, algorithm='NSGA2'):
            depDates = _input['input']['departureDates']

            # Create directories
            genDir = DIR / 'output' / exp / _input['instance'] / '{}_{}SP_B{}_ECA{}/{}'.format(algorithm,
                                                                                               speedIdx,
                                                                                               DEPTH,
                                                                                               ECA_F,
                                                                                               ITERS)

            createDirs = [genDir / 'tables/csv', genDir / 'figures', genDir / 'raw']
            [os.makedirs(directory) for directory in createDirs if not os.path.exists(directory)]

            for d, depDate in enumerate(depDates):
                print('date {} of {}'.format(d+1, len(depDates)))
                depS = '' if depDate is None else depDate.strftime('%Y_%m_%d')
                fileString = '{}{}'.format(R, depS)
                writer = pd.ExcelWriter(genDir / 'tables' / '{}_{}.xlsx'.format(timestamp, fileString))
                summary = pd.DataFrame(index=['compTime', 'T_fuel', 'T_time', 'C_fuel', 'C_time', 'L_fuel', 'L_time'])

                for routeIdx, (startTup, endTup) in enumerate(zip(_input['input']['from'], _input['input']['to'])):
                    startKey, start = startTup
                    endKey, end = endTup
                    print('location combination {} of {}'.format(routeIdx + 1, len(_input['input']['from'])))
                    fileString2 = fileString + '_{}{}'.format(startKey, endKey)
                    summary = init_experiment(writer, exp, summary, depDate, fileString2, start, end, algorithm,
                                              genDir)

                summary.to_excel(writer, sheet_name='summary')
                writer.save()
                print('saved', writer.path)
            print('DONE TESTING')


        # for alg in ['NSGA2', 'SPEA2', 'MPAES']:
        #     inputKC['departureDates'] = inputKC['departureDates'][0]
        gulfDepartures = [datetime(2014, 10, 28), datetime(2014, 11, 11), datetime(2014, 11, 25), datetime(2014, 4, 20),
                          datetime(2015, 5, 4), datetime(2015, 5, 18)]

        inputGulf = {'instance': 'Gulf', 'input': {'from': [], 'to': [], 'departureDates': []}}
        west = locations['westLocations'][2]
        east = locations['eastLocations'][0]
        inputGulf['input']['from'].append(('{}'.format(2), west))
        inputGulf['input']['to'].append(('{}'.format(0), east))
        inputGulf['input']['departureDates'].append(datetime(2014, 11, 25))
        multiple_experiments(inputGulf, exp='current')

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
