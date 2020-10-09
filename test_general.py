import case_studies.plot_results as plot_results
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import time

from datetime import datetime


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


def single_experiment(planner, inputDict, parameters, startEnd, depDate, fileString, genDir, seed, saveFig=True):
    rawListFP = genDir / 'raw/{}'.format(fileString)
    if os.path.exists(rawListFP):
        return None

    exp = parameters['exp']

    current = True if (exp == 'current' or exp == 'MOEA') and parameters['current'] else False
    weather = True if exp == 'weather' else False
    ecas = True if parameters['ecaFactor'] != 1.0 or exp == 'eca' else False

    rawList, procList = [], []
    for itr in range(parameters['iterations']):
        print('ITERATION {} of {}'.format(itr + 1, parameters['iterations']))
        t0 = time.time()
        seed = itr if seed else None
        raw = planner.compute(startEnd, startDate=depDate, recompute=True, weather=weather, current=current,
                              algorithm=parameters['MOEA'], seed=seed)
        t1 = time.time() - t0

        updateDict = {exp: 1.5593} if exp == 'eca' else {exp: depDate}
        proc, raw = planner.post_process(raw, updateEvaluator=updateDict)
        proc['computationTime'] = t1
        rawList.append(raw)
        procList.append(proc)

        if saveFig:

            statisticsPlotter = plot_results.StatisticsPlotter(raw, DIR=parameters['DIR'])
            frontFig, _ = statisticsPlotter.plot_fronts()
            statsFig, _ = statisticsPlotter.plot_stats()

            if current:
                cData = planner.evaluator.currentOp.data
                if inputDict['instance'] == 'KC':
                    lons0 = planner.evaluator.currentOp.lo
                    lats0 = planner.evaluator.currentOp.la
                    currentDict = {'u': cData[0], 'v': cData[1], 'lons': lons0, 'lats': lats0}
                else:
                    lons0 = np.linspace(-179.875, 179.875, 1440)
                    lats0 = np.linspace(-89.875, 89.875, 720)
                    currentDict = {'u': cData[0, 0], 'v': cData[1, 0], 'lons': lons0, 'lats': lats0}
            else:
                currentDict = None

            weatherDate = depDate if exp == 'weather' else None
            routePlotter = plot_results.RoutePlotter(parameters['DIR'], proc, rawResults=raw, vessel=planner.vessel)
            routeFig, routeAx = plt.subplots()
            routePlotter.results(routeAx, initial=False, ecas=ecas, bathymetry=parameters['bathymetry'], nRoutes=4,
                                 weatherDate=weatherDate, current=currentDict, colorbar=True, KC=inputDict['instance'] == 'KC')

            for figName, fig in {'front': frontFig, 'stats': statsFig, 'routes': routeFig}.items():
                fp = genDir / 'figures/{}_{}'.format(fileString, itr)
                fig.savefig(fp.as_posix() + '_{}.png'.format(figName), dpi=300)
                print('saved', fp)

            plt.close('all')

    with open(rawListFP, 'wb') as f:
        pickle.dump(rawList, f)
        print('saved', rawListFP)

    return get_df(procList)


def multiple_experiments(inputDict, planner, parameters, genDir, seed=None):
    depDates = inputDict['input']['departureDates']
    timestamp = datetime.now().strftime('%m%d-%H%M')

    origins, destinations = [], []
    for d in range(len(depDates)):
        if parameters['exp'] == 'weather':
            origins.append([inputDict['input']['from'][d]])
            destinations.append([inputDict['input']['to'][d]])
        else:
            origins.append(inputDict['input']['from'])
            destinations.append(inputDict['input']['to'])

    for d, depDate in enumerate(depDates):
        # if d > 0 and parameters['ref'] == 'R_':
        #     continue
        print('date {} of {}'.format(d+1, len(depDates)))
        depS = '' if depDate is None else depDate.strftime('%Y_%m_%d')
        fileString = '{}{}'.format(parameters['ref'], depS)
        writer = pd.ExcelWriter(genDir / 'tables' / '{}_{}.xlsx'.format(timestamp, fileString))
        summary = pd.DataFrame(index=['compTime', 'T_fuel', 'T_time', 'C_fuel', 'C_time', 'L_fuel', 'L_time'])

        for routeIdx, (startTup, endTup) in enumerate(zip(origins[d], destinations[d])):
            startKey, start = startTup
            endKey, end = endTup
            print('location combination {} of {}'.format(routeIdx + 1, len(origins[d])))
            fileString2 = fileString + '_{}{}'.format(startKey, endKey)
            fp = genDir / 'tables/csv/{}.csv'.format(fileString)
            df = single_experiment(planner, inputDict, parameters, (start, end), depDate, fileString2, genDir, seed,
                                   saveFig=True)
            if df is None:
                df = pd.read_csv(fp)
                print('read', fp)
            else:
                df.to_csv(fp)
                print('saved', fp)
            df.to_excel(writer, sheet_name=fileString)
            summary[fileString + '_mean'] = df['mean']
            summary[fileString + '_std'] = df['std']

        summary.to_excel(writer, sheet_name='summary')
        writer.save()
        print('saved', writer.path)
    print('DONE TESTING')

