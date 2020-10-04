import main
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import skopt
import time

from datetime import datetime
from pathlib import Path
# from skopt import plots
from skopt.callbacks import CheckpointSaver
from support import locations

DIR = Path('D:/')

for _ in range(5):
    N_CALLs = 1
    N_POINTS = 1
    DEPART = datetime(2016, 1, 1)
    CURRENT = False
    BATHYMETRY = False
    inputParameters = {'gen': 130,
                       'segLengthF': 100,
                       'mutationOperators': ['speed', 'insert', 'move', 'delete'],
                       'mutpb': 0.7}
    space1 = [
        skopt.space.Integer(50, 100, name='gen'),  # Minimal nr. generations
        skopt.space.Integer(5, 45, name='maxGDs'),
        skopt.space.Real(1e-6, 1e-4, name='minVar'),  # Minimal variance generational distance
        skopt.space.Integer(2, 15, name='nMutations'),
        skopt.space.Categorical([4 * i for i in range(1, 101)], name='n'),
        skopt.space.Real(0.5, 1.0, name='cxpb'),
        skopt.space.Real(0.2, 0.7, name='mutpb'),
        # skopt.space.Integer(5, 500, name='nBar', prior='log-uniform'),
        # skopt.space.Integer(1, 10, name='recomb'),
        # skopt.space.Integer(1, 20, name='fails'),
        # skopt.space.Integer(1, 20, name='moves'),
        # skopt.space.Real(0.001, 10., name='widthRatio'),
        # skopt.space.Real(0.001, 10., name='radius'),
        # skopt.space.Real(0.1, 5.0, name='delFactor')
        ]

    space2 = [
        # skopt.space.Integer(50, 100, name='gen'),  # Minimal nr. generations
        # skopt.space.Integer(5, 45, name='maxGDs'),
        # skopt.space.Real(1e-6, 1e-4, name='minVar'),  # Minimal variance generational distance
        # skopt.space.Integer(2, 15, name='nMutations'),
        # skopt.space.Categorical([4 * i for i in range(1, 101)], name='n'),
        # skopt.space.Real(0.5, 1.0, name='cxpb'),
        # skopt.space.Real(0.2, 0.7, name='mutpb'),
        # skopt.space.Integer(5, 500, name='nBar', prior='log-uniform'),
        # skopt.space.Integer(1, 10, name='recomb'),
        # skopt.space.Integer(1, 20, name='fails'),
        # skopt.space.Integer(1, 20, name='moves'),
        skopt.space.Real(0.001, 10., name='widthRatio'),
        skopt.space.Real(0.001, 10., name='radius'),
        # skopt.space.Real(0.1, 5.0, name='delFactor')
        ]

    spaces = [space1, space2]

    START_ENDS = [(locations['Caribbean Sea'], locations['North UK']), (locations['Singapore'], locations['Wellington']),
                  (locations['Houston'], locations['Paramaribo']), (locations['Sao Paulo'], locations['Sri Lanka']),
                  (locations['Keelung'], locations['Perth'])]

    iterations = len(START_ENDS)

    for SPACE in spaces:
        # Directories and filepaths
        timestamp = datetime.now().strftime('%m%d%H%M')
        tuningDir = DIR / 'output/tuning' / '{}_i{}_calls_{}points'.format(timestamp, N_CALLs, N_POINTS)
        figDir = tuningDir / 'figures'
        os.makedirs(figDir)
        excelFP = tuningDir / '{}_best_parameters_{}iterations.xlsx'.format(timestamp, iterations)
        writer = pd.ExcelWriter(excelFP)
        DF = None
        PLANNER = main.RoutePlanner(bathymetry=BATHYMETRY, inputParameters=inputParameters)

        for iteration in range(iterations):
            START_END = START_ENDS[iteration]

            CALLS = 0


            def tune(default_parameters=None):
                if default_parameters is None:
                    default_parameters = {}

                def train_evaluate(search_params):
                    global CALLS
                    start_func_time = time.time()

                    parameters = {**default_parameters, **search_params}
                    print(parameters)
                    CALLS += 1
                    print('Call', CALLS)

                    PLANNER.update_parameters(parameters)
                    result = PLANNER.compute(START_END, startDate=DEPART, current=CURRENT, recompute=True, seed=None)

                    end_func_time = time.time() - start_func_time
                    avgFitList = [subLog.chapters["fitness"].select("avg") for log in result['logs'] for subLog in log]
                    avgFit = np.sum(np.sum(avgFitList, axis=0), axis=0)
                    score = np.append(end_func_time, avgFit)
                    weights = np.array([1/0.1, 1, 1/100])
                    weightedSum = np.dot(score, weights)

                    return weightedSum

                @skopt.utils.use_named_args(SPACE)
                def objective(**params):
                    return train_evaluate(params)

                checkpoint_saver = CheckpointSaver("./checkpoint.pkl", compress=9, store_objective=False)
                res = skopt.forest_minimize(objective, SPACE, n_calls=N_CALLs, n_random_starts=N_POINTS,
                                            callback=[checkpoint_saver])
                return res


            def process_results(res, oldDF):
                # Save best parameters
                best_params = {par.name: res.x[i] for i, par in enumerate(SPACE)}
                print('best parameters: ', best_params)

                newDF = pd.DataFrame(best_params, index=[iteration])
                df = oldDF.append(newDF) if iteration > 0 else newDF



                # Save tuning results
                resFP = tuningDir / '{}.gz'.format(iteration)
                skopt.dump(_res, resFP, compress=9, store_objective=False)
                print('Saved tuning results to', resFP)

                # # Plot results
                # figFP = figDir / '{}_'.format(timestamp)
                # plots.plot_evaluations(res)
                # plt.savefig(figFP.as_posix() + 'eval.pdf')
                # plots.plot_objective(res)
                # plt.savefig(figFP.as_posix() + 'obj.pdf')
                # plots.plot_convergence(res)
                # plt.savefig(figFP.as_posix() + 'conv.pdf')
                # print('Saved figures to', figDir)

                return df


            _res = tune()
            DF = process_results(_res, DF)
        DF.to_excel(writer)
        print('written df to', writer.path)
        writer.close()

# timestamp = datetime.now().strftime('%m%d%H%M')
# fp = 'D:/JobS/Downloads/checkpoint_3.pkl'

# plt.show()
