import main
import matplotlib.pyplot as plt
import numpy as np
import skopt
# import time

from datetime import datetime
from pathlib import Path
from skopt import plots
from skopt.callbacks import CheckpointSaver
from support import locations

DIR = Path('D:/')
N_CALLs = 100
N_POINTS = 10
DEPART = datetime(2016, 1, 1)
CURRENT = False
BATHYMETRY = False

START_END = (locations['Caribbean Sea'], locations['North UK'])
CALLS = 0


inputParameters = {'n': 120,
                   'gen': 100,
                   'segLengthF': 100,
                   'mutationOperators': ['speed', 'insert', 'move', 'delete'],
                   'mutpb': 0.7}
PLANNER = main.RoutePlanner(bathymetry=BATHYMETRY, inputParameters=inputParameters)

SPACE = [
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
    # skopt.space.Real(0.001, 10., name='widthRatio'),
    # skopt.space.Real(0.001, 10., name='radius'),
    # skopt.space.Real(0.1, 5.0, name='delFactor')
    ]


def tune(default_parameters=None):
    if default_parameters is None:
        default_parameters = {}

    def train_evaluate(search_params):
        global CALLS
        # start_func_time = time.time()

        parameters = {**default_parameters, **search_params}
        print(parameters)
        CALLS += 1
        print('Call', CALLS)

        PLANNER.update_parameters(parameters)
        result = PLANNER.compute(START_END, startDate=DEPART, current=CURRENT, recompute=True, seed=None)

        # end_func_time = time.time() - start_func_time
        avgFitList = [subLog.chapters["fitness"].select("avg") for log in result['logs'] for subLog in log]
        avgFit = np.sum(np.sum(avgFitList, axis=0), axis=0)
        # score = np.append(  # end_func_time, avgFit )
        weights = np.array([10, 1])
        weightedSum = np.dot(avgFit, weights)

        return weightedSum

    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        return train_evaluate(params)

    checkpoint_saver = CheckpointSaver("./checkpoint.pkl", compress=9, store_objective=False)
    res = skopt.forest_minimize(objective, SPACE, n_calls=N_CALLs, n_random_starts=N_POINTS,
                                callback=[checkpoint_saver])

    # Save results
    timestamp = datetime.now().strftime('%m%d%H%M')
    fp = DIR / 'output/tuning/{}_{}calls_{}points_tuned_parameters_result.gz'.format(timestamp, N_CALLs, N_POINTS)
    skopt.dump(res, fp, compress=9, store_objective=False)
    print('Saved to', fp)

    return fp, timestamp


def show_results(fp, timestamp):
    res = skopt.load(fp)

    # Print results
    best_params = {par.name: res.x[i] for i, par in enumerate(SPACE)}
    print('best result: ', res.fun)
    print('best parameters: ', best_params)

    saveFP = DIR / '/output/tuning/figures/{}_'.format(timestamp)
    # Plot results
    plots.plot_evaluations(res)
    plt.savefig(saveFP + saveFP.as_posix() + 'eval.pdf')
    plots.plot_objective(res)
    plt.savefig(saveFP + saveFP.as_posix() + 'obj.pdf')
    plots.plot_convergence(res)
    plt.savefig(saveFP + saveFP.as_posix() + 'conv.pdf')


FP, STAMP = tune()

timestamp = datetime.now().strftime('%m%d%H%M')
# fp = 'D:/JobS/Downloads/checkpoint_3.pkl'
show_results(FP, timestamp)

# plt.show()
