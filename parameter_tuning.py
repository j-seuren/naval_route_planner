import matplotlib.pyplot as plt
import numpy as np
import skopt
# import time

from datetime import datetime
from main import RoutePlanner
from skopt import plots, dump, load
from skopt.callbacks import CheckpointSaver
from support import locations

DIR = 'output/tuning/'
N_CALLs = 200
N_POINTS = 10

START_END = (locations['Tokyo'], locations['San Francisco'])
CALLS = 0
PLANNER = RoutePlanner()
SPACE = [
    # skopt.space.Integer(10, 500, name='gen'),
    # skopt.space.Categorical([4 * i for i in range(1, 151)], name='n'),
    # skopt.space.Real(0.5, 1.0, name='cxpb'),
    skopt.space.Real(0.5, 1.0, name='mutpb'),
    # skopt.space.Integer(5, 500, name='nBar', prior='log-uniform'),
    # skopt.space.Integer(1, 10, name='recomb'),
    # skopt.space.Integer(1, 20, name='fails'),
    # skopt.space.Integer(1, 20, name='moves'),
    skopt.space.Real(0.0001, 10.0, name='widthRatio', prior='log-uniform'),
    skopt.space.Real(0.0001, 10.0, name='radius', prior='log-uniform'),
    # skopt.space.Real(0.1, 5.0, name='delFactor')
    ]


def tune(default_parameters):
    def train_evaluate(search_params):
        global CALLS
        # start_func_time = time.time()

        parameters = {**default_parameters, **search_params}
        print(parameters)

        PLANNER.update_parameters(parameters)
        result = PLANNER.compute(START_END, recompute=True, seed=None)

        # end_func_time = time.time() - start_func_time
        avgFitList = [subLog.chapters["fitness"].select("avg") for log in result['logs'] for subLog in log]
        avgFit = np.sum(np.sum(avgFitList, axis=0), axis=0)
        # score = np.append(  # end_func_time,
        #                   avgFit
        #                   )
        weights = np.array([10, 1])
        weightedSum = np.dot(avgFit, weights)
        CALLS += 1
        print('Call', CALLS)

        return weightedSum

    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        return train_evaluate(params)

    checkpoint_saver = CheckpointSaver("./checkpoint.pkl", compress=9, store_objective=False)
    res = skopt.forest_minimize(objective, SPACE, n_calls=N_CALLs, n_random_starts=N_POINTS,
                                callback=[checkpoint_saver])

    # Save results
    timestamp = datetime.now().strftime('%m%d%H%M')
    fn = '{}_{}calls_{}points_tuned_parameters_result.gz'.format(timestamp, N_CALLs, N_POINTS)
    fp = DIR + fn
    dump(res, fp, compress=9, store_objective=False)
    print('Saved to', fp)

    return fp, timestamp


def show_results(fp, timestamp):
    res = load(fp)

    # Print results
    best_params = {par.name: res.x[i] for i, par in enumerate(SPACE)}
    print('best result: ', res.fun)
    print('best parameters: ', best_params)

    # Plot results
    plots.plot_evaluations(res)
    fp_ev = DIR + '{}_eval.pdf'.format(timestamp)
    plt.savefig(fp_ev)
    plots.plot_objective(res)
    fp_obj = 'output/tuning/{}_obj.pdf'.format(timestamp)
    plt.savefig(fp_obj)
    plots.plot_convergence(res)


filesInfo = []
for gauss in [True, False]:
    _default_parameters = {'gauss': gauss, 'gen': 500, 'mutationOperators': ['insert', 'move', 'delete']}
    FP, STAMP = tune(_default_parameters)
    filesInfo.append((FP, STAMP))

for fileInfo in filesInfo:
    show_results(fileInfo[0], fileInfo[1])

plt.show()
