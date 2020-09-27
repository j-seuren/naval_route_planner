import main
import matplotlib.pyplot as plt
import numpy as np
import skopt
# import time

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from skopt import plots, dump, load
from skopt.callbacks import CheckpointSaver
from support import locations

DIR = 'D:/'
N_CALLs = 200
N_POINTS = 10

START_END = (locations['Caribbean Sea'], locations['North UK'])
CALLS = 0
PLANNER = main.RoutePlanner()
SPACE = [
    skopt.space.Integer(50, 300, name='gen'),
    skopt.space.Integer(10, 50, name='maxGDs'),
    skopt.space.Integer(2, 20, name='nMutations'),
    skopt.space.Categorical([4 * i for i in range(1, 101)], name='n'),
    skopt.space.Real(0.5, 1.0, name='cxpb'),
    skopt.space.Real(0.2, 0.7, name='mutpb'),
    skopt.space.Real(1e-7, 1e-4, name='minVar'),
    # skopt.space.Integer(5, 500, name='nBar', prior='log-uniform'),
    # skopt.space.Integer(1, 10, name='recomb'),
    # skopt.space.Integer(1, 20, name='fails'),
    # skopt.space.Integer(1, 20, name='moves'),
    # skopt.space.Real(0.0001, 10.0, name='widthRatio', prior='log-uniform'),
    # skopt.space.Real(0.0001, 10.0, name='radius', prior='log-uniform'),
    # skopt.space.Real(0.1, 5.0, name='delFactor')
    ]


def tune(default_parameters):
    def train_evaluate(search_params):
        global CALLS
        # start_func_time = time.time()

        parameters = {**default_parameters, **search_params}
        print(parameters)
        CALLS += 1
        print('Call', CALLS)

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

        return weightedSum

    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        return train_evaluate(params)

    checkpoint_saver = CheckpointSaver("./checkpoint.pkl", compress=9, store_objective=False)
    res = skopt.forest_minimize(objective, SPACE, n_calls=N_CALLs, n_random_starts=N_POINTS,
                                callback=[checkpoint_saver])

    # Save results
    timestamp = datetime.now().strftime('%m%d%H%M')
    fn = 'output/tuning/{}_{}calls_{}points_tuned_parameters_result.gz'.format(timestamp, N_CALLs, N_POINTS)
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
    with PdfPages('/output/tuning/figures/{}.pdf'.format(timestamp)) as pdf:
        plots.plot_evaluations(res)
        pdf.savefig()
        plots.plot_objective(res)
        pdf.savefig()
        plots.plot_convergence(res)
        pdf.savefig()


filesInfo = []
_default_parameters = {'mutationOperators': ['insert', 'move', 'delete']}
FP, STAMP = tune(_default_parameters)
filesInfo.append((FP, STAMP))

for fileInfo in filesInfo:
    show_results(fileInfo[0], fileInfo[1])

plt.show()
