import matplotlib.pyplot as plt
import numpy as np
import pickle
import skopt
import time

from datetime import datetime
from main import RoutePlanner
from skopt import plots, forest_minimize

for gauss in [True, False]:
    PARS = {'gauss': gauss}
    CALL = 0
    planner = RoutePlanner()


    def train_evaluate(search_params):
        global CALL
        start_func_time = time.time()

        start_ends = [((20.89, 58.46), (-85.06, 29.18))]  # Gulf of Bothnia, Gulf of Mexico
        parameters = {**PARS, **search_params}
        print(parameters)

        planner.update_parameters(parameters)
        result = planner.compute(start_ends, seed=None)[0]

        end_func_time = time.time() - start_func_time
        path_fits = []
        for path in result['fronts'].values():
            path_fitness = np.empty(2)
            for sub_path in path.values():
                path_fitness += sub_path[0].fitness.values
            path_fits.append(path_fitness)
        avg_fit = np.mean(path_fits, axis=0)
        score_arr = np.append(avg_fit, end_func_time)
        weights = np.array([200, 100, 1])
        score = np.dot(score_arr, weights)

        CALL += 1
        print('call:', CALL)

        return score


    SPACE = [
        # skopt.space.Integer(10, 500, name='gen'),
        # skopt.space.Categorical([4 * i for i in range(1, 151)], name='n'),
        # skopt.space.Real(0.5, 1.0, name='cxpb'),
        # skopt.space.Real(0.5, 1.0, name='mutpb'),
        # skopt.space.Integer(5, 500, name='nBar', prior='log-uniform'),


        # skopt.space.Integer(1, 10, name='recomb'),
        # skopt.space.Integer(1, 20, name='fails'),
        # skopt.space.Integer(1, 20, name='moves'),
        skopt.space.Real(0.1, 10.0, name='widthRatio'),
        skopt.space.Real(0.1, 10.0, name='radius', prior='log-uniform'),
        # skopt.space.Real(0.1, 5.0, name='delFactor')
    ]


    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        return train_evaluate(params)


    n_calls = 250
    results = forest_minimize(objective, SPACE, n_calls=n_calls, n_random_starts=10)

    timestamp = datetime.now()
    timestamp = '{0:%H_%M_%S}'.format(timestamp)

    fp = 'output/tuning/{}_{}_calls_tuned_parameters_result'.format(timestamp, n_calls)
    with open(fp, 'wb') as fh:
        pickle.dump(results, fh)
        print('Saved to: ', fp)

    best_params = {k.name: results.x[i] for i, k in enumerate(SPACE)}

    print('best result: ', results.fun)
    print('best parameters: ', best_params)

    ev_ax = plots.plot_evaluations(results)
    plt.savefig('output/tuning/{}_{}_calls_tune_eval.png'.format(timestamp, n_calls))
    obj_ax = plots.plot_objective(results)
    plt.savefig('output/tuning/{}_{}_calls_tune_obj.png'.format(timestamp, n_calls))

plt.show()