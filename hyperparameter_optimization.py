import numpy as np
import pickle
import skopt
import time

from datetime import datetime
from main import RoutePlanner

INPUT = {'start': (78, 5),
         'end': (49, 12),
         'start_date': datetime(2016, 1, 1),
         'vessel_name': 'Fairmaster',
         'current': False,
         'eca_f': 1.05,
         'vlsfo_price': 0.3}


def train_evaluate(search_params):
    print(search_params)
    start_func_time = time.time()

    params = {'res': 'c',
              'spl_th': 4,
              'shape': 3,
              'scale_factor': 0.1,
              'recomb': 5,
              'l_fails': 5,
              'l_moves': 10,
              'n_bar': 50,
              **search_params}

    start = INPUT['start']
    end = INPUT['end']
    start_date = INPUT['start_date']
    vessel_name = INPUT['vessel_name']
    incl_curr = INPUT['current']
    eca_f = INPUT['eca_f']
    vlsfo_price = INPUT['vlsfo_price']

    route_planner = RoutePlanner(start, end, start_date, vessel_name,
                                 incl_curr, eca_f, vlsfo_price, params)

    result = route_planner.nsgaii.compute(seed=1)

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

    return score


SPACE = [skopt.space.Integer(10, 2000, name='gen', prior='uniform'),
         skopt.space.Integer(8, 1000, name='n', prior='log-uniform'),
         skopt.space.Real(0.5, 1.0, name='cxpb', prior='uniform'),
         skopt.space.Real(0.5, 1.0, name='mutpb', prior='uniform'),
         # skopt.space.Integer(5, 500, name='n_bar', prior='log-uniform'),
         # skopt.space.Integer(1, 10, name='recomb', prior='uniform'),
         # skopt.space.Integer(1, 20, name='l_fails', prior='uniform'),
         # skopt.space.Integer(1, 20, name='l_moves', prior='uniform'),
         skopt.space.Real(0.1, 5.0, name='width_ratio', prior='uniform'),
         skopt.space.Real(0.1, 5.0, name='radius', prior='uniform'),
         skopt.space.Real(0.1, 5.0, name='del_factor', prior='uniform')]


@skopt.utils.use_named_args(SPACE)
def objective(**params):
    return train_evaluate(params)


results = skopt.forest_minimize(objective, SPACE, n_calls=500, n_random_starts=10)

with open('tuned_parameters_result', 'wb') as fh:
    pickle.dump(results, fh)

best_params = {k.name: results.x[i] for i, k in enumerate(SPACE)}

print('best result: ', results.fun)
print('best parameters: ', best_params)

