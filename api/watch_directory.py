import json
import os
import main
import numpy as np

from datetime import datetime


def get_best_inds(result_dict, json_file):
    best_inds = {}
    for path_key, path in result_dict['fronts'].items():
        best_inds[path_key] = {'min_fuel': [], 'min_time': []}
        for sp_key, front in path.items():
            # Get min time and min fuel
            fit_values = np.asarray([ind.fitness.values for ind in front])

            time_ind = front[np.argmin(fit_values[:, 0])]
            fuel_ind = front[np.argmin(fit_values[:, 1])]

            # Append best individuals to list
            best_inds[path_key]['min_fuel'].append(fuel_ind)
            best_inds[path_key]['min_time'].append(time_ind)

    output = {'input': json_file,
              'canal_paths': []
              }
    for path_key, path in best_inds.items():
        obj_dicts = {}
        for obj_key, sub_inds in path.items():
            fit = np.zeros(2)
            wps = []
            speeds = []
            for sub_ind in sub_inds:
                sub_fit = np.array(sub_ind.fitness.values)
                fit += sub_fit
                wps.extend([el[0] for el in sub_ind])
                speeds.extend([el[1] for el in sub_ind])
            obj_dict = {'fitness_optimized': {'fuel_cost': fit[1],
                                              'time_days': fit[0]},
                        'fitness_real': None,
                        'waypoints': {i: {'lng': pt[0],
                                          'lat': pt[1],
                                          'speed': speeds[i]}
                                      for i, pt in enumerate(wps)}
                        }
            del obj_dict['waypoints'][len(wps)-1]['speed']
            obj_dicts[obj_key] = obj_dict

        output['canal_paths'].append(obj_dicts)
    return output


path_to_load = os.path.abspath('api/start_end_IN')
path_to_save = os.path.abspath('api/routes_OUT')
file_name = 'test.json'

with open(os.path.join(path_to_load, file_name)) as f:
    json_in = json.load(f)

start = (json_in['from']['lng'], json_in['from']['lat'])
end = (json_in['to']['lng'], json_in['to']['lat'])
fuel_price = json_in['fuel_price'] / 1000.0
date_time_str = json_in['start_date'] + ' ' + json_in['start_time']
start_date = datetime.strptime(date_time_str,
                               '%Y-%m-%d %H:%M:%S')

route_planner = main.RoutePlanner(start,
                                  end,
                                  vessel_name=json_in['ship'],
                                  eca_f=json_in['seca_multiplier'],
                                  incl_curr=False,
                                  start_date=start_date)
result = route_planner.nsgaii.compute(seed=1)

dict_out = get_best_inds(result, json_in)

# Save result
with open(os.path.join(path_to_save, file_name), 'w') as f:
    json.dump(dict_out, f, indent=4)

print("Saved: {}".format(file_name))
