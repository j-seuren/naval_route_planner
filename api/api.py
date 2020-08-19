import flask
import uuid
import main
import numpy as np

from datetime import datetime
from flask import request, jsonify
from flask_cors import CORS

app = flask.Flask(__name__)
CORS(app)

# list of input configurations for requested routes
requested_routes = []


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

	output = {'input': json_file, 'canal_paths': []}
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
			obj_dict = {'fitness_optimized': {'fuel_cost': fit[1], 'time_days': fit[0]}, 'fitness_real': None,
						'waypoints': {i: {'lng': pt[0], 'lat': pt[1], 'speed': speeds[i]} for i, pt in enumerate(wps)}}
			del obj_dict['waypoints'][len(wps) - 1]['speed']
			obj_dicts[obj_key] = obj_dict

		output['canal_paths'].append(obj_dicts)
	return output


# request a new route
@app.route('/request_route', methods=['POST'])
def request_route():
	input_config = request.get_json()  # get input configuration from the request
	id = str(uuid.uuid4())  # create random id
	input_config['id'] = id  # put id on the input configuration
	requested_routes.append(input_config)  # append input configuration to the list of requested routes
	return jsonify({'id': id})


# trigger route calculation
@app.route('/route', methods=['GET'])
def get_route():
	# retrieve id from the request. If there is no id property on the request, return an error
	if 'id' in request.args:
		id = request.args['id']
	else:
		return jsonify({'error': 'No id field provided. Please specify an id.'})

	# search the list of requested routes for an input configuration with the correct id
	for r in requested_routes:
		if r['id'] == id:
			requested_routes.remove(r)  # remove the input configuration r from the list

			# Convert json parameter to input parameters
			start = (r['from']['lng'], r['from']['lat'])
			end = (r['to']['lng'], r['to']['lat'])
			startEnd = [(start, end)]
			fuelPrice = r['fuel_price'] / 1000.0
			ecaFactor = r['seca_multiplier']
			vesselName = r['ship']
			startDate = None
			if r['currents']:
				date_time_str = r['start_date'] + ' ' + r['start_time']
				startDate = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
			planner = main.RoutePlanner(vesselName=vesselName, ecaFactor=ecaFactor, fuelPrice=fuelPrice)
			result = planner.compute(startEnds=startEnd, startDate=startDate)

			dictOut = get_best_inds(result, r)

			return jsonify(dictOut)

	# if there is no requested route with the correct id, return an error
	return jsonify({'error': 'No requested route found for id: ' + id})


app.run()
