import flask
import uuid

from flask import request, jsonify
from flask_cors import CORS

app = flask.Flask(__name__)
CORS(app)

# list of input configurations for requested routes
requested_routes = []


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
			

			
			# for now the input configuration is returned, but ultimately the route calculation result should be return instead
			return jsonify(r)
	
	# if there is no requested route with the correct id, return an error
	return jsonify({'error': 'No requested route found for id: ' + id})


app.run()
