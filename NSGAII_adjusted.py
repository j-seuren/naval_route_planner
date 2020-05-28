# Importing required modules
import sys
import math
import random
import matplotlib.pyplot as plt
import pickle
import pyvisgraph as vg
import solution
import fiona
from copy import deepcopy
from operator import attrgetter
import operators
from classes import Vessel

random.seed(1)


# Generate as set of initial solutions
def initialization(start_point_f, end_point_f, vessel_f, pop_size_f, shorelines_f, max_edge_length_f):
    print('Loading files')
    # Load the visibility graph file
    graph = vg.VisGraph()
    graph.load('output\GSHHS_c_L1.graph')

    print('Calculating visibility path')
    # Calculate visibility path
    path_list = graph.shortest_path(start_point_f, end_point_f)

    # Create Route class
    waypoints = []
    for waypoint in path_list:
        wp = solution.Waypoint(waypoint.x, waypoint.y)
        waypoints.append(wp)

    edges = []
    v = waypoints[0]
    for w in waypoints[1:]:
        edge = solution.Edge(v, w, vessel_f.speeds[0])
        v = w
        edges.append(edge)

    visibility_route = solution.Route(edges)

    with open('output/visibility_route', 'wb') as file:
        pickle.dump(visibility_route, file)

    # Insert code to make visibility_route feasible.
    # With respect to waypoint locations, edge intersections and vessel speed

    initial_routes = []
    for i in range(pop_size_f):
        sys.stdout.write('\rGenerating initial routes {0}/{1}'.format(i+1, pop_size_f))
        sys.stdout.flush()
        init_route = deepcopy(visibility_route)
        long_edge_exists = True
        count = 0
        while count < 10 or long_edge_exists:
            count += 1
            longest_edge = max(init_route.edges, key=attrgetter('distance'))
            if longest_edge.distance < max_edge_length_f:
                long_edge_exists = False
            init_route.insert_waypoint(bisector_length_ratio=0.5, polygons=shorelines_f, edge=longest_edge)
        initial_routes.append(init_route)

    print('\n')
    return initial_routes


# Sort list of indices by their corresponding values
def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if values.index(min(values)) in list1:
            sorted_list.append(values.index(min(values)))
        values[values.index(min(values))] = math.inf
    return sorted_list


# Function to carry out NSGA-II's fast non dominated sort
def non_dominated_sort(v1, v2):
    S = [[]] * len(v1)
    F = [[]]
    n = [0] * len(v1)
    rank = [0] * len(v1)
    for p in range(len(v1)):
        S[p] = []
        n[p] = 0
        for q in range(len(v1)):
            # If p dominates q
            if (v1[p] < v1[q] and v2[p] < v2[q]) or (v1[p] <= v1[q] and v2[p] < v2[q]) or (v1[p] < v1[q] and v2[p] <= v2[q]):
                if q not in S[p]:
                    S[p].append(q)  # Add q to the set of solutions dominated by p
            # Else if q dominates p
            elif (v1[q] < v1[p] and v2[q] < v2[p]) or (v1[q] <= v1[p] and v2[q] < v2[p]) or (v1[q] < v1[p] and v2[q] <= v2[p]):
                n[p] += 1  # Increment the domination counter of p
        if n[p] == 0:
            rank[p] = 0
            if p not in F[0]:
                F[0].append(p)

    i = 0
    while F[i]:
        Q = []  # Used to store the members of the next front
        for p in F[i]:
            for q in S[p]:  # For each solution q, dominated by p:
                n[q] = n[q] - 1  # decrement the domination counter
                if n[q] == 0:  # If domination counter is zero, put solution q in next front
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        F.append(Q)

    del F[i]  # Delete last empty front
    return F


# Function to calculate crowding distance
def crowding_distance(v1, v2, front_f):
    L = len(front_f)
    distance = [0] * L
    sorted1 = sort_by_values(front_f, v1[:])
    sorted2 = sort_by_values(front_f, v2[:])
    distance[0] = 4444444444444444
    distance[L - 1] = 4444444444444444
    for i in range(1, L - 1):
        distance[i] = distance[i] + (v1[sorted1[i + 1]] - v1[sorted1[i - 1]]) / (max(v1) - min(v1))
    for i in range(1, L - 1):
        distance[i] = distance[i] + (v2[sorted2[i + 1]] - v2[sorted2[i - 1]]) / (max(v2) - min(v2))
    return distance


def recombination(population_f, shorelines_f, offspring_size_f, max_wp_distance_f):
    offspring_f = []
    while len(offspring_f) < offspring_size_f:
        crossover_route = False
        random.shuffle(population_f)
        for route_a in population_f:
            for route_b in population_f:
                if route_a is not route_b:
                    crossover_route = operators.crossover(route_a, route_b, shorelines_f, max_wp_distance_f)
                    if crossover_route:
                        offspring_f.append(crossover_route)
                        break
            if crossover_route:
                break
    return offspring_f


# Main program starts here
# Algorithm parameters
pop_size = 40
offspring_size = pop_size
max_gen = 1000
swaps = ['insert', 'move', 'delete', 'speed']
start_weight = 40
weights = {s: start_weight for s in swaps}
no_improvement_count = 0
no_improvement_limit = 5
max_edge_length = 200  # nautical miles

# Initialization
reinitialize = True

# Vessel characteristics
vessel = Vessel('Fairmaster')

# Route characteristics and navigation area
startLat = 48.021295
startLong = -5.352121
endLat = 46.403404
endLong = -52.865297
start_point = vg.Point(startLong, startLat)
end_point = vg.Point(endLong, endLat)
shorelines_shp_fp = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp'
shorelines = fiona.open(shorelines_shp_fp)

# Initialize population
if reinitialize:
    parents = initialization(start_point, end_point, vessel, pop_size, shorelines, max_edge_length)
    with open('output\parents', 'wb') as file:
        pickle.dump(parents, file)
else:
    with open('output\parents', 'rb') as file:
        parents = pickle.load(file)

gen_no = 0
while gen_no < max_gen:
    print('Generation nr. {}'.format(gen_no))

    # Evaluate objective values
    travel_times = [solution.travel_time() for solution in parents]
    print('Value: ', sum(travel_times) / len(travel_times))
    fuels = [solution.fuel(vessel) for solution in parents]

    # Non dominated sorting of solutions.
    # Returns list of fronts, which are lists of solution indices corresponding to non dominated solutions
    fronts = non_dominated_sort(travel_times[:], fuels[:])

    # Crowding distance
    # Returns crowding distance for each solution in each front in F. Boundary solutions of each front have inf distance
    crowding_distances = []
    for front in fronts:
        crowding_distances.append(crowding_distance(travel_times[:], fuels[:], front[:]))

    # Generating offsprings with crossover
    offspring = recombination(parents[:], shorelines, offspring_size, max_edge_length)

    # Mutate offspring
    if no_improvement_count > no_improvement_limit:
        weights = {s: start_weight for s in swaps}
        no_improvement_count = 0
    for child in offspring:
        weights, no_improvement_count = child.mutation(shorelines, weights, vessel, no_improvement_count, max_edge_length)

    # Combine parents and offspring
    combined_population = parents[:] + offspring[:]

    # Evaluate combined population objective values
    travel_times2 = [solution.travel_time() for solution in combined_population]
    fuels2 = [solution.fuel(vessel) for solution in combined_population]

    # Non dominated sorting of solutions.
    # Returns list of fronts, which are lists of solution indices corresponding to non dominated solutions
    fronts2 = non_dominated_sort(travel_times2[:], fuels2[:])

    # Crowding distance
    crowding_distances2 = []
    for front in fronts2:
        crowding_distances2.append(
            crowding_distance(travel_times2[:], fuels2[:], front[:]))

    # Fill new population with best fronts
    new_solutions = []
    i = 0
    while len(new_solutions) < pop_size:
        if len(fronts2[i]) < pop_size - len(new_solutions):
            new_solutions.extend(fronts2[i])
        else:
            sorted_front_indices = sort_by_values(range(len(fronts2[i])), crowding_distances2[i][:])
            sorted_front = [fronts2[i][sorted_front_indices[j]] for j in range(len(fronts2[i]))]

            sorted_front.reverse()
            for solution_idx in sorted_front:
                new_solutions.append(solution_idx)
                if len(new_solutions) == pop_size:
                    break
        i += 1
    parents = [combined_population[i] for i in new_solutions]
    gen_no += 1

# Get solutions from pareto front
pareto_front = fronts2[0]
pareto_solutions = []
for solution_index in pareto_front:
    pareto_solutions.append(combined_population[solution_index])

# Save solutions
output_file_name = 'pareto_solutions01'
with open('output/' + output_file_name, 'wb') as file:
    pickle.dump(pareto_solutions, file)

print('Save to: ' + output_file_name)

# Evaluate pareto objective values
travel_times = [solution.travel_time() for solution in pareto_solutions]
fuels = [solution.fuel(vessel) for solution in pareto_solutions]

# Plot the final front
print('Plotting')
function1 = [travel_time for travel_time in travel_times]
function2 = [fuel_consumption for fuel_consumption in fuels]
plt.xlabel('Travel time [hours]', fontsize=15)
plt.ylabel('Fuel consumption [tons]', fontsize=15)
plt.scatter(function1, function2)
plt.show()