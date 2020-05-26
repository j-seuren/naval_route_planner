# Importing required modules
import sys
import math
import random
import matplotlib.pyplot as plt
import pickle
import pyvisgraph as vg
import solution
import pandas as pd
import fiona
from copy import deepcopy
import operators

random.seed(1)


# Generate as set of initial solutions
def get_initial_solutions(start_point_f, end_point_f, vessel_speed_f, shorelines_f, pop_size_f):
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
        edge = solution.Edge(v, w, vessel_speed_f)
        v = w
        edges.append(edge)

    visibility_route = solution.Route(edges)

    # Insert code to make visibility_route feasible.
    # With respect to waypoint locations, edge intersections and vessel speed

    initial_routes = []
    for i in range(pop_size_f):
        sys.stdout.write('\rGenerating initial routes {0}/{1}'.format(i+1, pop_size_f))
        sys.stdout.flush()
        init_route = deepcopy(visibility_route)
        for j in range(10):
            init_route.insert_waypoint(bisector_length_ratio=0.5, polygons=shorelines_f, printing=False)
        initial_routes.append(init_route)

    print('\n')
    return initial_routes


# First function to optimize
def travel_time(solution_f):
    # return -solution_f ** 2
    return solution_f.travel_time()


# Second function to optimize
def fuel_consumption(solution_f, fuel_rate_f):
    # return -(solution_f - 2) ** 2
    return solution_f.fuel(fuel_rate_f)


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
            if (v1[p] > v1[q] and v2[p] > v2[q]) or (v1[p] >= v1[q] and v2[p] > v2[q]) or (v1[p] > v1[q] and v2[p] >= v2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (v1[q] > v1[p] and v2[q] > v2[p]) or (v1[q] >= v1[p] and v2[q] > v2[p]) or (v1[q] > v1[p] and v2[q] >= v2[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if p not in F[0]:
                F[0].append(p)

    i = 0
    while F[i]:
        Q = []
        for p in F[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        F.append(Q)

    del F[len(F) - 1]
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


def generate_offspring(population_f, shorelines_f, offspring_size_f):
    offspring_f = []
    while len(offspring_f) < offspring_size_f:
        crossover_route = False
        random.shuffle(population_f)
        for route_a in population_f:
            for route_b in population_f:
                if route_a is not route_b:
                    crossover_route = operators.crossover(route_a, route_b, shorelines_f)
                    if crossover_route:
                        offspring_f.append(crossover_route)
                        break
            if crossover_route:
                break
    return offspring_f


# Main program starts here
pop_size = 30
offspring_size = pop_size
max_gen = 500

# Initialization
# Example points
startLat = 48.021295
startLong = -5.352121
endLat = 46.403404
endLong = -52.865297

# Vessel speed
vessel_name = 'Fairmaster'
speeds_FM = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name=vessel_name)
vessel_speed = speeds_FM['Speed'][0]
fuel_rate = speeds_FM['Fuel'][0]
start_point = vg.Point(startLong, startLat)
end_point = vg.Point(endLong, endLat)
shorelines_shp_fp = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp'
shorelines = fiona.open(shorelines_shp_fp)

# Initialize population
parents = get_initial_solutions(start_point, end_point, vessel_speed, shorelines, pop_size)
gen_no = 0
while gen_no < max_gen:
    print('Generation nr. {}'.format(gen_no))

    # Evaluate objective values
    travel_times = [travel_time(solution) for solution in parents]
    fuels = [fuel_consumption(solution, fuel_rate) for solution in parents]

    # Non dominated sorting of solutions.
    # Returns list of fronts, which are lists of solution indices corresponding to non dominated solutions
    fronts = non_dominated_sort(travel_times[:], fuels[:])

    # Crowding distance
    # Returns crowding distance for each solution in each front in F. Boundary solutions of each front have inf distance
    crowding_distances = []
    for front in fronts:
        crowding_distances.append(crowding_distance(travel_times[:], fuels[:], front[:]))

    # Generating offsprings with crossover
    offspring = generate_offspring(parents[:], shorelines, offspring_size)

    # Mutate offspring
    for child in offspring:
        # Mutate half
        if random.random() < 0.5:
            child.mutation(shorelines)

    # Combine parents and offspring
    combined_population = parents[:] + offspring[:]

    # Evaluate combined population objective values
    travel_times2 = [travel_time(solution) for solution in combined_population]
    fuels2 = [fuel_consumption(solution, fuel_rate) for solution in combined_population]

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

# Plot the final front
print('Plotting')
function1 = [travel_time for travel_time in travel_times]
function2 = [fuel_consumption for fuel_consumption in fuels]
plt.xlabel('Travel time [hours]', fontsize=15)
plt.ylabel('Fuel consumption [tons]', fontsize=15)
plt.scatter(function1, function2)
plt.show()

# Save solutions
with open('output/population01', 'wb') as file:
    pickle.dump(parents, file)
