# Importing required modules
import random
import math
from operators import crossover
import cProfile, pstats, io


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
            if (v1[p] < v1[q] and v2[p] < v2[q]) or\
                    (v1[p] <= v1[q] and v2[p] < v2[q]) or\
                    (v1[p] < v1[q] and v2[p] <= v2[q]):
                if q not in S[p]:
                    S[p].append(q)  # Add q to the set of solutions dominated by p
            # Else if q dominates p
            elif (v1[q] < v1[p] and v2[q] < v2[p]) or\
                    (v1[q] <= v1[p] and v2[q] < v2[p]) or\
                    (v1[q] < v1[p] and v2[q] <= v2[p]):
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
def crowding_distance(v1, v2, front):
    L = len(front)
    distance = [0] * L
    sorted1 = sort_by_values(front, v1[:])
    sorted2 = sort_by_values(front, v2[:])
    distance[0] = 4444444444444444
    distance[L - 1] = 4444444444444444
    for i in range(1, L - 1):
        distance[i] = distance[i] + (v1[sorted1[i + 1]] - v1[sorted1[i - 1]]) / (max(v1) - min(v1))
    for i in range(1, L - 1):
        distance[i] = distance[i] + (v2[sorted2[i + 1]] - v2[sorted2[i - 1]]) / (max(v2) - min(v2))
    return distance


def recombination(population, rtree_idx, polygons, offspring_size, max_distance, vessel):
    offspring = []
    while len(offspring) < offspring_size:
        pop_copy = population[:]
        random.shuffle(pop_copy)
        while len(pop_copy) > 1 and len(offspring) < offspring_size:
            route_a = pop_copy.pop()
            route_b = pop_copy.pop()
            assert len(route_a.edges) > 2 and len(route_b.edges) > 2, 'One or both routes have too few edges'
            crossover_route = crossover(route_a, route_b, rtree_idx, polygons, max_distance, vessel)
            if crossover_route:
                offspring.append(crossover_route)
    return offspring


def nsga_ii(parents, vessel, rtree_idx, polygons, offspring_size, max_gen, max_distance, swaps):
    gen_no = 0
    pop_size = len(parents)

    while gen_no < max_gen:
        print('Generation nr. {}'.format(gen_no))

        # Evaluate objective values
        travel_times = [solution.travel_time for solution in parents]
        print('Value: ', min(travel_times), sum(travel_times) / len(travel_times))
        fuels = [solution.fuel_consumption for solution in parents]

        # Non dominated sorting of solutions.
        # Returns list of fronts, which are lists of solution indices corresponding to non dominated solutions
        fronts = non_dominated_sort(travel_times[:], fuels[:])

        # Crowding distance
        # Returns crowding distance for each solution in each front in F.
        # Boundary solutions of each front have inf distance
        crowding_distances = []
        for front in fronts:
            crowding_distances.append(crowding_distance(travel_times[:], fuels[:], front[:]))

        # Generating offsprings with crossover
        offspring = recombination(parents[:], rtree_idx, polygons, offspring_size, max_distance, vessel)

        # Mutate offspring
        for child in offspring:
            child.mutate(rtree_idx, polygons, swaps, vessel, max_distance)

        # Combine parents and offspring
        combined_population = parents[:] + offspring[:]

        # Evaluate combined population objective values
        travel_times2 = [solution.travel_time for solution in combined_population]
        fuels2 = [solution.fuel_consumption for solution in combined_population]

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

    return combined_population, fronts2