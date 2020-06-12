import pickle
from classes import Vessel
import matplotlib.pyplot as plt

with open('output\pareto_solutions01', 'rb') as file:
    solutions = pickle.load(file)



def Average(lst):
    return sum(lst) / len(lst)


vessel = Vessel('Fairmaster')
travel_times = [solution.travel_time() for solution in solutions]
fuel_consumptions = [solution.fuel(vessel) for solution in solutions]
distances = [solution.miles for solution in solutions]

min_distance_route = solutions[distances.index(min(distances))]
min_time_route = solutions[travel_times.index(min(travel_times))]
min_fuel_route = solutions[fuel_consumptions.index(min(fuel_consumptions))]

print('Nr. of solutions: ', len(solutions))
print('Min/Avg/Max distance [nm]', "%4.1f %4.1f %4.1f" % (min(distances), Average(distances), max(distances)))
print('Min/Avg/Max fuel consumption [tons]', "%4.1f %4.1f %4.1f" % (min(fuel_consumptions), Average(fuel_consumptions), max(fuel_consumptions)))
print('Min/Avg/Max travel time [hours]', "%4.1f %4.1f %4.1f" % (min(travel_times), Average(travel_times), max(travel_times)))

# Speed change during route
for solution in solutions[:5]:
    y = [edge.speed for edge in solution.edges]
    print(solution.travel_time())
    plt.plot(y)

plt.ylabel('Speed [knots]')
plt.show()

# for solution in solutions:
#     print(solution.distance)