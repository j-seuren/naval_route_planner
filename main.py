import pandas as pd
import pyvisgraph as vg
import get_data as get
import pickle

# line_a_point_a_long
# line_a_point_a_lat
# line_a_point_b_long
# line_a_point_b_lat
# line_b_point_a_long
# line_b_point_a_lat
# line_b_point_b_long
# line_b_point_b_lat

seca_areas = get.seca()



print(seca_areas.head())

# Load shortest path
with open('output\shortest_path', 'rb') as file:
    shortest_path = pickle.load(file)

print(shortest_path[0].y)
print(seca_areas[['latitude']].min())
print(shortest_path[0].y < seca_areas[['latitude']].min())

print(shortest_path[0].in_seca(seca_areas))
# Build visibility graph
