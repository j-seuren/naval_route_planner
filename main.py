import pandas as pd
from pyvisgraph import graph

data_fp = 'C:/dev/data'
seca_areas = pd.read_csv(data_fp + '/seca_areas.csv').drop('index', axis=1)

# line_a_point_a_long
# line_a_point_a_lat
# line_a_point_b_long
# line_a_point_b_lat
# line_b_point_a_long
# line_b_point_a_lat
# line_b_point_b_long
# line_b_point_b_lat

print(seca_areas.head())
