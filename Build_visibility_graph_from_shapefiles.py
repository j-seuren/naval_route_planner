import os
import pyvisgraph as vg
# import geopandas as gpd
import shapefile

# Parameters
data_fp = 'C:\dev\data'
startPort = 'SCHEVENINGEN'
endPort = 'KINGSTON UPON HULL'
sea = 'North Sea'

# World Shorelines
GSHHS_l_L1_fp = os.path.join(data_fp, 'gshhg-shp-2.3.7', 'GSHHS_shp', 'l', 'GSHHS_l_L1')
input_shapefile = shapefile.Reader(GSHHS_l_L1_fp)
output_graphfile = 'output/GSHHS_l_L1.graph'

# Number of CPU cores on host computer
workers = 1

# Get the shoreline shapes from the shape file
shapes = input_shapefile.shapes()
print('The shapefile contains {} shapes.'.format(len(shapes)))

# Create a list of polygons, where each polygon corresponds to a shape
polygons = []
for shape in shapes:
    polygon = []
    for point in shape.points:
        polygon.append(vg.Point(point[0], point[1]))
    polygons.append(polygon)

# Start building the visibility graph
graph = vg.VisGraph()
print('Starting building visibility graph')
graph.build(polygons, workers=workers)
print('Finished building visibility graph')

# Save the visibility graph to a file
graph.save(output_graphfile)
print('Saved visibility graph to file: {}'.format(output_graphfile))