from route import Waypoint
shorelines_shp_fp = 'C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp'
wp = Waypoint(113.54307238928701, 1.655213702150351)
print(wp.in_polygon(shorelines_shp_fp))