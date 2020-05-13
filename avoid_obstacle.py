from shapely.geometry import Polygon, LineString, Point
from math import sqrt
import random
from solution import Route, Waypoint
import matplotlib.pyplot as plt
import numpy as np

random.seed(3)

polygon1 = [(2, 1), (4, 3), (5, 2), (6, 3.1), (3.8, 4), (5.5, 5.5), (4.1, 6.5), (2.8, 5), (2.1, 6), (1.2, 3.9)]
polygon2 = [(7, 7), (8, 4), (9, 5), (10, 3), (11, 9), (8, 8), (6, 8)]
polygons = [Polygon(polygon1), Polygon(polygon2)]

waypoint_1 = Waypoint(-1, 1)
waypoint_2 = Waypoint(12, 8.5)

route = Route([waypoint_1, waypoint_2], 16.7)
edge = route.edges[0]
shapely_line = LineString([(edge.v.x, edge.v.y), (edge.w.x, edge.w.y)])

# PLOT
x, y = polygons[0].exterior.xy
xxx, yyy = polygons[1].exterior.xy
xx, yy = shapely_line.coords.xy
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x, y)
ax.plot(xxx, yyy)
ax.plot(xx, yy)

def avoid_obstacle(edge, polygons):
    polygon = edge.intersec_polygons(polygons, return_polygon=True)
    shapely_line = LineString([(edge.v.x, edge.v.y), (edge.w.x, edge.w.y)])
    if polygon:
        # Create points a and b, intersections of edge and polygon
        line_ab = polygon.intersection(shapely_line)
        x_ab, y_ab = line_ab.coords.xy
        x_c, y_c = line_ab.centroid.coords.xy

        # Create perpendicular bisector of line ab
        slope_ab = (y_ab[1] - y_ab[0]) / (x_ab[1] - x_ab[0])
        envelope = polygon.envelope
        big_m = envelope.length
        x_1 = x_c[0] + sqrt(big_m ** 2 / (1 + 1 / slope_ab ** 2))
        x_2 = x_c[0] - sqrt(big_m ** 2 / (1 + 1 / slope_ab ** 2))
        y_1 = -(1 / slope_ab) * (x_1 - x_c[0]) + y_c[0]
        y_2 = -(1 / slope_ab) * (x_2 - x_c[0]) + y_c[0]
        perpendicular_bisector = LineString([(x_1, y_1), (x_2, y_2)])

        # Create points g and h, intersections of perpendicular bisector and polygon
        line_gh = polygon.intersection(perpendicular_bisector)
        x_gh, y_gh = line_gh.coords.xy
        x_g, x_h = x_gh[0], x_gh[1]
        y_g, y_h = y_gh[0], y_gh[1]

        # Create points d and f, intersections of perpendicular bisector and envelope
        line_df = polygon.envelope.intersection(perpendicular_bisector)
        x_df, y_df = line_df.coords.xy
        x_d, x_f = x_df[0], x_df[1]
        y_d, y_f = y_df[0], y_df[1]

        # Randomly choose points g1 and g1 from line segments dg and hf, respectively
        while True:
            u_1 = random.uniform(0, 1)
            x_g1 = u_1 * (x_d - x_g) + x_g
            y_g1 = u_1 * (y_d - y_g) + y_g

            g1 = Point(x_g1, y_g1)
            for poly in iter(polygons):
                if not g1.within(poly):  # ADD: shape()
                    continue
            break

        while True:
            u_2 = random.uniform(0, 1)
            x_h1 = u_2 * (x_f - x_h) + x_h
            y_h1 = u_2 * (y_f - y_h) + y_h

            h1 = Point(x_h1, y_h1)
            for poly in iter(polygons):
                if not h1.within(poly):  # ADD: shape()
                    continue
            break

        # Create sg1 and sh1
        line_sg1 = LineString([(edge.v.x, edge.v.y), (x_g1, y_g1)])
        line_sh1 = LineString([(edge.v.x, edge.v.y), (x_h1, y_h1)])

        # Check if lines intersect polygon
        sg1_crosses_polygon = line_sg1.intersects(polygon)
        sh1_crosses_polygon = line_sh1.intersects(polygon)

        print(sg1_crosses_polygon, sh1_crosses_polygon)

        if not sg1_crosses_polygon:



        return x_ab, y_ab, x_gh, y_gh, x_df, y_df, envelope, x_g1, y_g1, x_h1, y_h1, line_sg1, line_sh1

x_ab, y_ab, x_gh, y_gh, x_df, y_df, qqq, x_g1, y_g1, x_h1, y_h1, line_sg1, line_sh1 = avoid_obstacle(edge, polygons)

ax.plot(x_ab, y_ab)
ax.plot(x_gh, y_gh)
ax.plot(x_df, y_df, '-.')
xxxx, yyyy = qqq.exterior.xy
ax.plot(xxxx, yyyy)
ax.plot(x_g1, y_g1, '.', markersize=10)
ax.plot(x_h1, y_h1, '.', markersize=10)
sg1_x, sg1_y = line_sg1.coords.xy
sh1_x, sh1_y = line_sh1.coords.xy
ax.plot(sg1_x, sg1_y)
ax.plot(sh1_x, sh1_y)

# ax.plot(line_df.coords.xy)
plt.show()