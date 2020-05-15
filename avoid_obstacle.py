from shapely.geometry import Polygon, LineString, Point
from math import sqrt
import random
from solution import Route, Waypoint
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

random.seed(16)

polygon1 = [(2, 1), (2.3, 1.3), (2, 3), (2.4, 3.3), (3, 2), (4, 3), (5, 2), (6, 3.1), (3.8, 4), (5.5, 5.5), (4.1, 6.5), (2.8, 5), (2.1, 5), (1.2, 2.9)]
polygon2 = [(7, 7), (8, 4), (9, 5), (10, 3), (11, 9), (8, 8), (6, 8)]
polygon3 = [(1,0), (-2, -1), (-1, -2), (2, 0), (-2, 4), (-3, 2), (-2, 3)]
IN_polygons = [Polygon(polygon1), Polygon(polygon2)]
# polygons = [Polygon(polygon1)]

waypoint_1 = Waypoint(-1, 1)
waypoint_2 = Waypoint(12, 8.5)

route = Route([waypoint_1, waypoint_2], 16.7)
edge = route.edges[0]
shapely_line = LineString([(edge.v.x, edge.v.y), (edge.w.x, edge.w.y)])

# PLOT
plt.ion()
fig = plt.figure()
ax = fig.add_subplot()
x, y = IN_polygons[0].exterior.xy
for p in IN_polygons:
    x_p, y_p = p.exterior.xy
    ax.plot(x_p, y_p, color='darkgrey')
xx, yy = shapely_line.coords.xy
ax.plot(xx, yy)

IN_v = Point(edge.v.x, edge.v.y)
IN_w = Point(edge.w.x, edge.w.y)
IN_line_vw = LineString([IN_v, IN_w])

IN_iteration = 0

plt.grid()
plt.show()
def get_route_segment(line_vw, polygons, segments, end_point, count, count_lim=4, extend=.5):
    def intersect_polygon(line, polygon_list):
        for a_polygon in iter(polygon_list):
            intersect = line.intersection(a_polygon)
            if intersect:
                return a_polygon
        return False
    count += 1
    # plt.clf()
    if count > 200:
        return segments, count, count_lim
    polygon = intersect_polygon(line_vw, polygons)
    v, w = line_vw.coords
    v, w = Point(v), Point(w)
    if not polygon:
        segments.append(line_vw)
        x_s, y_s = line_vw.coords.xy
        ax.plot(x_s, y_s, color='black')
        plt.pause(0.5)
        count = 0
        return segments, count
    else:
        # Create points a and b, intersections of edge and polygon
        line_ab = polygon.intersection(line_vw)
        if line_ab.geom_type == 'MultiLineString':
            line_ab = line_ab[0]
        x_ab, y_ab = line_ab.coords.xy
        x_c, y_c = line_ab.centroid.coords.xy

        ax.plot(x_ab, y_ab, color='green')
        plt.pause(0.001)

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
        if line_gh.geom_type == 'MultiLineString':
            x_first, y_first = line_gh[0].coords.xy
            x_last, y_last = line_gh[-1].coords.xy
            line_gh = LineString([Point(x_first[0], y_first[0]), Point(x_last[-1], y_last[-1])])
        x_gh, y_gh = line_gh.coords.xy
        x_g, x_h = x_gh[0], x_gh[1]
        y_g, y_h = y_gh[0], y_gh[1]

        ax.plot(x_gh, y_gh, color='red')
        plt.pause(0.001)

        # Create points d and f, intersections of perpendicular bisector and envelope
        bounds = polygon.envelope.bounds
        x1_mbr = bounds[0] - extend
        x2_mbr = bounds[2] + extend
        y1_mbr = bounds[1] - extend
        y2_mbr = bounds[3] + extend
        extended_mbr = Polygon([(x1_mbr, y1_mbr), (x2_mbr, y1_mbr), (x2_mbr, y2_mbr), (x1_mbr, y2_mbr)])
        line_df = extended_mbr.intersection(perpendicular_bisector)
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

        # Create vg1 and vh1
        line_vg1 = LineString([v, g1])
        line_vh1 = LineString([v, h1])
        if not line_vg1.intersects(polygon):
            if count < count_lim:
                segments, count = get_route_segment(line_vg1, polygons, segments, end_point, count)

                if not Point(segments[-1].coords[1]) == end_point:
                    line_g1w = LineString([g1, w])
                    x_g1w, y_g1w = line_g1w.coords.xy
                    ax.plot(x_g1w, y_g1w)
                    segments, count = get_route_segment(line_g1w, polygons, segments, end_point, count)
            else:
                segments, count = get_route_segment(line_vg1, polygons, segments, end_point, count)

                if not Point(segments[-1].coords[1]) == end_point:
                    line_g1_end = LineString([g1, end_point])
                    segments, count = get_route_segment(line_g1_end, polygons, segments, end_point, count)

            return segments, count
        elif not line_vh1.intersects(polygon):
            if count < count_lim:
                segments, count = get_route_segment(line_vh1, polygons, segments, end_point, count)

                if not Point(segments[-1].coords[1]) == end_point:
                    line_h1w = LineString([h1, w])
                    x_h1w, y_h1w = line_h1w.coords.xy
                    ax.plot(x_h1w, y_h1w)
                    segments, count = get_route_segment(line_h1w, polygons, segments, end_point, count)
            else:
                segments, count = get_route_segment(line_vh1, polygons, segments, end_point, count)

                if not Point(segments[-1].coords[1]) == end_point:
                    line_h1_end = LineString([h1, end_point])
                    segments, count = get_route_segment(line_h1_end, polygons, segments, end_point, count)

            return segments, count
        else:
            if count < count_lim:
                segments, count = get_route_segment(line_vg1, polygons, segments, end_point, count)

                if not Point(segments[-1].coords[1]) == end_point:
                    line_g1w = LineString([g1, w])
                    x_g1w, y_g1w = line_g1w.coords.xy
                    ax.plot(x_g1w, y_g1w)
                    segments, count = get_route_segment(line_g1w, polygons, segments, end_point, count)
                    store_segments = segments

                    if not Point(segments[-1].coords[1]) == end_point:
                        segments, count = get_route_segment(line_vh1, polygons, segments, end_point, count)
                        if len(segments) < len(store_segments):
                            if not Point(segments[-1].coords[1]) == end_point:
                                line_h1w = LineString([h1, w])
                                x_h1w, y_h1w = line_h1w.coords.xy
                                ax.plot(x_h1w, y_h1w)
                                segments, count = get_route_segment(line_h1w, polygons, segments, end_point, count)
                        else:
                            segments = store_segments
            else:
                segments, count = get_route_segment(line_vg1, polygons, segments, end_point, count)
                if not Point(segments[-1].coords[1]) == end_point:
                    line_g1_end = LineString([g1, end_point])
                    segments, count = get_route_segment(line_g1_end, polygons, segments, end_point, count)
                    store_segments = segments

                    if not Point(segments[-1].coords[1]) == end_point:
                        segments, count = get_route_segment(line_vh1, polygons, segments, end_point, count)
                        if len(segments) < len(store_segments):
                            if not Point(segments[-1].coords[1]) == end_point:
                                line_h1_end = LineString([g1, end_point])
                                segments, count = get_route_segment(line_h1_end, polygons, segments, end_point, count)
                        else:
                            segments = store_segments

        return segments, count

# ax.plot(x_ab, y_ab, color='firebrick')
# ax.plot(x_gh, y_gh, color='orange')
# ax.plot(x_g1, y_g1, '.', color='purple', markersize=10)
# ax.plot(x_h1, y_h1, '.', color='darkblue', markersize=10)


IN_segment_list = []
segment_list, iteration = get_route_segment(IN_line_vw, IN_polygons, IN_segment_list, IN_w, IN_iteration)

if iteration > 200:
    print('Iteration > ', 200)
else:
    print(iteration)

    for segment in segment_list:
        print(segment)
        x_s, y_s = segment.coords.xy
        ax.plot(x_s, y_s, color='black')
