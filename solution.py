from haversine import haversine
from shapely.geometry import shape, Polygon, Point, LineString
from shapely.ops import split
from math import atan2, cos, sin, degrees, pi, sqrt
import random
import numpy as np
from numpy import sign
from numpy.linalg import det
import matplotlib.pyplot as plt
from plot_edge import plot_edge


class Waypoint:
    def __init__(self, x, y, waypoint_id=-1):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return "(%.2f, %.2f)" % (self.x, self.y)

    def __repr__(self):
        return "wp(%.2f, %.2f)" % (self.x, self.y)

    def __eq__(self, wp):
        return wp and self.x == wp.x and self.y == wp.y

    def __ne__(self, wp):
        return not self.__eq__(wp)

    def in_polygon(self, polygons, return_polygon=False):
        for polygon in iter(polygons):
            shapely_point = Point(self.x, self.y)
            if shapely_point.within(shape(polygon['geometry'])):
                print('{} is in polygon.'.format(self))
                if return_polygon:
                    return polygon
                else:
                    return True
        return False

    def closest_point(self, polygon, length=0.001):
        """Assumes p is interior to a polygon. Returns the
            closest point c outside the polygon to p, where the distance from c to
            the intersect point from p to the edge of the polygon is length."""
        # Finds point closest to p, but on a edge of the polygon.
        # Solution from http://stackoverflow.com/a/6177788/4896361


class Edge:
    def __init__(self, v, w, speed):
        self.v = v  # Waypoint 1
        self.w = w  # Waypoint 2
        self.speed = float(speed)  # Knots
        self.distance = 0.539957 * haversine((v.y, v.x), (w.y, w.x))  # Nautical miles
        self.bearing = (degrees(
            atan2(cos(v.y) * sin(w.y) - sin(v.y) * cos(w.y) * cos(w.x - v.x), sin(w.x - v.x) * cos(w.y)))
                        + 360) % 360

    def __repr__(self):
        return "e({!r}, {!r})".format(self.v, self.w)

    def travel_time(self):  # Hours
        return self.distance / self.speed

    def fuel(self, fuel_rate):  # Fuel tons
        return fuel_rate * self.travel_time() / 24

    def crosses_polygon(self, polygons, return_polygon=False):
        shapely_line = LineString([(self.v.x, self.v.y), (self.w.x, self.w.y)])
        for polygon in iter(polygons):
            intersect = shapely_line.intersection(shape(polygon['geometry']))
            if intersect:
                print('{} intersects polygon.'.format(self))
                if return_polygon:
                    return polygon
                else:
                    return True
        return False

    def avoid_obstacle(self, polygons_input):
        def get_route_segment(line_vw, polygons, end_point, area,
                              extend, segments=[], checked_polygons=dict(),
                              direction=None, count=0):
            def intersect_polygon(line, polygon_list):
                intersect_dict = dict()
                min_distance = 10000000
                line_nr = 0
                intersects = False
                closest_line_nr = 0
                for a_polygon in iter(polygon_list):
                    polygon_shape = shape(a_polygon['geometry'])
                    intersect = line.intersection(polygon_shape)
                    if intersect:
                        intersects = True
                        line_nr += 1
                        intersect_dict[line_nr] = {'line_key': intersect,
                                                   'polygon_key': polygon_shape}
                if intersects:
                    for nr in intersect_dict:
                        distance_to_line = intersect_dict[nr]['line_key'].distance(Point(line.coords[0]))
                        if distance_to_line < min_distance:
                            closest_line_nr = nr
                            min_distance = distance_to_line
                    return intersect_dict[closest_line_nr]['polygon_key']

                return False
            count += 1
            print(count)
            v, w = line_vw.coords
            v, w = Point(v), Point(w)

            line_v_end = LineString([v, end_point])
            if not intersect_polygon(line_v_end, polygons):
                x_s, y_s = line_v_end.coords.xy
                ax.plot(x_s, y_s, color='black')
                plt.pause(0.5)
                segments.append(line_v_end)
                return segments

            polygon = intersect_polygon(line_vw, polygons)

            if not polygon:
                x_s, y_s = line_vw.coords.xy
                ax.plot(x_s, y_s, color='black')
                plt.pause(0.5)
                segments.append(line_vw)
                return segments
            else:
                # Create points a and b, intersections of edge and polygon
                line_ab = polygon.intersection(line_vw)
                if line_ab.geom_type == 'MultiLineString':
                    line_ab = line_ab[0]
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

                if line_gh.geom_type == 'MultiLineString':
                    x_first, y_first = line_gh[0].coords.xy
                    x_last, y_last = line_gh[-1].coords.xy
                    line_gh = LineString([(x_first[0], y_first[0]), (x_last[-1], y_last[-1])])
                x_gh, y_gh = line_gh.coords.xy
                x_g, x_h = x_gh[0], x_gh[1]
                y_g, y_h = y_gh[0], y_gh[1]

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
                        if not g1.within(shape(poly['geometry'])):
                            continue
                    break
                while True:
                    u_2 = random.uniform(0, 1)
                    x_h1 = u_2 * (x_f - x_h) + x_h
                    y_h1 = u_2 * (y_f - y_h) + y_h
                    h1 = Point(x_h1, y_h1)
                    for poly in iter(polygons):
                        if not h1.within(shape(poly['geometry'])):
                            continue
                    break

                # Create vg1 and vh1
                line_vg1 = LineString([v, g1])
                line_vh1 = LineString([v, h1])

                # Determine directions
                direction_g1 = np.sign((x_ab[1] - x_ab[0]) * (y_g1 - y_ab[0]) - (y_ab[1] - y_ab[0]) * (x_g1 - x_ab[0]))
                direction_h1 = np.sign((x_ab[1] - x_ab[0]) * (y_h1 - y_ab[0]) - (y_ab[1] - y_ab[0]) * (x_h1 - x_ab[0]))

                # If polygon partly outside designated range,
                if polygon.boundary.intersects(area.boundary):
                    pt = polygon.boundary.intersection(area.boundary)[-1]
                    x_pt, y_pt = pt.coords.xy
                    side_outside_polygon = np.sign(
                        (x_ab[1] - x_ab[0]) * (y_pt[0] - y_ab[0]) - (y_ab[1] - y_ab[0]) * (x_pt[0] - x_ab[0]))
                    if direction_h1 == side_outside_polygon:
                        checked_polygons[polygon.area] = direction_g1
                        segments = get_route_segment(line_vg1, polygons, end_point, area,
                                                     extend, segments, checked_polygons, direction_g1, count)
                        if not segments[-1].intersects(end_point):
                            line_g1w = LineString([g1, w])
                            segments = get_route_segment(line_g1w, polygons, end_point, area,
                                                         extend, segments, checked_polygons, direction_g1, count)
                    else:
                        checked_polygons[polygon.area] = direction_h1
                        segments = get_route_segment(line_vh1, polygons, end_point, area,
                                                     extend, segments, checked_polygons, direction_h1, count)
                        if not segments[-1].intersects(end_point):
                            line_h1w = LineString([h1, w])
                            segments = get_route_segment(line_h1w, polygons, end_point, area,
                                                         extend, segments, checked_polygons, direction_h1, count)
                elif polygon.area in checked_polygons:
                    if checked_polygons[polygon.area] == direction_g1:
                        segments = get_route_segment(line_vg1, polygons, end_point, area,
                                                     extend, segments, checked_polygons, direction_g1, count)
                        if not segments[-1].intersects(end_point):
                            line_g1w = LineString([g1, w])
                            segments = get_route_segment(line_g1w, polygons, end_point, area,
                                                         extend, segments, checked_polygons, direction_g1, count)
                    else:
                        segments = get_route_segment(line_vh1, polygons, end_point, area,
                                                     extend, segments, checked_polygons, direction_h1, count)
                        if not segments[-1].intersects(end_point):
                            line_h1w = LineString([h1, w])
                            segments = get_route_segment(line_h1w, polygons, end_point, area,
                                                         extend, segments, checked_polygons, direction_h1, count)
                else:
                    checked_polygons[polygon.area] = direction_g1
                    segments = get_route_segment(line_vg1, polygons, end_point, area,
                                                 extend, segments, checked_polygons, direction_g1, count)
                    if not segments[-1].intersects(end_point):
                        line_g1w = LineString([g1, w])
                        segments = get_route_segment(line_g1w, polygons, end_point, area,
                                                     extend, segments, checked_polygons, direction_g1, count)

                    checked_polygons[polygon.area] = direction_h1
                    segments = get_route_segment(line_vh1, polygons, end_point, area,
                                                 extend, segments, checked_polygons, direction_h1, count)
                    if not segments[-1].intersects(end_point):
                        line_h1w = LineString([h1, w])
                        segments = get_route_segment(line_h1w, polygons, end_point, area,
                                                     extend, segments, checked_polygons, direction_h1, count)

                return segments

        v_input = Point(self.v.x, self.v.y)
        w_input = Point(self.w.x, self.w.y)
        line_vw_input = LineString([v_input, w_input])
        area_radius = 0.5 * line_vw_input.length
        extend_input = 0.5
        area_input = line_vw_input.centroid.buffer(area_radius)

        # PLOT
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot()
        # for p in polygons_input:
        #     x_p, y_p = p.exterior.xy
        #     ax.plot(x_p, y_p, color='darkgrey')
        xx, yy = line_vw_input.coords.xy
        ax.plot(xx, yy)

        plt.grid()
        plt.show()

        # x_area, y_area = area_input.exterior.coords.xy
        # ax.plot(x_area, y_area)

        segment_list = get_route_segment(line_vw_input, polygons_input, w_input, area_input, extend_input)

        segments_dict = dict()
        routes_dict = dict()
        route_nr = 0
        for segment in segment_list:
            # Create route for every segment ending in the end_point
            if Point(segment.coords[1]) == w_input:
                route_nr += 1
                segments_dict[route_nr] = []
                # Construct route in reverse order
                for i, item in enumerate(reversed(segment_list[:segment_list.index(segment) + 1])):
                    # Append the last edge (i == 0) in the route list
                    if i == 0:
                        segments_dict[route_nr].append(item)
                    # If end point of current segment is equal to start point of previous segment:
                    #     append segment to list
                    elif item.coords[1] == segments_dict[route_nr][-1].coords[0]:
                        segments_dict[route_nr].append(item)
                        if Point(segments_dict[route_nr][-1].coords[0]) == v_input:
                            segments_dict[route_nr] = reversed(segments_dict[route_nr])
                            # Create Waypoint and Edge classes
                            waypoints = []
                            edges = []
                            for idx, route_segment in enumerate(segments_dict[route_nr]):
                                if idx == 0:
                                    v = Waypoint(route_segment.coords[0][0], route_segment.coords[0][1])
                                    w = Waypoint(route_segment.coords[1][0], route_segment.coords[1][1])
                                    waypoints.append(v)
                                else:
                                    v = waypoints[-1]
                                    w = Waypoint(route_segment.coords[1][0], route_segment.coords[1][1])
                                waypoints.append(w)
                                edges.append(Edge(v, w, self.speed))

                            routes_dict[route_nr] = {'waypoints': waypoints,
                                                     'edges': edges}
                            break
                    else:
                        continue

        return routes_dict


class Route:
    def __init__(self, edges):
        self.edges = edges
        wps = [edge.v for edge in edges]
        wps.append(edges[-1].w)
        self.waypoints = wps
        self.distance = sum(edge.distance for edge in self.edges)
        self.history = []

    def travel_time(self):  # Hours
        return sum(edge.travel_time() for edge in self.edges)

    def fuel(self, fuel_rate):  # Fuel tons
        return sum(edge.fuel(fuel_rate) for edge in self.edges)

    def insert_waypoint(self, bisector_length_ratio, polygons, printing=True):
        # Pick random edge and adjacent waypoints
        edge = self.edges[random.randint(0, len(self.edges) - 1)]
        x_v, y_v = edge.v.x, edge.v.y
        x_w, y_w = edge.w.x, edge.w.y

        # Edge center
        x_center = x_v + 0.5 * (x_w - x_v)
        y_center = y_v + 0.5 * (y_w - y_v)

        slope = (y_w - y_v) / (x_w - x_v)

        # Get half length of the edge's perpendicular bisector
        half_length_bisector = 1 / 2 * bisector_length_ratio * sqrt((y_w - y_v) ** 2 + (x_w - x_v) ** 2)

        # Calculate outer points of polygon
        x_a = x_center + sqrt(half_length_bisector ** 2 / (1 + 1 / slope ** 2))
        x_b = x_center - sqrt(half_length_bisector ** 2 / (1 + 1 / slope ** 2))
        y_a = -(1 / slope) * (x_a - x_center) + y_center
        y_b = -(1 / slope) * (x_b - x_center) + y_center

        polygon = Polygon([[x_v, y_v], [x_a, y_a], [x_w, y_w], [x_b, y_b]])

        # Create point within polygon bounds and check if polygon contains point
        min_x, min_y, max_x, max_y = polygon.bounds
        new_waypoint = new_edge_1 = new_edge_2 = []
        while True:
            point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if not polygon.contains(point):
                continue
            new_waypoint = Waypoint(point.x, point.y)
            if new_waypoint.in_polygon(polygons):
                continue
            new_edge_1 = Edge(edge.v, new_waypoint, edge.speed)
            new_edge_2 = Edge(new_waypoint, edge.w, edge.speed)
            if new_edge_1.crosses_polygon(polygons) or new_edge_2.crosses_polygon(polygons):
                continue
            break

        # Insert new waypoint in route
        waypoint_idx = self.waypoints.index(edge.w)
        self.waypoints[waypoint_idx:waypoint_idx] = [new_waypoint]

        # Remove old edge and insert two new edges
        edge_idx = self.edges.index(edge)
        del self.edges[edge_idx]
        self.edges[edge_idx:edge_idx] = [new_edge_1, new_edge_2]
        self.distance = sum(edge.distance for edge in self.edges)
        self.history.append('Insertion {0}, idx {1}'.format(new_waypoint, waypoint_idx))
        for e, edge in enumerate(self.edges[:-1]):
            if edge.w != self.edges[e + 1].v:
                print('Error: edges are not linked')

    def delete_random_waypoint(self, polygons):
        count = 0
        delete = False
        while count < 2 * len(self.waypoints) and len(self.waypoints) > 2:  # Why 2x nr. waypoints in route?
            wp_idx = random.randint(1, len(self.waypoints) - 2)
            new_edge = Edge(self.edges[wp_idx - 1].v, self.edges[wp_idx].w, self.edges[wp_idx - 1].speed)

            # Check if new edge crosses polygon
            if not new_edge.crosses_polygon(polygons):
                self.edges[wp_idx - 1] = new_edge
                del self.waypoints[wp_idx]
                del self.edges[wp_idx]
                delete = True
                break
            count += 1
        if delete:
            self.history.append('Deletion idx {}'.format(wp_idx))
        else:
            print('No waypoint deleted.')

        # Check edges
        for e, edge in enumerate(self.edges[:-1]):
            if edge.w != self.edges[e + 1].v:
                print('Error: edges are not linked')

    def move_random_waypoint(self, polygons, radius=0.1, check_polygon_crossing=True):
        while True:
            wp_idx = random.randint(1, len(self.waypoints) - 2)
            wp = self.waypoints[wp_idx]

            # Calculate x,y coordinates of new waypoint within a radius from the current waypoint
            u1 = random.uniform(0, 1)
            u2 = random.uniform(0, 1)

            # !!! sqrt(u2), since the average distance between points should be the same regardless of how far from the
            # !!! center we look
            r = radius * sqrt(u2)
            a = u1 * 2 * pi
            new_x = wp.x + r * cos(a)
            new_y = wp.y + r * sin(a)

            new_wp = Waypoint(new_x, new_y)
            wp_idx = self.waypoints.index(wp)
            if new_wp.in_polygon(polygons):
                continue
            new_edge_1 = Edge(self.waypoints[wp_idx - 1], new_wp, self.edges[wp_idx - 1].speed)
            new_edge_2 = Edge(new_wp, self.waypoints[wp_idx + 1], self.edges[wp_idx].speed)

            if check_polygon_crossing and (
                    new_edge_1.crosses_polygon(polygons) or new_edge_2.crosses_polygon(polygons)):
                continue

            self.edges[wp_idx - 1] = new_edge_1
            self.edges[wp_idx] = new_edge_2
            self.waypoints[wp_idx] = new_wp
            self.distance = sum(edge.distance for edge in self.edges)
            self.history.append('Move {0}, idx {1}'.format(new_wp, wp_idx))
            break
        # Check edges
        for e, edge in enumerate(self.edges[:-1]):
            if edge.w != self.edges[e + 1].v:
                print('Error: edges are not linked')

    # Function to carry out the mutation operator
    def mutation(self, shorelines_f):
        mutation_prob = random.random()
        if mutation_prob < 0.33:
            self.insert_waypoint(bisector_length_ratio=0.9, polygons=shorelines_f)
        elif mutation_prob < 0.667:
            self.move_random_waypoint(shorelines_f, radius=0.1)
        else:
            self.delete_random_waypoint(shorelines_f)

    def waypoint_feasible(self, wp, polygons, radius=0.01, check_polygon_crossing=True):
        count = 0
        max_count = 20
        while True:
            if wp == self.waypoints[0] or wp == self.waypoints[-1]:
                print('First or last waypoint cannot be moved')
                break
            else:
                wp_idx = self.waypoints.index(wp)
                wp1 = self.waypoints[wp_idx - 1]
                wp2 = self.waypoints[wp_idx + 1]

                pt1 = np.array([wp1.x, wp1.y])
                pt = np.array([wp.x, wp.y])
                pt2 = np.array([wp2.x, wp2.y])
                pt1_ext = pt1 + (pt - pt1) * 1.5
                pt2_ext = pt2 + (pt - pt2) * 1.5
                line1_ext = LineString(
                    [(pt1[0], pt1[1]), (pt1_ext[0], pt1_ext[1])])
                line2_ext = LineString(
                    [(pt2[0], pt2[1]), (pt2_ext[0], pt2_ext[1])])

                circle = Point(wp.x, wp.y).buffer(radius)

                vect1_ext = pt1_ext - pt1
                vect2_ext = pt2_ext - pt2
                pt1_pt2 = pt2 - pt1
                pt2_pt1 = pt1 - pt2

                side1 = sign(det([vect1_ext, pt1_pt2]))
                side2 = sign(det([vect2_ext, pt2_pt1]))

                split_circles = split(circle, line1_ext)
                for split_circle in split_circles:
                    x_c, y_c = split_circle.centroid.coords.xy
                    c_c = np.array([x_c[0], y_c[0]])
                    pt1_c = c_c - pt1
                    split_circle_side = sign(det([vect1_ext, pt1_c]))
                    if split_circle_side != side1:
                        half_circle = split_circle
                        break

                pizza_slices = split(half_circle, line2_ext)

                for pizza_slice in pizza_slices:
                    x_p, y_p = pizza_slice.centroid.coords.xy
                    p_c = np.array([x_p[0], y_p[0]])
                    pt1_p = p_c - pt1
                    split_hc_side = sign(det([vect2_ext, pt1_p]))
                    if split_hc_side != side2:
                        sample_area = pizza_slice
                        break

                # Generate random point in pizza slice area
                while True:
                    min_x, min_y, max_x, max_y = sample_area.bounds
                    while True:
                        pnt = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
                        if sample_area.contains(pnt):
                            break

                    pnt_x, pnt_y = pnt.coords.xy
                    new_wp = Waypoint(pnt_x[0], pnt_y[0])

                    # Check if new waypoint is not in polygon
                    if not new_wp.in_polygon(polygons):
                        break

                # Create new edges
                new_edge_1 = Edge(self.waypoints[wp_idx - 1], new_wp,
                                  self.edges[wp_idx - 1].speed)
                new_edge_2 = Edge(new_wp, self.waypoints[wp_idx + 1],
                                  self.edges[wp_idx].speed)

                if check_polygon_crossing and new_edge_1.crosses_polygon(polygons):
                        count += 1
                        if count > max_count:
                            print('Waypoint {} NOT changed'.format(wp_idx))
                            break
                    # elif new_edge_2.crosses_polygon(polygons):
                    #     plot_edge(new_edge_2)
                    #     print('Edge 2 x polygon')

                else:
                    self.edges[wp_idx - 1] = new_edge_1
                    self.edges[wp_idx] = new_edge_2
                    self.waypoints[wp_idx] = new_wp
                    print('------------   Waypoint {} location changed   ------------'.format(wp_idx))
                    break
        return self

    def move_waypoint(self, wp, polygons, radius=0.01, check_polygon_crossing=True):
        while True:
            if wp == self.waypoints[0] or wp == self.waypoints[-1]:
                print('First or last waypoint cannot be moved')
                break
            else:
                # Draw random variable to pick a random location within a radius from the current waypoint
                u = random.uniform(0, 1)

                # Calculate x,y coordinates of new waypoint
                new_x = wp.x + radius * cos(u * 2 * pi)
                new_y = wp.y + radius * sin(u * 2 * pi)
                new_wp = Waypoint(new_x, new_y)
                wp_idx = self.waypoints.index(wp)
                if new_wp.in_polygon(polygons):
                    # print('New waypoint {} is in polygon'.format(wp_idx))
                    continue
                new_edge_1 = Edge(self.waypoints[wp_idx - 1], new_wp,
                                              self.edges[wp_idx - 1].speed)
                new_edge_2 = Edge(new_wp, self.waypoints[wp_idx + 1],
                                          self.edges[wp_idx].speed)

                if check_polygon_crossing and (new_edge_1.crosses_polygon(polygons) or
                    new_edge_2.crosses_polygon(polygons)):
                    # print('New edge intersects polygon')
                    continue
                self.edges[wp_idx - 1] = new_edge_1
                self.edges[wp_idx] = new_edge_2
                self.waypoints[wp_idx] = new_wp
                print('------------   Waypoint {} location changed   ------------'.format(wp_idx))
                break
        return self
