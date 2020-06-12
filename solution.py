from haversine import haversine
from shapely.geometry import Polygon, Point, LineString
from math import cos, sin, pi, sqrt
import random
import numpy as np


class Waypoint:
    def __init__(self, lon, lat):
        self.lon = float(lon)
        self.lat = float(lat)
        self.xyz = (cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))

    def __str__(self):
        return "(%.2f, %.2f)" % (self.lon, self.lat)

    def __repr__(self):
        return "wp(%.2f, %.2f)" % (self.lon, self.lat)

    def __eq__(self, wp):
        return wp and self.lon == wp.lon and self.lat == wp.lat

    def __ne__(self, wp):
        return not self.__eq__(wp)

    def x_geometry(self, rtree_idx, geometries):
        waypoint_coordinates = (self.lon, self.lat)

        # Returns the geometry indices of the minimum bounding rectangles that intersect the bounding box of waypoint
        mbr_intersections = rtree_idx.intersection(waypoint_coordinates)
        if mbr_intersections:  # Create LineString if there is at least one minimum bounding rectangle intersection
            shapely_point = Point(self.lon, self.lat)
            for i in mbr_intersections:
                if geometries[i].intersects(shapely_point):
                    return geometries[i]
        return False


class Edge:
    def __init__(self, v, w, speed):
        self.v = v  # Waypoint 1
        self.w = w  # Waypoint 2
        self.speed = float(speed)  # Knots
        self.miles = haversine((self.v.lat, self.v.lon), (self.w.lat, self.w.lon), unit='nmi')  # Nautical miles

    def __repr__(self):
        return "e({!r}, {!r})".format(self.v, self.w)

    def __eq__(self, edge):
        return edge and self.v == edge.v and self.w == edge.w and self.speed == edge.speed

    def __ne__(self, edge):
        return not self.__eq__(edge)

    def travel_time(self):  # Hours
        return self.miles / self.speed

    def fuel(self, vessel):  # Fuel tons
        fuel_rate = vessel.fuel_rates[self.speed]
        return fuel_rate * self.travel_time() / 24

    def x_geometry(self, rtree_idx, geometries):
        line_bounds = (min(self.v.lon, self.w.lon), min(self.v.lat, self.w.lat),
                       max(self.v.lon, self.w.lon), max(self.v.lat, self.w.lat))

        # Returns the geometry indices of the minimum bounding rectangles that intersect the bounding box of edge
        mbr_intersections = rtree_idx.intersection(line_bounds)
        if mbr_intersections:  # Create LineString if there is at least one minimum bounding rectangle intersection
            shapely_line = LineString([(self.v.lon, self.v.lat), (self.w.lon, self.w.lat)])
            for i in mbr_intersections:
                if geometries[i].intersects(shapely_line):
                    return geometries[i]
        return False


class Route:
    def __init__(self, edges):
        self.edges = edges
        self.waypoints = [edge.v for edge in edges] + [edges[-1].w]
        self.distance = sum(edge.miles for edge in self.edges)
        self.history = []

    def __eq__(self, route):  # Gevaarlijke check?
        return route and self.distance == route.miles and len(self.waypoints) == len(route.waypoints)

    def __ne__(self, route):  # Gevaarlijke check?
        return not self.__eq__(route)

    def travel_time(self):  # Hours
        return sum(edge.travel_time() for edge in self.edges)

    def fuel(self, vessel):  # Fuel tons
        return sum(edge.fuel(vessel) for edge in self.edges)

    def insert_waypoint(self, bisector_length_ratio, rtree_idx, polygons, edge=False):
        if not edge:  # Pick random edge and adjacent waypoints
            edge = random.choice(self.edges)
        x_v, y_v = edge.v.lon, edge.v.lat
        x_w, y_w = edge.w.lon, edge.w.lat

        # Edge center
        x_center = (x_v + x_w) / 2
        y_center = (y_v + y_w) / 2

        try:
            slope = (y_w - y_v) / (x_w - x_v)
        except ZeroDivisionError:
            print('Division by zero')
            slope = 'inf'

        # Get half length of the edge's perpendicular bisector
        half_length_bisector = 1 / 2 * bisector_length_ratio * sqrt((y_w - y_v) ** 2 + (x_w - x_v) ** 2)

        # Calculate outer points of polygon
        if slope == 'inf':
            y_a = y_b = y_center
            x_a = x_center + half_length_bisector
            x_b = x_center - half_length_bisector
        elif slope == 0:
            x_a = x_b = x_center
            y_a = y_center + half_length_bisector
            y_b = y_center + half_length_bisector
        else:
            x_a = x_center + sqrt(half_length_bisector ** 2 / (1 + 1 / slope ** 2))
            x_b = x_center - sqrt(half_length_bisector ** 2 / (1 + 1 / slope ** 2))
            y_a = -(1 / slope) * (x_a - x_center) + y_center
            y_b = -(1 / slope) * (x_b - x_center) + y_center

        polygon = Polygon([[x_v, y_v], [x_a, y_a], [x_w, y_w], [x_b, y_b]])

        # Create point within polygon bounds and check if polygon contains point
        min_x, min_y, max_x, max_y = polygon.bounds
        new_waypoint = new_edge_1 = new_edge_2 = []
        cnt = 0
        while True:
            cnt += 1
            if cnt > 1000:
                print('No wp inserted')
                return
            point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if not polygon.contains(point):
                continue
            new_waypoint = Waypoint(point.x, point.y)
            if new_waypoint.x_geometry(rtree_idx, polygons):
                continue
            new_edge_1 = Edge(edge.v, new_waypoint, edge.speed)
            new_edge_2 = Edge(new_waypoint, edge.w, edge.speed)
            if new_edge_1.x_geometry(rtree_idx, polygons) or new_edge_2.x_geometry(rtree_idx, polygons):
                continue
            break

        # Insert new waypoint in route
        waypoint_idx = self.waypoints.index(edge.w)
        self.waypoints.insert(waypoint_idx, new_waypoint)

        # Remove old edge and insert two new edges
        edge_idx = self.edges.index(edge)
        del self.edges[edge_idx]
        self.edges[edge_idx:edge_idx] = [new_edge_1, new_edge_2]
        self.distance = sum(edge.miles for edge in self.edges)
        self.history.append('Insertion {0}, idx {1}'.format(new_waypoint, waypoint_idx))
        # for e, edge in enumerate(self.edges[:-1]):
        #     if edge.w != self.edges[e + 1].v:
        #         print('Error: edges are not linked')

    def delete_random_waypoint(self, rtree_idx, polygons, max_wp_distance_f):
        count = 0
        delete = False
        while count < 2 * len(self.waypoints) and len(self.waypoints) > 2:  # Why 2x nr. waypoints in route?
            count += 1
            wp_idx = random.randrange(1, len(self.waypoints) - 1)
            new_edge = Edge(self.edges[wp_idx - 1].v, self.edges[wp_idx].w, self.edges[wp_idx - 1].speed)
            if new_edge.miles > max_wp_distance_f:
                continue

            # Check if new edge crosses polygon
            if not new_edge.x_geometry(rtree_idx, polygons):
                self.edges[wp_idx - 1] = new_edge
                del self.waypoints[wp_idx]
                del self.edges[wp_idx]
                delete = True
                break
        if delete:
            self.distance = sum(edge.miles for edge in self.edges)
            self.history.append('Deletion idx {}'.format(wp_idx))
        # else:
        #     print('No waypoint deleted.')

        # Check edges
        # for e, edge in enumerate(self.edges[:-1]):
        #     if edge.w != self.edges[e + 1].v:
        #         print('Error: edges are not linked')

        return delete

    def mutation(self, rtree_idx, shorelines_f, weights_f, vessel, no_improvement_count_f, max_wp_distance_f, labda=0.9):
        # Check if a waypoint can be deleted
        if len(self.edges) <= 2:
            weights_without_delete = {i:weights_f[i] for i in weights_f if i!='delete'}
            total = sum(weights_without_delete.values())
            prob = [v / total for v in weights_without_delete.values()]
            swap = np.random.choice(list(weights_without_delete.keys()))  #, p=prob)
        else:
            total = sum(weights_f.values())
            prob = [v / total for v in weights_f.values()]
            swap = np.random.choice(list(weights_f.keys()))  #, p=prob)
        distance_prev = self.distance
        fuel_prev = self.fuel(vessel)

        if swap == 'insert':
            self.insert_waypoint(bisector_length_ratio=2, rtree_idx=rtree_idx, polygons=shorelines_f)
        elif swap == 'move':
            radius = 0.1

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
                new_x = wp.lon + r * cos(a)
                new_y = wp.lat + r * sin(a)

                new_wp = Waypoint(new_x, new_y)
                wp_idx = self.waypoints.index(wp)
                if new_wp.x_geometry(rtree_idx, shorelines_f):
                    continue
                new_edge_1 = Edge(self.waypoints[wp_idx - 1], new_wp, self.edges[wp_idx - 1].speed)
                new_edge_2 = Edge(new_wp, self.waypoints[wp_idx + 1], self.edges[wp_idx].speed)

                if new_edge_1.x_geometry(rtree_idx, shorelines_f) or new_edge_2.x_geometry(rtree_idx, shorelines_f):
                    continue

                self.edges[wp_idx - 1] = new_edge_1
                self.edges[wp_idx] = new_edge_2
                self.waypoints[wp_idx] = new_wp
                self.distance = sum(edge.miles for edge in self.edges)
                self.history.append('Move {0}, idx {1}'.format(new_wp, wp_idx))
                break
            # # Check edges
            # for e, edge in enumerate(self.edges[:-1]):
            #     if edge.w != self.edges[e + 1].v:
            #         print('Error: edges are not linked')
        elif swap == 'delete':
            count = 0
            delete = False
            while count < 2 * len(self.waypoints) and len(self.waypoints) > 2:  # Why 2x nr. waypoints in route?
                count += 1
                wp_idx = random.randrange(1, len(self.waypoints) - 1)
                new_edge = Edge(self.edges[wp_idx - 1].v, self.edges[wp_idx].w, self.edges[wp_idx - 1].speed)
                if new_edge.miles > max_wp_distance_f:
                    continue

                # Check if new edge crosses polygon
                if not new_edge.x_geometry(rtree_idx, shorelines_f):
                    self.edges[wp_idx - 1] = new_edge
                    del self.waypoints[wp_idx]
                    del self.edges[wp_idx]
                    delete = True
                    break
            if delete:
                self.distance = sum(edge.miles for edge in self.edges)
                self.history.append('Deletion idx {}'.format(wp_idx))
            else:
                swap = 'insert'
                self.insert_waypoint(2, rtree_idx, shorelines_f)
        elif swap == 'speed':
            edge = random.choice(self.edges)
            current_speed = edge.speed
            new_speed = random.choice([speed for speed in vessel.speeds if speed != current_speed])

            # Change edge speed
            edge.speed = float(new_speed)

        distance_nb = self.distance
        fuel_nb = self.fuel(vessel)
        score = max(0, (distance_prev - distance_nb) / distance_prev + (fuel_prev - fuel_nb) / fuel_prev)
        if score == 0:
            no_improvement_count_f += 1
        else:
            no_improvement_count_f = 0
        weights_f[swap] = labda * weights_f[swap] + (1 - labda) * 1000 * score

        weight_rounded = ['%.1f' % elem for elem in weights_f.values()]
        return weights_f, no_improvement_count_f

    def move_waypoint(self, wp, rtree_idx, polygons, radius=0.01, check_polygon_crossing=True):
        while True:
            if wp == self.waypoints[0] or wp == self.waypoints[-1]:
                print('First or last waypoint cannot be moved')
                break
            else:
                # Draw random variable to pick a random location within a radius from the current waypoint
                u = random.uniform(0, 1)

                # Calculate x,y coordinates of new waypoint
                new_x = wp.lon + radius * cos(u * 2 * pi)
                new_y = wp.lat + radius * sin(u * 2 * pi)
                new_wp = Waypoint(new_x, new_y)
                wp_idx = self.waypoints.index(wp)
                if new_wp.x_geometry(rtree_idx, polygons):
                    # print('New waypoint {} is in polygon'.format(wp_idx))
                    continue
                new_edge_1 = Edge(self.waypoints[wp_idx - 1], new_wp,
                                              self.edges[wp_idx - 1].speed)
                new_edge_2 = Edge(new_wp, self.waypoints[wp_idx + 1],
                                          self.edges[wp_idx].speed)

                if check_polygon_crossing and (new_edge_1.x_geometry(rtree_idx, polygons) or
                                               new_edge_2.x_geometry(rtree_idx, polygons)):
                    # print('New edge intersects polygon')
                    continue
                self.edges[wp_idx - 1] = new_edge_1
                self.edges[wp_idx] = new_edge_2
                self.waypoints[wp_idx] = new_wp
                print('------------   Waypoint {} location changed   ------------'.format(wp_idx))
                break
        return self
