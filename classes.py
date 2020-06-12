from haversine import haversine
from shapely.geometry import Polygon, Point, LineString
from math import cos, sin, pi, sqrt
import pandas as pd
import numpy as np
import random


class Vessel:
    def __init__(self, name):
        table = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name=name)
        self.name = name
        self.speeds = table['Speed']
        self.fuel_rates = {speed: table['Fuel'][idx] for idx, speed in enumerate(table['Speed'])}


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
        self.travel_time = self.miles / self.speed  # Hours

    def __repr__(self):
        return "e({!r}, {!r})".format(self.v, self.w)

    def __eq__(self, edge):
        return edge and self.v == edge.v and self.w == edge.w and self.speed == edge.speed

    def __ne__(self, edge):
        return not self.__eq__(edge)

    def fuel_consumption(self, vessel):  # Fuel tons
        fuel_rate = vessel.fuel_rates[self.speed]
        return fuel_rate * self.travel_time / 24

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

    def __eq__(self, route):  # Gevaarlijke check?
        return route and self.distance == route.miles and len(self.waypoints) == len(route.waypoints)

    def __ne__(self, route):  # Gevaarlijke check?
        return not self.__eq__(route)

    def travel_time(self):  # Hours
        return sum(edge.travel_time() for edge in self.edges)

    def fuel(self, vessel):  # Fuel tons
        return sum(edge.fuel(vessel) for edge in self.edges)

    def mutate(self, rtree_idx, polygons, swaps, vessel, max_distance):
        # total = sum(weights.values())
        # prob = [v / total for v in weights.values()]
        swap = np.random.choice(swaps)  # , p=prob)

        if swap == 'insert':
            self.insert_waypoint(1, rtree_idx, polygons)
        elif swap == 'move':
            wp = random.choice(self.waypoints[1:-1])
            self.move_waypoint(wp, rtree_idx, polygons)
        elif swap == 'delete':
            self.delete_random_waypoint(rtree_idx, polygons, max_distance)
        elif swap == 'speed':
            self.change_speed(vessel)

    def insert_waypoint(self, width_ratio, rtree_idx, polygons, edge=False):
        if not edge:
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
        half_width = 1 / 2 * width_ratio * sqrt((y_w - y_v) ** 2 + (x_w - x_v) ** 2)

        # Calculate outer points of polygon
        if slope == 'inf':
            y_a = y_b = y_center
            x_a = x_center + half_width
            x_b = x_center - half_width
        elif slope == 0:
            x_a = x_b = x_center
            y_a = y_center + half_width
            y_b = y_center + half_width
        else:
            x_a = x_center + sqrt(half_width ** 2 / (1 + 1 / slope ** 2))
            x_b = x_center - sqrt(half_width ** 2 / (1 + 1 / slope ** 2))
            y_a = -(1 / slope) * (x_a - x_center) + y_center
            y_b = -(1 / slope) * (x_b - x_center) + y_center

        origin = np.array([x_v, y_v])
        v1, v2 = np.array([x_a - x_v, y_a - y_v]), np.array([x_b - x_v, y_b - y_v])
        while True:
            u1, u2 = random.uniform(0, 1), random.uniform(0, 1)

            quad_pt = np.add(u1 * v1, u2 * v2)  # Random point in quadrilateral
            quad_pt = np.add(quad_pt, origin)

            new_waypoint = Waypoint(quad_pt[0], quad_pt[1])
            new_edge_1 = Edge(edge.v, new_waypoint, edge.speed)
            new_edge_2 = Edge(new_waypoint, edge.w, edge.speed)
            if new_edge_1.x_geometry(rtree_idx, polygons) or new_edge_2.x_geometry(rtree_idx, polygons):
                continue

            # Update Route attributes
            # Insert waypoint
            waypoint_idx = self.waypoints.index(edge.w)
            self.waypoints.insert(waypoint_idx, new_waypoint)

            # Remove old edge, insert new edges
            edge_idx = self.edges.index(edge)
            del self.edges[edge_idx]
            self.edges[edge_idx:edge_idx] = [new_edge_1, new_edge_2]

            # Recompute distance
            self.distance = sum(edge.miles for edge in self.edges)
            return

    def delete_random_waypoint(self, rtree_idx, polygons, max_distance):
        waypoints = self.waypoints[1:-1]
        random.shuffle(waypoints)
        while waypoints:
            # Pop waypoint from shuffled list and get its index
            wp = waypoints.pop()
            wp_idx = self.waypoints.index(wp)

            # Create new edge
            new_edge = Edge(self.edges[wp_idx - 1].v, self.edges[wp_idx].w, self.edges[wp_idx - 1].speed)

            # Check if edge is greater than max distance or intersects a polygon
            if new_edge.miles > max_distance or new_edge.x_geometry(rtree_idx, polygons):
                continue

            # Update Route attributes
            # Delete waypoint
            del self.waypoints[wp_idx]

            # Change first edge, remove second edge
            self.edges[wp_idx - 1] = new_edge
            del self.edges[wp_idx]

            # Recompute distance
            self.distance = sum(edge.miles for edge in self.edges)
            return
        print("No waypoint deleted")

    def change_speed(self, vessel):
        n_edges = random.randint(1, len(self.edges) - 1)
        first = random.randrange(len(self.edges) - n_edges)
        new_speed = random.choice([speed for speed in vessel.speeds])
        for edge in self.edges[first:first + n_edges]:
            edge.speed = float(new_speed)

    def move_waypoint(self, wp, rtree_idx, polygons, radius=0.01):
        assert wp != self.waypoints[0] and wp != self.waypoints[-1], 'First or last waypoint cannot be moved'
        wp_idx = self.waypoints.index(wp)
        while True:
            # Pick a random location within a radius from the current waypoint
            u1, u2 = random.uniform(0, 1), random.uniform(0, 1)
            r = radius * sqrt(u2)  # Square root for a uniform probability of choosing a point in the circle
            a = u1 * 2 * pi
            x = wp.lon + r * cos(a)
            y = wp.lat + r * sin(a)

            # Create new waypoint and adjacent edges
            new_wp = Waypoint(x, y)
            new_edge_1 = Edge(self.waypoints[wp_idx-1], new_wp, self.edges[wp_idx-1].speed)
            new_edge_2 = Edge(new_wp, self.waypoints[wp_idx+1], self.edges[wp_idx].speed)

            # Check if waypoint and edges do not intersect a polygon
            if new_edge_1.x_geometry(rtree_idx, polygons) or new_edge_2.x_geometry(rtree_idx, polygons):
                continue

            # Update Route attributes
            # Substitute waypoint
            self.waypoints[wp_idx] = new_wp

            # Substitute edges
            self.edges[wp_idx - 1] = new_edge_1
            self.edges[wp_idx] = new_edge_2

            # Recompute distance
            self.distance = sum(edge.miles for edge in self.edges)
            return
