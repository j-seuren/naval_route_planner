from haversine import haversine
from shapely.geometry import shape, Polygon, Point, LineString
from math import atan2, cos, sin, degrees, pi, sqrt
import random


class Waypoint:
    def __init__(self, x, y, waypoint_id=-1):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return "(%.2f, %.2f)" % (self.x, self.y)

    def __repr__(self):
        return "wp(%.2f, %.2f)" % (self.x, self.y)

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
        for polygon in iter(polygons):
            shapely_line = LineString([(self.v.x, self.v.y), (self.w.x, self.w.y)])
            intersect = shapely_line.intersection(shape(polygon['geometry']))
            if intersect and intersect.geom_type == 'LineString':
                print('{} intersects polygon.'.format(self))
                if return_polygon:
                    return polygon
                else:
                    return True
        return False
    # TEMPORARY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def intersec_polygons(self, polygons, return_polygon=False):
        for polygon in iter(polygons):
            shapely_line = LineString([(self.v.x, self.v.y), (self.w.x, self.w.y)])
            intersect = shapely_line.intersection(polygon)
            if intersect and intersect.geom_type == 'LineString':
                print('{} intersects polygon.'.format(self))
                if return_polygon:
                    return polygon
                else:
                    return True
        return False
    # TEMPORARY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def create_edges(waypoints, initial_speed):
    edges = []
    v = waypoints[0]
    for w in waypoints[1:]:
        edge = Edge(v, w, initial_speed)
        v = w
        edges.append(edge)
    return edges


class Route:
    def __init__(self, waypoints, initial_speed):
        self.edges = create_edges(waypoints, initial_speed)
        self.waypoints = waypoints
        self.distance = sum(edge.distance for edge in self.edges)

    def travel_time(self):  # Hours
        return sum(edge.travel_time() for edge in self.edges)

    def fuel(self, fuel_rate):  # Fuel tons
        return sum(edge.fuel(fuel_rate) for edge in self.edges)

    def move_waypoint(self, waypoint, shoreline_polygons, radius=0.01):
        # Draw random variable to pick a random location within a radius from the current waypoint
        while True:
            u = random.uniform(0, 1)
            # Calculate x,y coordinates of new waypoint
            new_x = waypoint.x + radius * cos(u * 2 * pi)
            new_y = waypoint.y + radius * sin(u * 2 * pi)
            waypoint_idx = self.waypoints.index(waypoint)
            new_waypoint = Waypoint(new_x, new_y)
            if new_waypoint.in_polygon(shoreline_polygons):
                print('Wp in polygon')
                continue
            self.waypoints[waypoint_idx] = new_waypoint
            if waypoint == self.waypoints[0]:
                self.edges[0] = Edge(new_waypoint, self.waypoints[1], self.edges[0].speed)
                if self.edges[0].crosses_polygon(shoreline_polygons):
                    print('New edge intersects polygon')
                    continue
                print('Waypoint {} location changed.'.format(waypoint_idx))
                break
            elif waypoint == self.waypoints[-1]:
                self.edges[-1] = Edge(self.waypoints[-2], new_waypoint, self.edges[-1].speed)
                if self.edges[-1].crosses_polygon(shoreline_polygons):
                    print('New edge intersects polygon')
                    continue
                print('Waypoint {} location changed.'.format(waypoint_idx))
                break
            else:
                self.edges[waypoint_idx - 1] = Edge(self.waypoints[waypoint_idx - 1], new_waypoint,
                                                   self.edges[waypoint_idx - 1].speed)
                self.edges[waypoint_idx] = Edge(new_waypoint, self.waypoints[waypoint_idx + 1],
                                               self.edges[waypoint_idx].speed)
                if self.edges[waypoint_idx - 1].crosses_polygon(shoreline_polygons) or \
                        self.edges[waypoint_idx].crosses_polygon(shoreline_polygons):
                    print('New edge intersects polygon')
                    continue
                print('Waypoint {} location changed.'.format(waypoint_idx))
                break
        return self

    def insert_waypoint(self, bisector_length_ratio, shore_polygons):
        # Pick random edge and adjacent waypoints
        edge = self.edges[random.randint(0, len(self.edges) - 1)]
        print('Selected edge is ', edge)
        x_v, y_v = edge.v.x, edge.v.y
        x_w, y_w = edge.w.x, edge.w.y

        # Edge center
        x_center = x_v + 0.5 * (x_w - x_v)
        y_center = y_v + 0.5 * (y_w - y_v)

        slope = (y_w - y_v) / (x_w - x_v)

        # Set half length of the edge's perpendicular bisector
        half_length_bisector = 1/2 * bisector_length_ratio * sqrt((y_w - y_v) ** 2 + (x_w - x_v) ** 2)

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
            if new_waypoint.in_polygon(shore_polygons):
                continue
            new_edge_1 = Edge(edge.v, new_waypoint, edge.speed)
            new_edge_2 = Edge(new_waypoint, edge.w, edge.speed)
            if new_edge_1.crosses_polygon(shore_polygons) or new_edge_2.crosses_polygon(shore_polygons):
                continue
            break

        # Insert new waypoint in route
        waypoint_idx = self.waypoints.index(edge.w)
        self.waypoints[waypoint_idx:waypoint_idx] = [new_waypoint]

        # Remove old edge and insert two new edges
        edge_idx = self.edges.index(edge)
        del self.edges[edge_idx]
        self.edges[edge_idx:edge_idx] = [new_edge_1, new_edge_2]

        return self

    def delete_waypoint(self, shore_polygons):
        count = 0
        delete = False
        while count < len(2 * self.waypoints):  # Why 2x nr. waypoints in route?
            waypoint_idx = random.randint(1, len(self.waypoints) - 2)

            # Check if new edge crosses polygon
            if not Edge(self.edges[waypoint_idx].v, self.edges[waypoint_idx+1].w,
                        self.edges[waypoint_idx].speed).crosses_polygon(shore_polygons):
                self.edges[waypoint_idx] = Edge(self.edges[waypoint_idx].v, self.edges[waypoint_idx+1].w,
                                                self.edges[waypoint_idx].speed)
                del self.waypoints[waypoint_idx]
                del self.edges[waypoint_idx + 1]
                delete = True
                break
            count += 1
        if delete:
            print('Deleted waypoint {}'.format(waypoint_idx))
        else:
            print('No waypoint deleted.')
        return self
