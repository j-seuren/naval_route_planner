from haversine import haversine
import fiona
from shapely.geometry import shape, Point, LineString


class Waypoint:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.shapely_point = Point(x, y)

    def __str__(self):
        return "(%.2f, %.2f)" % (self.x, self.y)

    def __repr__(self):
        return "wp(%.2f, %.2f)" % (self.x, self.y)

    def in_polygon(self, polygons_shp_fp):
        multi_pol = fiona.open(polygons_shp_fp)
        multi = multi_pol.next()
        if self.shapely_point.within(shape(multi['geometry'])):
            print('{} is in polygon.'.format(self))
            return True
        else:
            return False


class Edge:
    def __init__(self, v, w, speed):
        self.v = v  # Waypoint 1
        self.w = w  # Waypoint 2
        self.speed = float(speed)  # Knots
        self.distance = 0.539957 * haversine((v.y, v.x), (w.y, w.x))  # Nautical miles
        self.shapely_line = LineString([(v.x, v.y), (w.x, w.y)])

    def __repr__(self):
        return "e({!r}, {!r})".format(self.v, self.w)

    def travel_time(self):  # Hours
        return self.distance / self.speed

    def fuel(self, fuel_rate):  # Fuel tons
        return fuel_rate * self.travel_time() / 24

    def intersect_polygon(self, polygons_shp_fp):
        multi_pol = fiona.open(polygons_shp_fp)
        multi = multi_pol.next()
        if self.shapely_line.intersects(shape(multi['geometry'])):
            print('{} intersects polygon.'.format(self))
            return True
        else:
            return False


class Route:
    def __init__(self, edges):
        self.edges = edges
        self.distance = sum(edge.distance for edge in edges)

    def travel_time(self):  # Hours
        return sum(edge.travel_time() for edge in self.edges)

    def fuel(self, fuel_rate):  # Fuel tons
        return sum(edge.fuel(fuel_rate) for edge in self.edges)
