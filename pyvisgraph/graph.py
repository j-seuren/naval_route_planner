from collections import defaultdict
from math import sqrt

def line_intersect(line_a_point_a_long, line_a_point_a_lat, line_a_point_b_long, line_a_point_b_lat,
                  line_b_point_a_long, line_b_point_a_lat, line_b_point_b_long, line_b_point_b_lat):
    """
    Quick checks if lines can never intersect
    If one of the lines (arc between two noonreports) crosses the International Date Line (IDL),
    then return to avoid wrong calculations. The world is a globe...
    """
    if min(line_a_point_a_long, line_a_point_b_long) > max(line_b_point_a_long, line_b_point_b_long):
        # Line A above line B
        return 0
    elif min(line_b_point_a_long, line_b_point_b_long) > max(line_a_point_a_long, line_a_point_b_long):
        # Line B above line A
        return 0
    elif min(line_a_point_a_lat, line_a_point_b_lat) > max(line_b_point_a_lat, line_b_point_b_lat):
        # Line A right of line B
        return 0
    elif min(line_b_point_a_lat, line_b_point_b_lat) > max(line_a_point_a_lat, line_a_point_b_lat):
        # Line B right of line A
        return 0
    elif line_a_point_a_long*line_a_point_b_long < -1000:
        # Line A crosses the IDL
        return 0
    elif line_b_point_a_long*line_b_point_b_long < -1000:
        # Line B crosses the IDL
        return 0
    else:
        """Calculate delta y and delta x for both lines"""
        line_a_long_delta = line_a_point_b_long - line_a_point_a_long
        line_a_lat_delta = line_a_point_b_lat - line_a_point_a_lat

        line_b_long_delta = line_b_point_b_long - line_b_point_a_long
        line_b_lat_delta = line_b_point_b_lat - line_b_point_a_lat

        """Calculate both lines length using pythagoras"""
        line_a_length = sqrt(line_a_long_delta**2 + line_a_lat_delta**2)
        line_b_length = sqrt(line_b_long_delta ** 2 + line_b_lat_delta ** 2)

        """Calculate direction of both lines (between 0 and 2pi"""
        if line_a_length > 0:
            line_a_direction_sin = line_a_lat_delta / line_a_length
            line_a_direction_cos = line_a_long_delta / line_a_length
        else:
            line_a_direction_sin = 0
            line_a_direction_cos = 1
        if line_b_length > 0:
            line_b_direction_sin = line_b_lat_delta / line_b_length
            line_b_direction_cos = line_b_long_delta / line_b_length
        else:
            line_b_direction_sin = 0
            line_b_direction_cos = 1

        """Calculate if and where the two lines cross"""
        if line_a_direction_sin*line_b_direction_cos != line_b_direction_sin * line_a_direction_cos \
                and line_a_length > 0 and line_b_length > 0:
            intersect_a_length = ((line_b_point_a_lat - line_a_point_a_lat)*line_b_direction_cos
                                  - (line_b_point_a_long - line_a_point_a_long)*line_b_direction_sin) \
                                / (line_a_direction_sin*line_b_direction_cos
                                     - line_b_direction_sin*line_a_direction_cos)
            intersect_b_length = ((line_b_point_a_long - line_a_point_a_long) * line_a_direction_sin
                                  - (line_b_point_a_lat - line_a_point_a_lat) * line_a_direction_cos) \
                                 / (line_b_direction_sin * line_a_direction_cos
                                    - line_a_direction_sin * line_b_direction_cos)
            if 0 <= intersect_a_length <= line_a_length and 0 <= intersect_b_length <= line_b_length:
                return 1
            else:
                return 0
        else:
            return 0

class Point(object):
    __slots__ = ('long', 'lat', 'polygon_id')

    def __init__(self, x, y, polygon_id=-1):
        self.long = float(x)
        self.lat = float(y)
        self.polygon_id = polygon_id

    def __eq__(self, point):
        return point and self.long == point.x and self.lat == point.y

    def __ne__(self, point):
        return not self.__eq__(point)

    def __lt__(self, point):
        """ This is only needed for shortest path calculations where heapq is
            used. When there are two points of equal distance, heapq will
            instead evaluate the Points, which doesnt work in Python 3 and
            throw a TypeError."""
        return hash(self) < hash(point)

    def __str__(self):
        return "(%.2f, %.2f)" % (self.long, self.lat)

    def __hash__(self):
        return self.long.__hash__() ^ self.lat.__hash__()

    def __repr__(self):
        return "Point(%.2f, %.2f)" % (self.long, self.lat)

    def in_seca(self, seca_areas):
        count = 0
        if self.lat < seca_areas[['latitude']].idxmin():
            return False
        elif self.lat > seca_areas[['latitude']].idxmax():
            return False
        elif self.long < seca_areas[['longitude']].idxmin():
            return False
        elif self.long > seca_areas[['longitude']].idxmax():
            return False
        else:
            first_index = min(seca_areas.index)
            last_index = max(seca_areas.index)
            previous_index = first_index

            for seca_index, seca_area in seca_areas.iterrows():

                count += line_intersect(self.long, self.lat, 0, 90, seca_areas.iloc[seca_index][['longitude']], seca_areas.iloc[seca_index][['latitude']]
                                        , seca_areas.iloc[previous_index][['longitude']], seca_areas.iloc[previous_index][['latitude']])
                previous_index = seca_index

            count += line_intersect(self.long, self.lat, 0, 90, seca_areas.iloc[last_index][['longitude']], seca_areas.iloc[last_index][['latitude']]
                                    , seca_areas.iloc[first_index][['longitude']], seca_areas.iloc[first_index][['latitude']])

            if count % 2 == 1:
                return True
            else:
                return False


class Edge(object):
    __slots__ = ('p1', 'p2')

    def __init__(self, point1, point2):
        self.p1 = point1
        self.p2 = point2

    def get_adjacent(self, point):
        if point == self.p1:
            return self.p2
        return self.p1

    def __contains__(self, point):
        return self.p1 == point or self.p2 == point

    def __eq__(self, edge):
        if self.p1 == edge.p1 and self.p2 == edge.p2:
            return True
        if self.p1 == edge.p2 and self.p2 == edge.p1:
            return True
        return False

    def __ne__(self, edge):
        return not self.__eq__(edge)

    def __str__(self):
        return "({}, {})".format(self.p1, self.p2)

    def __repr__(self):
        return "Edge({!r}, {!r})".format(self.p1, self.p2)

    def __hash__(self):
        return self.p1.__hash__() ^ self.p2.__hash__()


class Graph(object):
    """
    A Graph is represented by a dict where the keys are Points in the Graph
    and the dict values are sets containing Edges incident on each Point.
    A separate set *edges* contains all Edges in the graph.
    The input must be a list of polygons, where each polygon is a list of
    in-order (clockwise or counter clockwise) Points. If only one polygon,
    it must still be a list in a list, i.e. [[Point(0,0), Point(2,0),
    Point(2,1)]].
    *polygons* dictionary: key is a integer polygon ID and values are the
    edges that make up the polygon. Note only polygons with 3 or more Points
    will be classified as a polygon. Non-polygons like just one Point will be
    given a polygon ID of -1 and not maintained in the dict.
    """

    def __init__(self, polygons):
        self.graph = defaultdict(set)
        self.edges = set()
        self.polygons = defaultdict(set)
        pid = 0
        for polygon in polygons:
            if polygon[0] == polygon[-1] and len(polygon) > 1:
                polygon.pop()
            for i, point in enumerate(polygon):
                sibling_point = polygon[(i + 1) % len(polygon)]
                edge = Edge(point, sibling_point)
                if len(polygon) > 2:
                    point.polygon_id = pid
                    sibling_point.polygon_id = pid
                    self.polygons[pid].add(edge)
                self.add_edge(edge)
            if len(polygon) > 2:
                pid += 1

    def get_adjacent_points(self, point):
        return [edge.get_adjacent(point) for edge in self[point]]

    def get_points(self):
        return list(self.graph)

    def get_edges(self):
        return self.edges

    def add_edge(self, edge):
        self.graph[edge.p1].add(edge)
        self.graph[edge.p2].add(edge)
        self.edges.add(edge)

    def __contains__(self, item):
        if isinstance(item, Point):
            return item in self.graph
        if isinstance(item, Edge):
            return item in self.edges
        return False

    def __getitem__(self, point):
        if point in self.graph:
            return self.graph[point]
        return set()

    def __str__(self):
        res = ""
        for point in self.graph:
            res += "\n" + str(point) + ": "
            for edge in self.graph[point]:
                res += str(edge)
        return res

    def __repr__(self):
        return self.__str__()