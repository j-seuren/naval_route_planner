import random
from math import sqrt
from shapely.geometry import Polygon, Point
from route import Waypoint, Edge, Route


def insertion(route, probability, d_factor, shore_polygons):
    if random.uniform(0, 1) < probability:
        # Pick random edge and adjacent waypoints
        edge = route.edges[random.randint(0, len(route.edges) - 1)]
        wp1 = edge.v
        wp2 = edge.w

        x1, y1 = wp1.x, wp1.y
        x2, y2 = wp2.x, wp2.y

        # Edge center
        x_center = min(x1, x2) + 0.5 * abs(x2 - x1)
        y_center = min(y1, y2) + 0.5 * abs(y2 - y1)

        m = (y2 - y1) / (x2 - x1)  # Slope

        d = d_factor * sqrt((y2 - y1)**2 + (x2 - x1)**2)  # Set radius from edge center as function of edge length

        # Calculate outer points of polygon
        x3 = x_center + sqrt(d**2 / (1 + 1/m**2))
        x4 = x_center - sqrt(d**2 / (1 + 1/m**2))
        y3 = -(1/m) * (x3 - x_center) + y_center
        y4 = -(1/m) * (x4 - x_center) + y_center

        polygon = Polygon([[x1, y1], [x3, y3], [x2, y2], [x4, y4]])

        # Create point within polygon bounds and check if polygon contains point
        min_x, min_y, max_x, max_y = polygon.bounds
        count = 0
        while True:
            point = Point(random.uniform(min_x, max_x), random. uniform(min_y, max_y))
            new_waypoint = Waypoint(point.x, point.y)
            count += 1
            if new_waypoint.in_polygon(shore_polygons):
                continue
            if polygon.contains(point):
                break

        # Insert new waypoint in Route
        new_waypoint = Waypoint(point.x, point.y)
        waypoint_idx = route.waypoints.index(edge.v)
        route.waypoints[waypoint_idx+1:waypoint_idx+1] = [new_waypoint]

        # Remove old edge and insert two new edges
        edge_idx = route.edges.index(edge)
        del route.edges[edge_idx]
        new_edge_1 = Edge(edge.v, new_waypoint, edge.speed)
        new_edge_2 = Edge(new_waypoint, edge.w, edge.speed)
        route.edges[edge_idx:edge_idx] = [new_edge_1, new_edge_2]
        return route
    else:
        return route

