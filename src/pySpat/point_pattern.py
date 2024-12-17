from shapely.geometry import Point, Polygon



class PointPattern:
    def __init__(self, points, window):
        """
        Represents a spatial point pattern.
        :param points: List of tuples [(x1, y1), (x2, y2), ...]
        :param window: A Shapely Polygon defining the observation area.
        """
        self.points = [Point(p) for p in points]
        self.window = window

    def n_points(self):
        """Return the number of points in the pattern."""
        return len(self.points)

    def intensity(self):
        """Calculate the intensity (points per unit area)."""
        area = self.window.area
        if area <= 0:
            raise ValueError("Window area must be positive.")
        return self.n_points() / area



class ObservationWindow:
    def __init__(self, boundary):
        """
        Represents an observation window in 2D space.
        :param boundary: A list of (x, y) tuples defining the boundary of the window,
                         or a Shapely Polygon object.
        """
        if isinstance(boundary, Polygon):
            self.boundary = boundary
        elif isinstance(boundary, list) and all(isinstance(pt, tuple) and len(pt) == 2 for pt in boundary):
            self.boundary = Polygon(boundary)
        else:
            raise ValueError("Boundary must be a Shapely Polygon or a list of (x, y) tuples.")

        if not self.boundary.is_valid:
            raise ValueError("The boundary must define a valid polygon.")

    def area(self):
        """Returns the area of the observation window."""
        return self.boundary.area

    def perimeter(self):
        """Returns the perimeter of the observation window."""
        return self.boundary.length

    def contains_point(self, point):
        """
        Checks if a given point is inside the observation window.
        :param point: A tuple (x, y) or a Shapely Point object.
        :return: True if the point is inside the window, False otherwise.
        """
        if isinstance(point, Point):
            return self.boundary.contains(point)
        elif isinstance(point, tuple) and len(point) == 2:
            return self.boundary.contains(Point(point))
        else:
            raise ValueError("Point must be a Shapely Point or a tuple (x, y).")

    def intersects(self, other_window):
        """
        Checks if this observation window intersects with another window.
        :param other_window: An ObservationWindow object.
        :return: True if the two windows intersect, False otherwise.
        """
        if isinstance(other_window, ObservationWindow):
            return self.boundary.intersects(other_window.boundary)
        else:
            raise ValueError("Other window must be an instance of ObservationWindow.")

    def intersection(self, other_window):
        """
        Computes the intersection of this window with another window.
        :param other_window: An ObservationWindow object.
        :return: A new ObservationWindow representing the intersection.
        """
        if isinstance(other_window, ObservationWindow):
            intersected_boundary = self.boundary.intersection(other_window.boundary)
            if intersected_boundary.is_empty:
                return None
            return ObservationWindow(intersected_boundary)
        else:
            raise ValueError("Other window must be an instance of ObservationWindow.")


