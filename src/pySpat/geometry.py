import numpy as np
from shapely.geometry import Polygon


def random_window(bounds):
    """
    Generate a random observation window within given bounds.

    Parameters:
    ----------
    bounds : tuple
        A tuple of coordinates representing the bounding box (minx, miny, maxx, maxy).

    Returns:
    -------
    Shapely Polygon object
        A random observation window.
    """
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    random_x = np.random.uniform(minx, maxx)
    random_y = np.random.uniform(miny, maxy)

    # Create a square window (for simplicity)
    window = Polygon([(random_x, random_y),
                      (random_x + width * 0.1, random_y),
                      (random_x + width * 0.1, random_y + height * 0.1),
                      (random_x, random_y + height * 0.1)])

    return window



def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    ----------
    point1 : Shapely Point object
        First point.
    point2 : Shapely Point object
        Second point.

    Returns:
    -------
    float
        Distance between the two points.
    """
    return point1.distance(point2)



def check_within_window(point, window):
    """
    Check if a point is within the given observation window.

    Parameters:
    ----------
    point : Shapely Point object
        The point to check.
    window : Shapely Polygon object
        The observation window.

    Returns:
    -------
    bool
        True if the point is within the window, False otherwise.
    """
    return window.contains(point)



def check_overlap(window1, window2):
    """
    Check if two observation windows overlap.

    Parameters:
    ----------
    window1 : Shapely Polygon object
        First observation window.
    window2 : Shapely Polygon object
        Second observation window.

    Returns:
    -------
    bool
        True if the windows overlap, False otherwise.
    """
    return window1.intersects(window2)



def calculate_area(polygon):
    """
    Calculate the area of a polygon.

    Parameters:
    ----------
    polygon : Shapely Polygon object
        The polygon for which to calculate the area.

    Returns:
    -------
    float
        Area of the polygon.
    """
    return polygon.area



def calculate_centroid(polygon):
    """
    Calculate the centroid of a polygon.

    Parameters:
    ----------
    polygon : Shapely Polygon object
        The polygon for which to calculate the centroid.

    Returns:
    -------
    Shapely Point object
        The centroid of the polygon.
    """
    return polygon.centroid



def get_bounding_box(polygon):
    """
    Get the bounding box of a polygon.

    Parameters:
    ----------
    polygon : Shapely Polygon object
        The polygon for which to get the bounding box.

    Returns:
    -------
    tuple
        A tuple of coordinates representing the bounding box (minx, miny, maxx, maxy).
    """
    return polygon.bounds



def nearest_neighbor(point, points_list):
    """
    Find the nearest neighbor point from a list of points.

    Parameters:
    ----------
    point : Shapely Point object
        The reference point.
    points_list : list of Shapely Point objects
        List of points to compare against.

    Returns:
    -------
    Shapely Point object
        The nearest neighbor point.
    """
    min_distance = float('inf')
    nearest = None
    for p in points_list:
        dist = point.distance(p)
        if dist < min_distance:
            min_distance = dist
            nearest = p
    return nearest



def point_in_polygon(point, polygon):
    """
    Check if a point is inside a given polygon.

    Parameters:
    ----------
    point : Shapely Point object
        The point to check.
    polygon : Shapely Polygon object
        The polygon in which to check.

    Returns:
    -------
    bool
        True if the point is inside the polygon, False otherwise.
    """
    return polygon.contains(point)



def calculate_distance_to_boundary(point, polygon):
    """
    Calculate the distance from a point to the nearest boundary of a polygon.

    Parameters:
    ----------
    point : Shapely Point object
        The point.
    polygon : Shapely Polygon object
        The polygon.

    Returns:
    -------
    float
        The distance to the nearest boundary.
    """
    return polygon.exterior.distance(point)
