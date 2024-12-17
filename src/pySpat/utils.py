import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon



def random_window(width, height):
    """
    Generate a random rectangular observation window.

    Parameters:
    ----------
    width : float
        Width of the window.
    height : float
        Height of the window.

    Returns:
    -------
    Shapely Polygon object
        A polygon representing the observation window.
    """
    min_x, min_y = 0, 0
    max_x, max_y = width, height
    return Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])



def from_dataframe(dataframe):
    """
    Convert a DataFrame of coordinates into a list of Shapely Points.

    Parameters:
    ----------
    dataframe : pandas DataFrame
        A dataframe containing 'x' and 'y' columns representing coordinates of points.

    Returns:
    -------
    list of Shapely Point objects
        A list of point objects.
    """
    return [Point(row['x'], row['y']) for index, row in dataframe.iterrows()]



def to_dataframe(points):
    """
    Convert a list of Shapely Points into a DataFrame.

    Parameters:
    ----------
    points : list of Shapely Point objects
        A list of point objects.

    Returns:
    -------
    pandas DataFrame
        A dataframe containing 'x' and 'y' columns.
    """
    data = [{'x': point.x, 'y': point.y} for point in points]
    return pd.DataFrame(data)



def distance_matrix(points):
    """
    Compute the distance matrix for a set of points.

    Parameters:
    ----------
    points : list of Shapely Point objects
        The list of points.

    Returns:
    -------
    numpy.ndarray
        A square matrix containing distances between points.
    """
    n = len(points)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = points[i].distance(points[j])
            matrix[i, j] = distance
            matrix[j, i] = distance

    return matrix
