import numpy as np
from scipy.spatial import cKDTree, KDTree
from shapely.geometry import Point



def calculate_rmax(points, method='bounding_box', k=5):
    """
        Calculate rmax for Ripley's K-function.

        :param points: 2D NumPy array of point coordinates (shape: n x 2).
        :param method: Method to determine rmax ('bounding_box' or 'nearest_neighbour').
        :param k: Factor for nearest neighbor method.
        :return: Calculated rmax value.
        """
    # Bounding box method
    if method == 'bounding_box':
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        rmax = 0.5 * min(x_max - x_min, y_max - y_min)

    # Nearest neighbor method
    elif method == 'nearest_neighbour':
        tree = KDTree(points)
        distances, _ = tree.query(points, k=k+1)  # k=2 includes the point itself
        rmax = np.mean(distances[:, 1:k+1])  # Use 2nd column for nearest neighbor

    else:
        raise ValueError("Invalid method. Use 'bounding_box' or 'nearest_neighbour'.")

    return rmax



def ripley_k(points, r):
    """
    Calculate Ripley's K function for a set of points.

    :param points: A list of Shapely Point objects.
    :param r: The distance at which to compute K(r).
    :return: Ripley's K(r) value.
    """
    n = len(points)
    if n < 2:
        raise ValueError("At least two points are required to compute K(r).")

    # Create a KDTree for efficient neighbor search
    tree = cKDTree([(p.x, p.y) for p in points])
    distances, _ = tree.query([(p.x, p.y) for p in points], k=2)

    # Compute K(r)
    rad = distances[:, 1]
    k_r = np.sum(rad <= r) / (n * np.pi * r ** 2)
    return k_r



def ripley_l(points, r):
    """
    Calculate Ripley's L function for a set of points.

    :param points: A list of Shapely Point objects.
    :param r: The distance at which to compute L(r).
    :return: Ripley's L(r) value.
    """
    n = len(points)
    if n < 2:
        raise ValueError("At least two points are required to compute L(r).")

    k_r = ripley_k(points, r)
    r_mean = np.mean([point.distance(Point(0, 0)) for point in points])

    l_r = np.sqrt(k_r / (np.pi * r ** 2)) - r / r_mean
    return l_r



def pair_correlation_function(points, r):
    """
    Calculate the pair correlation function g(r).

    :param points: A list of Shapely Point objects.
    :param r: The distance at which to compute g(r).
    :return: g(r) value.
    """
    k_r = ripley_k(points, r)
    return k_r / (np.pi * r ** 2)



def nearest_neighbor_distance(points):
    """
    Calculate the nearest neighbor distance for a set of points.

    :param points: A list of Shapely Point objects.
    :return: The mean nearest neighbor distance.
    """
    n = len(points)
    if n < 2:
        raise ValueError("At least two points are required.")

    # Create a KDTree for efficient neighbor search
    tree = cKDTree([(p.x, p.y) for p in points])
    distances, _ = tree.query([(p.x, p.y) for p in points], k=2)

    # The nearest neighbor distance is the distance to the first neighbor
    nn_distances = distances[:, 1]
    return np.mean(nn_distances)



def density_estimate(points, window):
    """
    Estimate the density of points in a window.

    :param points: A list of Shapely Point objects.
    :param window: A Shapely Polygon object representing the observation window.
    :return: Estimated density of points per unit area.
    """
    n = len(points)
    if n == 0:
        return 0

    window_area = window.area
    if window_area <= 0:
        raise ValueError("The window area must be positive.")

    return n / window_area



def summary_statistics(points, window):
    """
    Calculate various summary statistics for a point pattern.

    :param points: A list of Shapely Point objects.
    :param window: A Shapely Polygon object representing the observation window.
    :return: A dictionary with statistics including intensity, clustering measure, etc.
    """
    n = len(points)
    area = window.area

    # Intensity (points per unit area)
    intensity = n / area if area > 0 else 0

    # Mean nearest neighbor distance
    nn_dist = nearest_neighbor_distance(points)

    # Ripley's K function at nearest neighbor distance
    k_nn = ripley_k(points, nn_dist)

    # Pair correlation function at nearest neighbor distance
    g_nn = pair_correlation_function(points, nn_dist)

    stats = {
        "intensity": intensity,
        "mean_nearest_neighbor_distance": nn_dist,
        "Ripley_K_at_nn": k_nn,
        "pair_correlation_at_nn": g_nn
    }

    return stats
