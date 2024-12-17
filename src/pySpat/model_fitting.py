import numpy as np
from shapely.geometry import Point
from .geometry import random_window


class SpatialPointPattern:
    """
    A class to represent a spatial point pattern.

    Attributes:
    ----------
    points : list of Shapely Point objects
        The points in the spatial point pattern.
    window : Shapely Polygon object
        The observation window.
    """

    def __init__(self, points, window):
        """
        Constructs all the necessary attributes for the spatial point pattern object.

        Parameters:
        ----------
        points : list of Shapely Point objects
            The points in the spatial point pattern.
        window : Shapely Polygon object
            The observation window.
        """
        self.points = points
        self.window = window
        self.n_points = len(points)

    def from_dataframe(self, dataframe):
        """
        Initialize the spatial point pattern from a dataframe with 'x' and 'y' columns.

        Parameters:
        ----------
        dataframe : pandas DataFrame
            A dataframe containing 'x' and 'y' columns representing coordinates of points.
        """
        self.points = [Point(row['x'], row['y']) for index, row in dataframe.iterrows()]
        minx, miny, maxx, maxy = self.window.bounds
        bounds = (minx, miny, maxx, maxy)  # Create bounds tuple
        self.window = random_window(bounds)

    def summary_statistics(self):
        """
        Calculate summary statistics for the spatial point pattern.

        Returns:
        -------
        dict
            A dictionary containing summary statistics:
            - 'total_points': Number of points in the pattern.
            - 'mean_distance': Mean distance between points.
            - 'min_distance': Minimum distance between points.
            - 'max_distance': Maximum distance between points.
        """
        n = len(self.points)
        if n < 2:
            return {
                'total_points': n,
                'mean_distance': None,
                'min_distance': None,
                'max_distance': None
            }

        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                distance = self.points[i].distance(self.points[j])
                distances.append(distance)

        distances = np.array(distances)
        summary = {
            'total_points': n,
            'mean_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
        return summary


def fit_poisson_process(spatial_pattern):
    """
    Fit a Poisson process to the spatial point pattern.

    Parameters:
    ----------
    spatial_pattern : SpatialPointPattern object
        The spatial point pattern to fit.

    Returns:
    -------
    float
        The intensity (lambda) of the Poisson process.
    """
    n_points = spatial_pattern.n_points
    area = spatial_pattern.window.area
    if area <= 0:
        raise ValueError("The observation window area must be positive.")

    intensity = n_points / area
    return intensity


def fit_clustered_process(spatial_pattern, cluster_radius):
    """
    Fit a clustered process to the spatial point pattern.

    Parameters:
    ----------
    spatial_pattern : SpatialPointPattern object
        The spatial point pattern to fit.
    cluster_radius : float
        The radius around each point to consider as a cluster.

    Returns:
    -------
    dict
        A dictionary with:
        - 'mean_cluster_size': Mean number of points per cluster.
    """
    points = spatial_pattern.points
    n_points = len(points)

    if n_points < 2:
        return {'mean_cluster_size': None}

    clusters = []
    unvisited = set(range(n_points))
    while unvisited:
        point_index = unvisited.pop()
        cluster = [point_index]
        stack = [point_index]
        while stack:
            current_index = stack.pop()
            current_point = points[current_index]
            for i in list(unvisited):
                other_point = points[i]
                if current_point.distance(other_point) <= cluster_radius:
                    cluster.append(i)
                    stack.append(i)
                    unvisited.remove(i)
        clusters.append(cluster)

    cluster_sizes = [len(cluster) for cluster in clusters]
    mean_cluster_size = np.mean(cluster_sizes)
    return {'mean_cluster_size': mean_cluster_size}


def fit_stratified_process(spatial_pattern, strata_width, strata_height):
    """
    Fit a spatially stratified process to the spatial point pattern.

    Parameters:
    ----------
    spatial_pattern : SpatialPointPattern object
        The spatial point pattern to fit.
    strata_width : float
        Width of each stratum.
    strata_height : float
        Height of each stratum.

    Returns:
    -------
    dict
        A dictionary with:
        - 'strata_distribution': Number of points per stratum.
    """
    points = spatial_pattern.points
    n_strata_x = int(np.ceil(spatial_pattern.window.bounds[2] / strata_width))
    n_strata_y = int(np.ceil(spatial_pattern.window.bounds[3] / strata_height))

    strata_distribution = np.zeros((n_strata_x, n_strata_y))

    for point in points:
        x_stratum = int(point.x // strata_width)
        y_stratum = int(point.y // strata_height)
        strata_distribution[x_stratum, y_stratum] += 1

    return {'strata_distribution': strata_distribution.tolist()}
