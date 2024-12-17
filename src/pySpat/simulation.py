import numpy as np
from shapely.geometry import Point, Polygon
from scipy.stats import uniform


def generate_poisson_points(window, intensity, n_points=None):
    """
    Generate a random point pattern using a Poisson process.

    :param window: A Shapely Polygon object representing the observation window.
    :param intensity: Intensity of the Poisson process (number of points per unit area).
    :param n_points: Optional number of points to generate. If not provided, the total number of points is
                     determined based on the intensity and the area of the window.
    :return: A list of Shapely Point objects.
    """
    window_area = window.area
    if window_area <= 0:
        raise ValueError("The window area must be positive.")

    if n_points is None:
        n_points = int(intensity * window_area)

    points = []
    for _ in range(n_points):
        x = uniform.rvs(0, window.bounds[2] - window.bounds[0])
        y = uniform.rvs(0, window.bounds[3] - window.bounds[1])
        point = Point(x, y)
        if window.contains(point):
            points.append(point)

    return points



def generate_cluster_points(window, n_clusters, points_per_cluster, cluster_radius):
    """
    Generate a clustered point pattern.

    :param window: A Shapely Polygon object representing the observation window.
    :param n_clusters: Number of clusters to generate.
    :param points_per_cluster: Number of points per cluster.
    :param cluster_radius: Radius within which points are clustered.
    :return: A list of Shapely Point objects.
    """
    points = []
    for _ in range(n_clusters):
        center_x = uniform.rvs(window.bounds[0], window.bounds[2] - window.bounds[0])
        center_y = uniform.rvs(window.bounds[1], window.bounds[3] - window.bounds[1])

        for _ in range(points_per_cluster):
            angle = uniform.rvs(0, 2 * np.pi)
            radius = uniform.rvs(0, cluster_radius)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            point = Point(x, y)
            if window.contains(point):
                points.append(point)

    return points



def generate_stratified_points(window, n_strata, points_per_stratum):
    """
    Generate a spatially stratified random point pattern.

    :param window: A Shapely Polygon object representing the observation window.
    :param n_strata: Number of strata to divide the window into.
    :param points_per_stratum: Number of points per stratum.
    :return: A list of Shapely Point objects.
    """
    points = []
    strata_width = (window.bounds[2] - window.bounds[0]) / n_strata
    strata_height = (window.bounds[3] - window.bounds[1]) / n_strata

    for i in range(n_strata):
        for j in range(n_strata):
            x_start = window.bounds[0] + i * strata_width
            y_start = window.bounds[1] + j * strata_height

            for _ in range(points_per_stratum):
                x = uniform.rvs(x_start, strata_width)
                y = uniform.rvs(y_start, strata_height)
                point = Point(x, y)
                if window.contains(point):
                    points.append(point)

    return points



def simulate_random_patterns(window, n_patterns, intensity=None, n_clusters=None, points_per_cluster=None,
                             cluster_radius=None, points_per_stratum=None, n_strata=None):
    """
    Simulate multiple random spatial point patterns.

    :param window: A Shapely Polygon object representing the observation window.
    :param n_patterns: Number of patterns to generate.
    :param intensity: Intensity for Poisson point pattern.
    :param n_clusters: Number of clusters for clustered pattern.
    :param points_per_cluster: Points per cluster for clustered pattern.
    :param cluster_radius: Radius of clusters.
    :param points_per_stratum: Points per stratum for stratified pattern.
    :param n_strata: Number of strata for stratified pattern.
    :return: A list of lists containing Shapely Point objects for each pattern.
    """
    patterns = []
    for _ in range(n_patterns):
        if intensity is not None:
            patterns.append(generate_poisson_points(window, intensity))
        elif n_clusters is not None and points_per_cluster is not None and cluster_radius is not None:
            patterns.append(generate_cluster_points(window, n_clusters, points_per_cluster, cluster_radius))
        elif points_per_stratum is not None and n_strata is not None:
            patterns.append(generate_stratified_points(window, n_strata, points_per_stratum))
        else:
            raise ValueError("Insufficient parameters to generate patterns.")

    return patterns
