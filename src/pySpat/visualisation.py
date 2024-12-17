import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import numpy as np



def plot_point_pattern(points, window, title="Spatial Point Pattern"):
    """
    Plot the spatial point pattern within a given observation window.

    Parameters:
    ----------
    points : list of Shapely Point objects
        The points in the spatial pattern.
    window : Shapely Polygon object
        The observation window.
    title : str, optional
        The title of the plot (default is "Spatial Point Pattern").
    """
    fig, ax = plt.subplots()
    gdf = gpd.GeoDataFrame(geometry=points)

    # Plot the points
    gdf.plot(ax=ax, marker='o', color='blue', markersize=5, alpha=0.6)

    # Plot the window
    x, y = window.exterior.xy
    ax.fill(x, y, color='gray', alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.show()



def plot_intensity(window, intensity_func, title="Point Intensity"):
    """
    Plot the intensity of the spatial point pattern.

    Parameters:
    ----------
    points : list of Shapely Point objects
        The points in the spatial pattern.
    window : Shapely Polygon object
        The observation window.
    intensity_func : function
        A function that computes intensity (e.g., point density or kernel density estimate).
    title : str, optional
        The title of the plot (default is "Point Intensity").
    """
    x, y = window.exterior.xy
    grid_x, grid_y = np.mgrid[window.bounds[0]:window.bounds[2]:100j, window.bounds[1]:window.bounds[3]:100j]
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    intensity_values = np.array([intensity_func(Point(x, y)) for x, y in grid_points])
    intensity_matrix = intensity_values.reshape(grid_x.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, intensity_matrix, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Intensity')
    plt.plot(x, y, 'k-', alpha=0.5)
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()



def plot_clusters(points, cluster_radius, window, title="Clustered Points"):
    """
    Plot the clustered points within a given observation window.

    Parameters:
    ----------
    points : list of Shapely Point objects
        The points in the spatial pattern.
    cluster_radius : float
        Radius to define clusters.
    window : Shapely Polygon object
        The observation window.
    title : str, optional
        The title of the plot (default is "Clustered Points").
    """
    clusters = []
    unvisited = set(range(len(points)))

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

    fig, ax = plt.subplots()
    for cluster in clusters:
        cluster_points = [points[i] for i in cluster]
        gdf = gpd.GeoDataFrame(geometry=cluster_points)
        gdf.plot(ax=ax, marker='o', markersize=5, alpha=0.6)

    # Plot the window
    x, y = window.exterior.xy
    ax.fill(x, y, color='gray', alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.show()



def plot_strata(points, window, strata_width, strata_height, title="Stratified Points"):
    """
    Plot the spatially stratified point pattern within a given observation window.

    Parameters:
    ----------
    points : list of Shapely Point objects
        The points in the spatial pattern.
    window : Shapely Polygon object
        The observation window.
    strata_width : float
        Width of each stratum.
    strata_height : float
        Height of each stratum.
    title : str, optional
        The title of the plot (default is "Stratified Points").
    """
    n_strata_x = int(np.ceil(window.bounds[2] / strata_width))
    n_strata_y = int(np.ceil(window.bounds[3] / strata_height))

    fig, ax = plt.subplots()
    for i in range(n_strata_x):
        for j in range(n_strata_y):
            x_start = window.bounds[0] + i * strata_width
            y_start = window.bounds[1] + j * strata_height
            ax.add_patch(plt.Rectangle((x_start, y_start), strata_width, strata_height, fill=None, edgecolor='blue', alpha=0.5))

    gdf = gpd.GeoDataFrame(geometry=points)
    gdf.plot(ax=ax, marker='o', color='red', markersize=5, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.show()



def plot_ripley_l(spatial_pattern, max_distance, step=0.1, title="Ripley's L function"):
    """
    Plot Ripley's L function for a spatial point pattern.

    Parameters:
    ----------
    spatial_pattern : SpatialPointPattern object
        The spatial point pattern to analyze.
    max_distance : float
        The maximum distance to calculate the function.
    step : float, optional
        The step size for distance calculation (default is 0.1).
    title : str, optional
        The title of the plot (default is "Ripley's L function").
    """
    distances = np.arange(0, max_distance, step)
    l_values = []

    for d in distances:
        pairs = 0
        for i, point in enumerate(spatial_pattern.points):
            for j in range(i + 1, len(spatial_pattern.points)):
                if point.distance(spatial_pattern.points[j]) <= d:
                    pairs += 1
        ll = np.sqrt(pairs / (len(spatial_pattern.points) * np.pi * d**2)) - d
        l_values.append(ll)

    plt.figure(figsize=(8, 6))
    plt.plot(distances, l_values, marker='o')
    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('L(d) - d')
    plt.show()
