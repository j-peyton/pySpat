"""
distances.py

This module provides geometric distance-related functions for spatial
point pattern analysis in pySpat. All functions are compatible with the
PointPattern class and rely only on NumPy for computation.

Included functions:
- pairwise_distances: full NxN matrix of distances
- k_nearest_neighbors: distances to the k-th nearest neighbors
- distance_matrix_stats: summary stats on pairwise distances

Author: j-peyton
Date: 2025-04-17
"""

import numpy as np
from pyspat import PointPattern


def pairwise_distances(pp: PointPattern) -> np.ndarray:
    """
    Compute the NxN pairwise Euclidean distance matrix.

    Args:
        pp: A PointPattern object.

    Returns:
        NxN numpy array of distances.
    """
    coords = pp.coordinates()
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist


def k_nearest_neighbors(pp: PointPattern, k: int = 1) -> np.ndarray:
    """
    Compute distances to the k-th nearest neighbors for each point.

    Args:
        pp: A PointPattern object.
        k: The number of nearest neighbors to consider (default: 1).

    Returns:
        Nxk array if k > 1, or 1D array if k == 1, with distances to k nearest neighbors.

    Raises:
        ValueError: If k >= number of points.
    """
    n = len(pp)
    if k >= n:
        raise ValueError("k must be less than the number of points in the pattern.")

    dists = pairwise_distances(pp)
    np.fill_diagonal(dists, np.inf)
    sorted_dists = np.sort(dists, axis=1)

    if k == 1:
        return sorted_dists[:, 0]
    else:
        return sorted_dists[:, :k]


def distance_matrix_stats(pp: PointPattern) -> dict:
    """
    Compute basic statistics from the pairwise distance matrix.

    Args:
        pp: A PointPattern object.

    Returns:
        Dictionary with mean, std, min, max (excluding self-distances).
    """
    d = pairwise_distances(pp)
    mask = ~np.eye(len(pp), dtype=bool)  # exclude diagonal
    vals = d[mask]
    return {
        "mean": np.mean(vals),
        "std": np.std(vals),
        "min": np.min(vals),
        "max": np.max(vals)
    }