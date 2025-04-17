"""
test_distances.py

Unit tests for distance-related functions in pyspat.geometry.distances:
- pairwise_distances
- k_nearest_neighbors
- distance_matrix_stats

These tests check correctness of distance outputs, shapes, expected statistics,
and error handling for invalid inputs.

Author: j-peyton
Date: 2025-04-17
"""

import numpy as np
import pytest
from pyspat import PointPattern, Window
from pyspat.geom.distances import (
    pairwise_distances,
    k_nearest_neighbors,
    distance_matrix_stats
)


def test_pairwise_distances_shape():
    coords = [(0, 0), (1, 0), (0, 1)]
    window = Window((0, 2), (0, 2))
    pp = PointPattern(coords, window)
    d = pairwise_distances(pp)
    assert d.shape == (3, 3)
    assert np.allclose(np.diag(d), 0)


def test_k_nearest_neighbors_k1():
    coords = [(0, 0), (1, 0), (0, 1)]
    window = Window((0, 2), (0, 2))
    pp = PointPattern(coords, window)
    knn = k_nearest_neighbors(pp, k=1)
    assert isinstance(knn, np.ndarray)
    assert knn.shape == (3,)
    assert np.all(knn > 0)


def test_k_nearest_neighbors_k2():
    coords = [(0, 0), (1, 0), (0, 1), (1, 1)]
    window = Window((0, 2), (0, 2))
    pp = PointPattern(coords, window)
    knn = k_nearest_neighbors(pp, k=2)
    assert knn.shape == (4, 2)
    assert np.all(knn[:, 1] >= knn[:, 0])


def test_k_too_large():
    coords = [(0, 0), (1, 1)]
    window = Window((0, 2), (0, 2))
    pp = PointPattern(coords, window)
    with pytest.raises(ValueError):
        k_nearest_neighbors(pp, k=2)


def test_distance_matrix_stats_valid():
    coords = [(0, 0), (3, 4)]  # distance = 5
    window = Window((0, 5), (0, 5))
    pp = PointPattern(coords, window)
    stats = distance_matrix_stats(pp)
    assert isinstance(stats, dict)
    assert np.isclose(stats["mean"], 5)
    assert np.isclose(stats["min"], 5)
    assert np.isclose(stats["max"], 5)