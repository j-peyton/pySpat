"""
summary.py

This module provides spatial summary statistics for point patterns.
Specifically, it implements:
- Ripley's K function
- Besag's L function
- Pair correlation function (g)

These are foundational tools in spatial point pattern analysis to detect
clustering or inhibition over spatial scales.

All functions accept a PointPattern object from pyspat.core.pointpattern.

Author: j-peyton
Date: 2025-04-17
"""

import numpy as np
from pyspat import PointPattern


def ripley_k(pp: PointPattern, radii: np.ndarray, edge_correction: bool = False) -> np.ndarray:
    """
    Compute Ripley's K function for a range of radii.

    Args:
        pp: A PointPattern object.
        radii: 1D numpy array of radii at which to compute K.
        edge_correction: If True, apply border correction (currently not implemented).

    Returns:
        1D numpy array of K(r) values.
    """
    n = len(pp)
    if n < 2:
        raise ValueError("Ripley's K requires at least 2 points.")

    area = pp.window.area()
    coords = pp.coordinates()
    dists = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :])**2, axis=2))

    np.fill_diagonal(dists, np.inf)  # exclude self-distances

    k_values = []
    for r in radii:
        count = np.sum(dists <= r)
        k = (area / (n * (n - 1))) * count
        k_values.append(k)

    return np.array(k_values)


def besag_l(pp: PointPattern, radii: np.ndarray) -> np.ndarray:
    """
    Compute Besag's L function based on Ripley's K.

    Args:
        pp: A PointPattern object.
        radii: 1D numpy array of radii at which to compute L.

    Returns:
        1D numpy array of L(r) values.
    """
    k = ripley_k(pp, radii)
    return np.sqrt(k / np.pi)


def pair_correlation(pp: PointPattern, radii: np.ndarray, dr: float) -> np.ndarray:
    """
    Estimate the pair correlation function g(r).

    Args:
        pp: A PointPattern object.
        radii: 1D numpy array of radii at which to evaluate g(r).
        dr: Bin width for annuli.

    Returns:
        1D numpy array of g(r) values.
    """
    n = len(pp)
    if n < 2:
        raise ValueError("Ripley's K requires at least 2 points.")

    area = pp.window.area()
    density = n / area
    coords = pp.coordinates()
    dists = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :])**2, axis=2))

    np.fill_diagonal(dists, np.inf)  # exclude self-distances

    g_values = []
    for r in radii:
        mask = (dists > (r - dr / 2)) & (dists <= (r + dr / 2))
        count = np.sum(mask)
        shell_area = np.pi * ((r + dr / 2)**2 - (r - dr / 2)**2)
        expected = density * shell_area * (n - 1)
        g = count / (n * expected)
        g_values.append(g)

    return np.array(g_values)
