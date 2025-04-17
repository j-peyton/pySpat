"""
csr.py

This module provides functions to simulate complete spatial randomness (CSR)
within a rectangular window. The CSR model assumes points are uniformly
distributed over the observation window, either with a fixed number of points
or based on a Poisson process with intensity lambda.

Author: j-peyton
Date: 2025-04-17
"""

import numpy as np
from pyspat import Window
from pyspat import PointPattern


def runifpoint(n: int, window: Window) -> PointPattern:
    """
    Simulate a fixed number of points under CSR.

    Args:
        n: Number of points to generate.
        window: A Window object defining the observation region.

    Returns:
        A PointPattern object of n uniform random points within the window.
    """
    x = np.random.uniform(window.x_range[0], window.x_range[1], n)
    y = np.random.uniform(window.y_range[0], window.y_range[1], n)
    points = list(zip(x, y))
    return PointPattern(points, window)


def rpoispp(lambda_: float, window: Window) -> PointPattern:
    """
    Simulate a Poisson point process (CSR with random count).

    Args:
        lambda_: Intensity (points per unit area).
        window: A Window object defining the observation region.

    Returns:
        A PointPattern object with Poisson-distributed number of points.
    """
    area = window.area()
    n = np.random.poisson(lambda_ * area)
    return runifpoint(n, window)
