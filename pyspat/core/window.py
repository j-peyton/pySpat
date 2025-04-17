"""
window.py

This module defines the Window class for pySpat. A Window represents the
spatial observation domain for point pattern analysis. In this initial version,
we support only rectangular windows aligned with the axes.

The Window object provides basic geometry, point containment checks,
and area computations, serving as the spatial foundation for PointPattern
and other operations in pySpat.

Author: j-peyton
Date: 2025-04-16
"""

import numpy as np
from typing import Tuple


class Window:
    """
    Represents a 2D rectangular observation window.

    Attributes:
        x_range (Tuple[float, float]): (xmin, xmax) limits.
        y_range (Tuple[float, float]): (ymin, ymax) limits.
    """

    def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        """
        Initialize a rectangular window.

        Args:
            x_range: Tuple (xmin, xmax) defining horizontal bounds.
            y_range: Tuple (ymin, ymax) defining vertical bounds.

        Raises:
            ValueError: If the bounds are invalid (min >= max).
        """
        if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
            raise ValueError("Invalid window ranges: must have min < max for both axes.")

        self.x_range = x_range
        self.y_range = y_range

    def contains(self, points: np.ndarray) -> bool:
        """
        Check if all points lie within the window.

        Args:
            points: Nx2 numpy array of (x, y) coordinates.

        Returns:
            True if all points are inside the window, False otherwise.
        """
        if points.size == 0:
            return True  # or False, depending on design
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points array must be of shape (N, 2)")
        x, y = points[:, 0], points[:, 1]

        within_x = (self.x_range[0] <= x) & (x <= self.x_range[1])
        within_y = (self.y_range[0] <= y) & (y <= self.y_range[1])
        return np.all(within_x & within_y)

    def area(self) -> float:
        """Return the area of the window."""
        width = self.x_range[1] - self.x_range[0]
        height = self.y_range[1] - self.y_range[0]
        return width * height

    def width(self) -> float:
        """Return the width of the window."""
        return self.x_range[1] - self.x_range[0]

    def height(self) -> float:
        """Return the height of the window."""
        return self.y_range[1] - self.y_range[0]

    def __repr__(self) -> str:
        """String representation of the window."""
        return f"Window(x_range={self.x_range}, y_range={self.y_range})"
