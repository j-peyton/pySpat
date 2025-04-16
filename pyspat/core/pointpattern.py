"""
pointpattern.py

This module defines the PointPattern class, which is the foundational data structure
for representing spatial point patterns in the pySpat library.

A PointPattern object stores 2D coordinates and optional marks, and it is always
associated with a Window object that defines the spatial domain of observation.

This class provides basic validation, accessors, and summary statistics to support
spatial statistical methods. It is inspired by the 'ppp' class from the R spatstat package.

Author: j-peyton
Date: 2025-04-16
"""

import numpy as np
from typing import Optional, List, Tuple
from .window import Window


class PointPattern:
    """
    Represents a 2D spatial point pattern.

    Attributes:
        points (np.ndarray): Nx2 array of (x, y) coordinates.
        window (Window): The spatial window within which the points lie.
        marks (Optional[List]): Optional list of marks associated with each point.
    """

    def __init__(self, points: List[Tuple[float, float]], window: Window, marks: Optional[List] = None):
        """
        Initialize a PointPattern object.

        Args:
            points: A list of (x, y) tuples representing point coordinates.
            window: A Window object defining the observation region.
            marks: Optional list of marks (must be same length as points).

        Raises:
            ValueError: If points lie outside the window or marks are mismatched.
        """
        self.points = np.array(points, dtype=float)
        self.window = window

        if not self.window.contains(self.points):
            raise ValueError("Some points lie outside the observation window.")

        if marks is not None:
            if len(marks) != len(points):
                raise ValueError("Length of marks must match number of points.")
            self.marks = marks
        else:
            self.marks = None

    def __len__(self) -> int:
        """Return the number of points."""
        return self.points.shape[0]

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"PointPattern(n={len(self)}, window={self.window})"

    def coordinates(self) -> np.ndarray:
        """
        Return a copy of the coordinates.

        Returns:
            Nx2 numpy array of (x, y) point coordinates.
        """
        return self.points.copy()

    def has_marks(self) -> bool:
        """Return True if the pattern has marks."""
        return self.marks is not None

    def summary(self) -> str:
        """
        Return a summary of the point pattern.

        Returns:
            A human-readable summary string.
        """
        s = f"PointPattern with {len(self)} points\n"
        s += f"Window: {self.window}\n"
        if self.has_marks():
            s += f"Marked: Yes (first 5 marks: {self.marks[:5]})\n"
        else:
            s += f"Marked: No\n"
        return s
