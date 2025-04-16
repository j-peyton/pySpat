"""
test_window.py

Test suite for the Window class in pySpat.

This test file verifies that the Window class behaves correctly:
- Initializes with valid boundaries
- Rejects invalid configurations
- Correctly computes width, height, and area
- Accurately checks if points lie within its bounds

Author: j-peyton
Date: 2025-04-16
"""

import numpy as np
import pytest
from pyspat import Window


def test_window_initialization_valid():
    w = Window((0, 10), (0, 5))
    assert w.x_range == (0, 10)
    assert w.y_range == (0, 5)

def test_window_initialization_invalid():
    with pytest.raises(ValueError):
        Window((5, 1), (0, 10))  # Invalid x range

    with pytest.raises(ValueError):
        Window((0, 10), (7, 2))  # Invalid y range

def test_window_geometry():
    w = Window((0, 4), (1, 6))
    assert w.width() == 4
    assert w.height() == 5
    assert w.area() == 20

def test_window_contains():
    w = Window((0, 5), (0, 5))
    inside = np.array([[1, 1], [2.5, 3.3], [0, 0], [5, 5]])
    outside = np.array([[5.1, 1], [-1, 2], [4, 5.1]])

    assert w.contains(inside) == True
    assert w.contains(outside) == False
