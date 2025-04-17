"""
test_ppplot.py

Basic non-visual tests for the pyspat.plot.ppplot module.
Tests that Plotly plotting code executes without error for valid inputs.
We don't check visual correctness (manual) but ensure no exceptions are raised.

Author: j-peyton
Date: 2025-04-17
"""

import pytest
from pyspat import PointPattern, Window
from pyspat.plot.ppplot import plot_pp


def test_plot_point_pattern_unmarked():
    coords = [(1, 1), (2, 2), (3, 3)]
    window = Window((0, 5), (0, 5))
    pp = PointPattern(coords, window)
    try:
        plot_pp(pp, title="Unmarked Pattern")
    except Exception as e:
        pytest.fail(f"plot_point_pattern raised an exception on unmarked input: {e}")


def test_plot_point_pattern_marked():
    coords = [(1, 1), (2, 2), (3, 3)]
    marks = ['a', 'b', 'a']
    window = Window((0, 5), (0, 5))
    pp = PointPattern(coords, window, marks=marks)
    try:
        plot_pp(pp, title="Marked Pattern")
    except Exception as e:
        pytest.fail(f"plot_point_pattern raised an exception on marked input: {e}")
