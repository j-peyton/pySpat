"""
test_pointpattern.py

Test suite for the PointPattern class in pySpat.

This test file checks that PointPattern:
- Initializes correctly with valid data
- Rejects mismatched marks or out-of-bounds coordinates
- Provides accurate coordinate access
- Supports basic methods like len(), has_marks(), and summary()

Author: j-peyton
Date: 2025-04-16
"""

import numpy as np
import pytest
from pyspat import Window
from pyspat import PointPattern


def test_pointpattern_initialization():
    w = Window((0, 10), (0, 10))
    coords = [(1, 2), (3, 4), (5, 6)]
    pp = PointPattern(coords, window=w)
    assert len(pp) == 3
    assert np.allclose(pp.coordinates(), np.array(coords))


def test_pointpattern_with_marks():
    w = Window((0, 10), (0, 10))
    coords = [(1, 2), (3, 4), (5, 6)]
    marks = ['a', 'b', 'c']
    pp = PointPattern(coords, window=w, marks=marks)
    assert pp.has_marks() is True
    assert pp.marks == marks


def test_pointpattern_mark_length_mismatch():
    w = Window((0, 10), (0, 10))
    coords = [(1, 2), (3, 4)]
    marks = ['only one']
    with pytest.raises(ValueError):
        PointPattern(coords, window=w, marks=marks)


def test_pointpattern_out_of_bounds():
    w = Window((0, 2), (0, 2))
    coords = [(1, 1), (2.1, 0.5)]  # one point outside
    with pytest.raises(ValueError):
        PointPattern(coords, window=w)


def test_pointpattern_summary():
    w = Window((0, 10), (0, 10))
    coords = [(1, 1), (2, 2)]
    marks = ['x', 'y']
    pp = PointPattern(coords, w, marks)
    summary = pp.summary()
    assert "PointPattern with 2 points" in summary
    assert "Marked: Yes" in summary
    assert "Window:" in summary
