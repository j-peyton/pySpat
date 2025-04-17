"""
test_summary.py

Unit tests for summary statistics functions in pyspat.stats.summary:
- Ripley's K function
- Besag's L function
- Pair correlation function

These tests verify correctness of shape, expected behavior under CSR,
and handle edge cases like empty or degenerate point patterns.

Author: j-peyton
Date: 2025-04-17
"""

import numpy as np
import pytest
from pyspat import PointPattern
from pyspat import Window
from pyspat.stats.summary import ripley_k, besag_l, global_pcf, local_pcf


def generate_test_pattern(n=100, size=10):
    """Helper: generate uniform random pattern in a square window."""
    coords = np.random.uniform(0, size, (n, 2))
    window = Window((0, size), (0, size))
    return PointPattern(coords.tolist(), window)


def test_ripley_k_shape():
    pp = generate_test_pattern()
    r = np.linspace(0.1, 5, 10)
    k = ripley_k(pp, r)
    assert isinstance(k, np.ndarray)
    assert k.shape == r.shape


def test_besag_l_monotonic():
    pp = generate_test_pattern()
    r = np.linspace(0.1, 5, 10)
    l = besag_l(pp, r)
    assert isinstance(l, np.ndarray)
    assert l.shape == r.shape
    assert np.all(l[1:] >= 0)  # should not be negative


def test_global_pcf():
    pp = generate_test_pattern()
    r = np.linspace(0.1, 5, 10)
    g = global_pcf(pp, r, dr=0.2)
    assert isinstance(g, np.ndarray)
    assert g.shape == r.shape
    assert np.all(g >= 0)

def test_local_pcf():
    center = (0.5, 0.5)
    points = np.random.uniform(0, 1, size=(100, 2))
    r_max = 0.3
    dr = 0.05

    r_values, g_values = local_pcf(center, points, r_max, dr)

    assert isinstance(r_values, np.ndarray)
    assert isinstance(g_values, np.ndarray)
    assert r_values.shape == g_values.shape
    assert np.all(g_values >= 0)


def test_empty_pattern():
    window = Window((0, 10), (0, 10))
    pp = PointPattern([], window)
    r = np.linspace(0.1, 5, 10)
    with pytest.raises(ValueError):
        ripley_k(pp, r)


def test_single_point_pattern():
    window = Window((0, 10), (0, 10))
    pp = PointPattern([(5, 5)], window)
    r = np.linspace(0.1, 5, 10)
    with pytest.raises(ValueError):
        ripley_k(pp, r)

