"""
test_csr.py

Unit tests for the CSR simulation module (pyspat.sim.csr).
Includes tests for fixed-count CSR and Poisson-based CSR simulations.

Author: j-peyton
Date: 2025-04-17
"""

from pyspat import Window, PointPattern
from pyspat.sim.csr import runifpoint, rpoispp


def test_runifpoint_fixed_count():
    window = Window((0, 1), (0, 1))
    pp = runifpoint(50, window)
    assert isinstance(pp, PointPattern)
    assert len(pp) == 50
    assert window.contains(pp.coordinates())


def test_runifpoint_zero():
    window = Window((0, 1), (0, 1))
    pp = runifpoint(0, window)
    assert isinstance(pp, PointPattern)
    assert len(pp) == 0


def test_rpoispp_random_count():
    window = Window((0, 1), (0, 1))
    lambda_ = 100  # points per unit area
    pp = rpoispp(lambda_, window)
    assert isinstance(pp, PointPattern)
    assert pp.window == window
    assert len(pp) >= 0


def test_rpoispp_low_lambda():
    window = Window((0, 1), (0, 1))
    lambda_ = 0.001
    pp = rpoispp(lambda_, window)
    assert isinstance(pp, PointPattern)
    assert len(pp) >= 0
