"""
test_integration.py

Integration test to verify that all core components of pySpat
work correctly when used together:
- Simulate CSR via Poisson process
- Validate as PointPattern
- Plot using Plotly
- Compute summary statistics (K, L, g)

Author: j-peyton
Date: 2025-04-17
"""

import numpy as np
from pyspat import Window, PointPattern
from pyspat.sim.csr import rpoispp
from pyspat.plot.ppplot import plot_pp
from pyspat.stats.summary import ripley_k, besag_l, global_pcf


def test_csr_pipeline():
    # Simulate CSR pattern
    window = Window((0, 1), (0, 1))
    lambda_ = 100  # intensity
    pp = rpoispp(lambda_, window)

    # Check it is a valid PointPattern
    assert isinstance(pp, PointPattern)
    assert pp.window == window
    assert len(pp) >= 0

    # Plot the result (this will display an interactive plot)
    plot_pp(pp, title="Simulated CSR via rpoispp")

    # Calculate summary statistics
    r = np.linspace(0.01, 0.2, 10)
    k = ripley_k(pp, r)
    l = besag_l(pp, r)
    g = global_pcf(pp, r, dr=0.01)

    # Check output shapes
    assert k.shape == r.shape
    assert l.shape == r.shape
    assert g.shape == r.shape

    # Ensure all values are non-negative
    assert np.all(k >= 0)
    assert np.all(l >= 0)
    assert np.all(g >= 0)