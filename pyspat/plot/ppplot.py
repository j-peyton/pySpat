"""
ppplot.py

This module provides plotting utilities for visualizing point patterns.
It uses Plotly for interactive rendering and supports optional
mark-based color coding.

Author: j-peyton
Date: 2025-04-17
"""

import plotly.express as px
import pandas as pd
from pyspat.core.pointpattern import PointPattern


def plot_pp(pp: PointPattern, title="Point Pattern"):
    """
    Display an interactive scatter plot using Plotly.
    Color points by mark if available.

    Args:
        pp: A PointPattern object.
        title: Title of the plot.
    """
    coords = pp.coordinates()
    df = pd.DataFrame(coords, columns=["x", "y"])
    if pp.has_marks():
        df["mark"] = pp.marks
        fig = px.scatter(df, x="x", y="y", color="mark",
                         title=title, width=600, height=600)
    else:
        fig = px.scatter(df, x="x", y="y",
                         title=title, width=600, height=600)

    fig.update_layout(yaxis_scaleanchor="x")
    fig.show()
