"""
Utilities for Spatial CIMIS weekly climatology and Fourier analysis.
"""

from .io import load_spatial_cimis_variable
from .climatology import calculate_weekly_climatology
from .fourier import (
    fourier_transform_climatology_components,
    reconstruct_climatology_from_fourier_components,
)
from .processing import run_pipeline

__all__ = [
    "load_spatial_cimis_variable",
    "calculate_weekly_climatology",
    "fourier_transform_climatology_components",
    "reconstruct_climatology_from_fourier_components",
    "run_pipeline",
]

