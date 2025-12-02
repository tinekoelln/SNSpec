# src/single_sne/__init__.py

"""
Top-level package for SNSpec.

Expose the main subpackages and a few key functions/classes.
"""

from . import spectra, io, pseudobol_lightcurve
from .pseudobol_lightcurve.make_bol_lc import mklcbol
from .pseudobol_lightcurve.aux import rd_lcbol_data

__all__ = [
    "spectra",
    "io",
    "pseudobol_lightcurve",
    "mklcbol",
    "rd_lcbol_data",
]