"""
Postprocessing module for pyMNPBEM simulations.

Provides:
- Data loading from simulation results
- Spectrum analysis (peak finding, FWHM, etc.)
- Field analysis (hotspots, statistics)
- Visualization tools (spectra, fields, surface charges)
- Data export (TXT, JSON, CSV)
"""

from .postprocess import PostprocessManager

__all__ = ['PostprocessManager']
