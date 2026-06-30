"""
swot_analysis
=============

Tools for computing along-track SSH(A) wavenumber spectra and along-track
SSH slope / geostrophic velocity from the SWOT L2 LR Basic product.

Submodules
----------
swot_alongtrack_spectra
    Along-track wavenumber spectra (segment-based, per-swath, NaN/land
    and nadir-gap aware).
swot_alongtrack_slope
    Along-track SSH slope and geostrophic velocity, following Stammer
    (1997), plus slope wavenumber spectra.

The most commonly used functions/classes are re-exported here for
convenience, so e.g. ``from swot_analysis import compute_pass_spectra``
works without reaching into the submodules directly.
"""

from importlib.metadata import version, PackageNotFoundError

from .swot_alongtrack_spectra import (
    SegmentSpectrum,
    PassSpectrumResult,
    load_swot_l2,
    along_track_distance_km,
    split_left_right_swaths,
    compute_swath_spectra,
    compute_pass_spectra,
)

#from .swot_alongtrack_slope import (
#    SwathSlopeResult,
#    SlopeSegmentStats,
#    compute_swath_slope,
#    compute_slope_segment_stats,
#    compute_pass_slope,
#    compute_swath_slope_spectrum,
#)

try:
    __version__ = version("swot_analysis")
except PackageNotFoundError:  # package not installed, e.g. running from source
    __version__ = "0.0.0+unknown"

__all__ = [
    "SegmentSpectrum",
    "PassSpectrumResult",
    "load_swot_l2",
    "along_track_distance_km",
    "split_left_right_swaths",
    "compute_swath_spectra",
    "compute_pass_spectra",
    "SwathSlopeResult",
    "SlopeSegmentStats",
    "compute_swath_slope",
    "compute_slope_segment_stats",
    "compute_pass_slope",
    "compute_swath_slope_spectrum",
    "__version__",
]
