from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from swot_analysis.swot_alongtrack_spectra import (
    load_swot_l2,
    EARTH_RADIUS_KM,
    along_track_distance_km,
    split_left_right_swaths,
    _interp_nan_1d,
    _segment_bounds,
    compute_swath_spectra,
    PassSpectrumResult,
)

GRAVITY = 9.81  # m s^-2
OMEGA_EARTH = 7.2921150e-5  # rad s^-1, Earth's rotation rate


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class SwathSlopeResult:
    """Full-pass along-track slope field for one swath, plus segment stats."""

    swath: str
    along_track_distance_km: np.ndarray   # (num_lines_swath,)
    latitude: np.ndarray                  # (num_lines_swath, num_pixels_swath)
    longitude: np.ndarray
    slope: np.ndarray                     # (num_lines_swath, num_pixels_swath), m/m (dimensionless), NaN where not computable
    geostrophic_velocity: Optional[np.ndarray]  # same shape, m/s, or None
    segments: list  # list[SlopeSegmentStats]


@dataclasses.dataclass
class SlopeSegmentStats:
    """Slope statistics (RMS slope, slope variance) for one along-track segment."""

    swath: str
    segment_index: int
    lat_mean: float
    lat_min: float
    lat_max: float
    lon_mean: float
    along_track_distance_start_km: float
    along_track_distance_end_km: float
    n_pixels_used: int
    valid_fraction: float
    slope_mean: float        # mean slope (should be ~0 after detrend-like behaviour; informative bias check)
    slope_rms: float         # RMS along-track slope, m/m
    slope_variance: float    # slope variance, (m/m)^2
    geostrophic_velocity_rms: Optional[float]  # RMS geostrophic velocity, m/s, or None


# --------------------------------------------------------------------------- #
# Core slope computation
# --------------------------------------------------------------------------- #

def _coriolis_parameter(lat_deg: np.ndarray) -> np.ndarray:
    """f = 2 * Omega * sin(latitude), in rad/s."""
    return 2.0 * OMEGA_EARTH * np.sin(np.radians(lat_deg))


def compute_swath_slope(
    ssha: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    swath_mask: np.ndarray,
    swath_name: str,
    max_gap_fraction: float = 0.25,
    min_gap_run_km: float = 10.0,
    compute_velocity: bool = True,
    min_abs_latitude_deg: float = 2.0,
) -> SwathSlopeResult:
    """
    Compute the along-track SSH slope (Stammer 1997 style, centered
    finite difference along the satellite ground track) for one KaRIn
    swath of a SWOT L2 LR Basic pass.

    Parameters
    ----------
    ssha : (num_lines, num_pixels) array
        SSH or SSHA field for the full pass (both swaths + nadir gap),
        NaN where invalid/flagged/land.
    latitude, longitude : (num_lines, num_pixels) arrays
        Geolocation, same shape as ssha.
    swath_mask : (num_pixels,) boolean array
        Pixel columns belonging to this swath (from split_left_right_swaths).
    swath_name : str
        "left" or "right".
    max_gap_fraction : float
        Used only when computing segment-level statistics: maximum
        allowed fraction of interpolated (originally NaN) samples within
        a segment/pixel column before that column is excluded from the
        segment's slope statistics.
    min_gap_run_km : float
        Any single run of consecutive NaNs longer than this distance is
        treated as land/an unbridgeable gap: slope is left as NaN across
        that run (and the points immediately adjacent to it, where a
        centered difference would otherwise quietly span the gap),
        rather than being linearly bridged. Shorter runs (isolated bad
        pixels, brief dropouts) are interpolated before differencing,
        consistent with Stammer's treatment of small data gaps in the
        single-satellite altimeter records he analyses.
    compute_velocity : bool
        If True, also compute the geostrophic velocity via
        v_g = (g/f) * slope (eq. 1, Stammer 1997, section 2).
    min_abs_latitude_deg : float
        Below this absolute latitude, f is too small for the
        geostrophic relation to be meaningful; velocity is set to NaN
        there (slope itself is still returned).

    Returns
    -------
    SwathSlopeResult
        Full-pass slope (and optional velocity) field for this swath,
        with along-track distance and geolocation metadata. Segment-level
        summary statistics are not filled in here; use
        compute_slope_segment_stats() for that (kept separate so the
        full-resolution field and the segment summary are independent,
        reusable products).
    """
    ssha = np.asarray(ssha, dtype=float)
    cols = np.where(swath_mask)[0]
    if cols.size == 0:
        raise ValueError(f"No pixels found for swath '{swath_name}'.")

    sub_ssha = ssha[:, cols]
    sub_lat = latitude[:, cols]
    sub_lon = longitude[:, cols]

    distance_km = along_track_distance_km(sub_lat, sub_lon)
    spacing_km = np.median(np.diff(distance_km))

    n_lines, n_pix = sub_ssha.shape
    slope = np.full_like(sub_ssha, np.nan)

    max_gap_samples = max(1, int(round(min_gap_run_km / spacing_km)))

    for c in range(n_pix):
        col = sub_ssha[:, c]
        valid = ~np.isnan(col)
        if valid.sum() < 3:
            continue

        # Identify NaN runs; only bridge (interpolate) runs shorter than
        # max_gap_samples. Longer runs (land, big dropouts) stay NaN and
        # block the centered difference across them.
        filled = col.copy()
        isnan = np.isnan(col)
        if isnan.any():
            run_start = None
            for i in range(n_lines):
                if isnan[i] and run_start is None:
                    run_start = i
                elif not isnan[i] and run_start is not None:
                    run_len = i - run_start
                    if run_len <= max_gap_samples and run_start > 0:
                        # bridgeable interior gap -> interpolate just this run
                        filled[run_start:i] = np.interp(
                            distance_km[run_start:i],
                            [distance_km[run_start - 1], distance_km[i]],
                            [col[run_start - 1], col[i]],
                        )
                    run_start = None
            if run_start is not None and run_start == 0:
                pass  # leading NaN run: cannot interpolate (no left anchor); leave as NaN

        # Centered finite difference for the SSH slope:
        #   delta_i = (zeta_{i+1} - zeta_{i-1}) / (x_{i+1} - x_{i-1})
        # which is the standard along-track centered slope estimator
        # used for single-track altimetric slope/geostrophic-velocity
        # estimation (Stammer 1997, sec. 2).
        d_zeta = filled[2:] - filled[:-2]          # meters (or SSH units)
        d_x_km = distance_km[2:] - distance_km[:-2]  # km
        d_x_m = d_x_km * 1000.0

        with np.errstate(invalid="ignore", divide="ignore"):
            s = d_zeta / d_x_m

        # any point whose value (or either neighbour used in the
        # difference) is still NaN -> slope undefined there
        still_nan = np.isnan(filled)
        bad = still_nan[2:] | still_nan[:-2] | (d_x_m <= 0)
        s[bad] = np.nan

        slope[1:-1, c] = s

    velocity = None
    if compute_velocity:
        f = _coriolis_parameter(sub_lat)
        with np.errstate(invalid="ignore", divide="ignore"):
            velocity = (GRAVITY / f) * slope
        velocity[np.abs(sub_lat) < min_abs_latitude_deg] = np.nan

    return SwathSlopeResult(
        swath=swath_name,
        along_track_distance_km=distance_km,
        latitude=sub_lat,
        longitude=sub_lon,
        slope=slope,
        geostrophic_velocity=velocity,
        segments=[],
    )


def compute_slope_segment_stats(
    slope_result: SwathSlopeResult,
    segment_length_km: float,
    overlap: float = 0.0,
    max_gap_fraction: float = 0.25,
    min_pixels_per_segment: int = 3,
) -> list:
    """
    Summarize the full-resolution slope field of one swath into
    along-track segment statistics (RMS slope, slope variance, and -- if
    available -- RMS geostrophic velocity), with latitude/segment
    metadata, analogous to the segment bookkeeping used for the
    wavenumber spectra (SegmentSpectrum in swot_alongtrack_spectra.py).

    These per-segment slope variances are directly comparable to the
    "equivalent slope variance" K_sl Stammer (1997) maps in his Fig. 2a,
    K_sl = KE * sin^2(phi) -- i.e. the eddy-kinetic-energy-equivalent
    quantity derived purely from along-track slope statistics without
    needing a dual-track (parallel) velocity estimate.

    Parameters
    ----------
    slope_result : SwathSlopeResult
        Output of compute_swath_slope().
    segment_length_km : float
        Along-track segment length, km.
    overlap : float
        Fractional overlap between consecutive segments (0 <= overlap < 1).
    max_gap_fraction : float
        Maximum allowed fraction of NaN (undefined) slope samples in a
        pixel column within a segment before that column is excluded.
    min_pixels_per_segment : int
        Minimum number of valid pixel columns required to keep a segment.

    Returns
    -------
    list[SlopeSegmentStats]
    """
    distance_km = slope_result.along_track_distance_km
    bounds = _segment_bounds(distance_km, segment_length_km, overlap=overlap)

    segments = []
    for seg_idx, (i0, i1) in enumerate(bounds):
        seg_slope = slope_result.slope[i0:i1, :]
        seg_lat = slope_result.latitude[i0:i1, :]
        seg_lon = slope_result.longitude[i0:i1, :]
        seg_vel = (slope_result.geostrophic_velocity[i0:i1, :]
                   if slope_result.geostrophic_velocity is not None else None)

        if seg_slope.shape[0] < 3:
            continue

        valid_fracs = []
        kept_slope_cols = []
        kept_vel_cols = [] if seg_vel is not None else None

        for c in range(seg_slope.shape[1]):
            col = seg_slope[:, c]
            valid = ~np.isnan(col)
            valid_frac = valid.mean() if len(col) else 0.0
            valid_fracs.append(valid_frac)
            if valid_frac < (1.0 - max_gap_fraction) or valid.sum() < 3:
                continue
            kept_slope_cols.append(col[valid])
            if seg_vel is not None:
                vcol = seg_vel[:, c]
                vvalid = ~np.isnan(vcol)
                if vvalid.any():
                    kept_vel_cols.append(vcol[vvalid])

        n_pixels_used = len(kept_slope_cols)
        if n_pixels_used < min_pixels_per_segment:
            continue

        all_slope = np.concatenate(kept_slope_cols)
        slope_mean = float(np.mean(all_slope))
        slope_var = float(np.var(all_slope))
        slope_rms = float(np.sqrt(np.mean(all_slope ** 2)))

        vel_rms = None
        if kept_vel_cols:
            all_vel = np.concatenate(kept_vel_cols)
            if all_vel.size:
                vel_rms = float(np.sqrt(np.nanmean(all_vel ** 2)))

        segments.append(SlopeSegmentStats(
            swath=slope_result.swath,
            segment_index=seg_idx,
            lat_mean=float(np.nanmean(seg_lat)),
            lat_min=float(np.nanmin(seg_lat)),
            lat_max=float(np.nanmax(seg_lat)),
            lon_mean=float(np.nanmean(seg_lon)),
            along_track_distance_start_km=float(distance_km[i0]),
            along_track_distance_end_km=float(distance_km[i1 - 1]),
            n_pixels_used=n_pixels_used,
            valid_fraction=float(np.mean(valid_fracs)) if valid_fracs else 0.0,
            slope_mean=slope_mean,
            slope_rms=slope_rms,
            slope_variance=slope_var,
            geostrophic_velocity_rms=vel_rms,
        ))

    return segments


def compute_pass_slope(
    ssha: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    cross_track_distance: np.ndarray,
    segment_length_km: Optional[float] = None,
    overlap: float = 0.0,
    max_gap_fraction: float = 0.25,
    min_gap_run_km: float = 10.0,
    compute_velocity: bool = True,
    min_pixels_per_segment: int = 3,
):
    """
    Top-level convenience function: compute the along-track SSH slope
    (and optional geostrophic velocity) for BOTH KaRIn swaths of a SWOT
    L2 LR Basic pass, handling land/NaN gaps and the nadir gap
    automatically (each swath is processed as a fully separate
    sub-array, so the gap never enters the slope computation).

    Parameters mirror compute_swath_slope() / compute_slope_segment_stats();
    see those functions' docstrings for full details.

    Returns
    -------
    dict with keys "left" and "right". Each value is a dict:
        {
            "field": SwathSlopeResult,         # full-resolution slope field
            "segments": list[SlopeSegmentStats]  # only if segment_length_km given, else []
        }
    """
    left_mask, right_mask = split_left_right_swaths(cross_track_distance)

    results = {}
    for name, mask in (("left", left_mask), ("right", right_mask)):
        field = compute_swath_slope(
            ssha=ssha,
            latitude=latitude,
            longitude=longitude,
            swath_mask=mask,
            swath_name=name,
            max_gap_fraction=max_gap_fraction,
            min_gap_run_km=min_gap_run_km,
            compute_velocity=compute_velocity,
        )
        segments = []
        if segment_length_km is not None:
            segments = compute_slope_segment_stats(
                field,
                segment_length_km=segment_length_km,
                overlap=overlap,
                max_gap_fraction=max_gap_fraction,
                min_pixels_per_segment=min_pixels_per_segment,
            )
        results[name] = {"field": field, "segments": segments}

    return results


# --------------------------------------------------------------------------- #
# Optional: slope wavenumber spectra (Stammer's Fig. 9 "from ... slope")
# --------------------------------------------------------------------------- #

def compute_swath_slope_spectrum(
    slope_result: SwathSlopeResult,
    segment_length_km: float,
    overlap: float = 0.0,
    max_gap_fraction: float = 0.25,
    detrend: str = "linear",
    window: str = "hann",
    min_pixels_per_segment: int = 3,
) -> PassSpectrumResult:
    """
    Compute the along-track wavenumber spectrum of the SSH *slope*
    field itself, reusing the exact same segmenting / windowing /
    periodogram machinery as the SSH spectra (compute_swath_spectra in
    swot_alongtrack_spectra.py). This reproduces Stammer (1997)'s
    practice of presenting SSH and slope wavenumber spectra side by side
    (his Figs. 8 vs 9): the slope spectrum is simply k^2 times the SSH
    spectrum (since slope = d(zeta)/dx implies a k^2 multiplication of
    the power spectral density), and is most sensitive at the smaller
    scales where SSH spectra are hard to interpret visually.

    Parameters
    ----------
    slope_result : SwathSlopeResult
        Output of compute_swath_slope().
    Other parameters: see compute_swath_spectra() in
        swot_alongtrack_spectra.py.

    Returns
    -------
    PassSpectrumResult (wavenumber in cycles/km, PSD in (slope units)^2 / (cycles/km))
    """
    n_pix = slope_result.slope.shape[1]
    full_mask = np.ones(n_pix, dtype=bool)
    return compute_swath_spectra(
        ssha=slope_result.slope,
        latitude=slope_result.latitude,
        longitude=slope_result.longitude,
        swath_mask=full_mask,
        swath_name=slope_result.swath,
        segment_length_km=segment_length_km,
        overlap=overlap,
        max_gap_fraction=max_gap_fraction,
        detrend=detrend,
        window=window,
        min_pixels_per_segment=min_pixels_per_segment,
    )


