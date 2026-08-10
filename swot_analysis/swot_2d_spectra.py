"""
Two-dimensional (along-track x cross-track) wavenumber spectra for SWOT
L2 LR SSH Unsmoothed granules.

This extends swot_alongtrack_spectra.py: it reuses that module's
segmentation, swath-splitting, and along-track-distance helpers, and
adds a genuine 2-D periodogram estimator S(kx, ky), Welch-averaged over
along-track segments. Unlike the 1-D pipeline (which averages a 1-D
along-track periodogram over cross-track pixel columns and throws away
cross-track wavenumber information), this keeps the cross-track
wavenumber ky, so the result can be used directly as S_0(kx, ky) in the
filter/aliasing prediction pipeline:

    S_filt(kx, ky) = |H(kx, ky)|^2 * S_0(kx, ky)
    S_1D^filt(kx)  = integral over ky of S_filt(kx, ky)
    S_1D^{2km, no-alias}(kx) = integral over ky of S_filt(kx, ky),
                               restricted to |kx| <= k_Ny^2km

where H(kx, ky) is the 2-D Hamming-filter transfer function used to go
from the 250-m Unsmoothed product to the 2-km Expert product.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy import signal

from swot_analysis.swot_alongtrack_spectra import (
    along_track_distance_km,
    split_left_right_swaths,
    _segment_bounds,
)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Segment2DSpectrum:
    """2-D spectral estimate and metadata for a single along-track segment."""

    swath: str                  # "left" or "right"
    segment_index: int
    wavenumber_x: np.ndarray    # along-track wavenumber, cpkm, 1-D (Nx,)
    wavenumber_y: np.ndarray    # cross-track wavenumber, cpkm, 1-D (Ny,)
    psd_2d: np.ndarray          # 2-D PSD, (SSH units)^2/(cpkm)^2, shape (Nx, Ny)
    n_columns_used: int         # cross-track pixel columns with real data
    n_columns_total: int        # total cross-track pixel columns in the swath
    lat_mean: float
    lat_min: float
    lat_max: float
    lon_mean: float
    along_track_distance_start_km: float
    along_track_distance_end_km: float
    valid_fraction: float       # fraction of non-NaN samples before fill
    gap_filled: bool            # whether any along- or cross-track gaps were filled
    month: int


@dataclasses.dataclass
class Pass2DSpectrumResult:
    """Full 2-D result for one pass/swath: per-segment spectra + the
    Welch-averaged mean 2-D spectrum, directly usable as S_0(kx, ky)."""

    swath: str
    wavenumber_x: np.ndarray            # common along-track axis (cpkm)
    wavenumber_y: np.ndarray            # common cross-track axis (cpkm)
    mean_psd_2d: np.ndarray             # shape (Nx, Ny), averaged over segments
    n_segments_used: int
    n_segments_total: int
    segments: list  # list[Segment2DSpectrum]
    month: int

    def segment_latitudes(self) -> np.ndarray:
        return np.array([s.lat_mean for s in self.segments])


# --------------------------------------------------------------------------- #
# Core 2-D spectral computation
# --------------------------------------------------------------------------- #

def compute_swath_spectra_2d(
    ssha: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    cross_track_distance: np.ndarray,
    swath_mask: np.ndarray,
    swath_name: str,
    segment_length_km: float,
    month: Optional[int] = None,
    along_track_spacing_km: Optional[float] = None,
    cross_track_spacing_km: Optional[float] = None,
    overlap: float = 0.0,
    max_gap_fraction: float = 0.25,
    max_column_gap_fraction: float = 0.5,
    detrend: str = "linear",
    detrend_cross_track: bool = True,
    window: str = "hann",
    cross_track_window: str = "hann",
) -> Pass2DSpectrumResult:
    """
    Compute the 2-D (along-track x cross-track) wavenumber spectrum
    S_0(kx, ky) for ONE swath (left or right) of a SWOT L2 LR
    Unsmoothed pass, Welch-averaged over along-track segments.

    This mirrors compute_swath_spectra() in swot_alongtrack_spectra.py
    (same segmentation, gap-tolerance, and detrending conventions) but
    keeps the full 2-D field per segment instead of collapsing to a
    per-column 1-D periodogram, so the cross-track wavenumber ky is
    retained.

    Parameters
    ----------
    ssha, latitude, longitude : (num_lines, num_pixels) arrays
        Same convention as compute_swath_spectra(): full-pass fields
        (both swaths + nadir gap), NaN where invalid/flagged.
    cross_track_distance : (num_pixels,) or (num_lines, num_pixels)
        Used both to select/order this swath's columns and to estimate
        the nominal cross-track pixel spacing (km) for the ky axis.
    swath_mask : (num_pixels,) boolean array
        Output of split_left_right_swaths(); columns belonging to this
        swath.
    segment_length_km : float
        Along-track segment length (Nx dimension), same trade-off as
        in the 1-D pipeline: longer segments -> better low-kx
        resolution, fewer independent segments to average.
    along_track_spacing_km, cross_track_spacing_km : float, optional
        Nominal sample spacing in km along each axis. If None,
        estimated from the data (median spacing), exactly as
        along_track_spacing_km is estimated in compute_swath_spectra().
    overlap : float
        Fractional overlap between consecutive along-track segments.
    max_gap_fraction : float
        Per-column along-track gap tolerance, identical role to its
        namesake in compute_swath_spectra(): a column with more than
        this fraction missing (or edge gaps beyond this tolerance) is
        left unfilled at this stage and patched by cross-track
        interpolation instead (see max_column_gap_fraction).
    max_column_gap_fraction : float
        Maximum fraction of cross-track columns in a segment that may
        need cross-track gap-filling before the *whole segment* is
        dropped. Cross-track pixels are not independently droppable
        the way along-track samples are (dropping a column would break
        the uniform ky grid needed for the FFT and for averaging
        segments together), so isolated bad columns are filled by
        linear interpolation across neighboring columns at each
        along-track row instead of being excluded.
    detrend : str or False
        Along-track detrending (axis=0), passed to scipy.signal.detrend.
    detrend_cross_track : bool
        If True, additionally remove the cross-track mean (axis=1,
        type="constant") per along-track row. Leaves the along-track
        trend removal (`detrend`) untouched; this only removes the
        cross-track-constant (DC in ky) component, analogous to
        removing a per-line bias.
    window : str
        Along-track taper (same role/default as compute_swath_spectra()).
    cross_track_window : str
        Cross-track taper, applied across the swath width. Hann by
        default to control cross-track spectral leakage from the finite
        50-km swath.

    Returns
    -------
    Pass2DSpectrumResult
        .mean_psd_2d has shape (len(wavenumber_x), len(wavenumber_y))
        and is the Welch-averaged S_0(kx, ky) estimate, directly usable
        in the filter/aliasing prediction pipeline.
    """
    cols = np.where(swath_mask)[0]
    if cols.size == 0:
        raise ValueError(f"No pixels found for swath '{swath_name}'.")

    sub_ssha = ssha[:, cols]
    sub_lat = latitude[:, cols]
    sub_lon = longitude[:, cols]

    # Cross-track distance -> nominal spacing (km) and column ordering.
    xt = np.asarray(cross_track_distance, dtype=float)
    if xt.ndim == 2:
        xt = np.nanmedian(xt, axis=0)
    sub_xt = xt[cols]

    order = np.argsort(sub_xt)
    if not np.array_equal(order, np.arange(sub_xt.size)):
        sub_ssha = sub_ssha[:, order]
        sub_lat = sub_lat[:, order]
        sub_lon = sub_lon[:, order]
        sub_xt = sub_xt[order]

    Ny = sub_xt.size
    if cross_track_spacing_km is None:
        diffs = np.diff(sub_xt)
        diffs = diffs[diffs > 0]
        cross_track_spacing_km = float(np.median(diffs)) if diffs.size else 0.25

    distance_km = along_track_distance_km(sub_lat, sub_lon)

    if along_track_spacing_km is None:
        diffs = np.diff(distance_km)
        diffs = diffs[diffs > 0]
        along_track_spacing_km = float(np.median(diffs)) if diffs.size else 0.25

    bounds = _segment_bounds(distance_km, segment_length_km, overlap=overlap)

    nperseg = int(round(segment_length_km / along_track_spacing_km))
    nperseg = max(nperseg, 8)

    wx = signal.get_window(window, nperseg)
    wy = signal.get_window(cross_track_window, Ny)
    taper = np.outer(wx, wy)
    win_norm = (wx ** 2).mean() * (wy ** 2).mean()

    min_columns_per_segment = max(1, int(np.ceil((1.0 - max_column_gap_fraction) * Ny)))

    segments: list = []
    for seg_idx, (i0, i1) in enumerate(bounds):
        seg_ssha = sub_ssha[i0:i1, :]
        seg_lat = sub_lat[i0:i1, :]
        seg_lon = sub_lon[i0:i1, :]
        seg_dist = distance_km[i0:i1]

        if seg_ssha.shape[0] < 8:
            continue

        uniform_dist = seg_dist[0] + np.arange(nperseg) * along_track_spacing_km
        if uniform_dist[-1] > seg_dist[-1] + along_track_spacing_km:
            continue

        # --- Step 1: along-track gap-fill, per column (same tolerance
        # logic as compute_swath_spectra) -----------------------------
        field = np.full((nperseg, Ny), np.nan)
        col_valid_frac = np.zeros(Ny)
        edge_tol_km = max_gap_fraction * segment_length_km

        for c in range(Ny):
            col = seg_ssha[:, c]
            valid = ~np.isnan(col)
            valid_frac = float(valid.mean()) if len(col) else 0.0
            col_valid_frac[c] = valid_frac

            if valid_frac < (1.0 - max_gap_fraction) or valid.sum() < 8:
                continue  # left as NaN -> patched cross-track below, or segment dropped

            good_dist = seg_dist[valid]
            good_val = col[valid]
            needs_left_km = max(0.0, good_dist[0] - uniform_dist[0])
            needs_right_km = max(0.0, uniform_dist[-1] - good_dist[-1])
            if needs_left_km > edge_tol_km or needs_right_km > edge_tol_km:
                continue

            field[:, c] = np.interp(uniform_dist, good_dist, good_val,
                                     left=good_val[0], right=good_val[-1])

        # --- Step 2: patch remaining fully-NaN columns by cross-track
        # linear interpolation, so the grid stays uniform for the FFT --
        col_ok = ~np.all(np.isnan(field), axis=0)
        n_columns_used = int(col_ok.sum())
        if n_columns_used < min_columns_per_segment:
            continue

        any_gap_filled = bool(np.any(col_valid_frac < 1.0))
        if not np.all(col_ok):
            col_idx = np.arange(Ny)
            good_idx = col_idx[col_ok]
            for r in range(nperseg):
                row = field[r, :]
                bad = ~col_ok
                row[bad] = np.interp(col_idx[bad], good_idx, row[good_idx])
            any_gap_filled = True

        # --- Step 3: detrend --------------------------------------------
        if detrend:
            field = signal.detrend(field, axis=0, type=detrend)
        if detrend_cross_track:
            field = signal.detrend(field, axis=1, type="constant")

        # --- Step 4: 2-D taper -------------------------------------------
        tapered = field * taper

        # --- Step 5: 2-D periodogram --------------------------------------
        # Density-scaled 2-D PSD (Parseval-consistent): integrating psd_2d
        # over (kx, ky) recovers the (windowed, bias-corrected) field
        # variance. win_norm corrects for power lost to tapering, exactly
        # as compute_swath_spectra() does for its 1-D window.
        Fxy = np.fft.fft2(tapered)
        psd = (along_track_spacing_km * cross_track_spacing_km) / (nperseg * Ny) \
            * np.abs(Fxy) ** 2
        psd = psd / win_norm

        kx = np.fft.fftshift(np.fft.fftfreq(nperseg, d=along_track_spacing_km))
        ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=cross_track_spacing_km))
        psd = np.fft.fftshift(psd, axes=(0, 1))

        segments.append(Segment2DSpectrum(
            swath=swath_name,
            segment_index=seg_idx,
            wavenumber_x=kx,
            wavenumber_y=ky,
            psd_2d=psd,
            n_columns_used=n_columns_used,
            n_columns_total=Ny,
            lat_mean=np.nanmean(seg_lat).item(),
            lat_min=np.nanmin(seg_lat).item(),
            lat_max=np.nanmax(seg_lat).item(),
            lon_mean=np.nanmean(seg_lon).item(),
            along_track_distance_start_km=seg_dist[0].item(),
            along_track_distance_end_km=seg_dist[-1].item(),
            valid_fraction=float(np.mean(col_valid_frac)),
            gap_filled=any_gap_filled,
            month=month,
        ))

    if segments:
        wavenumber_x = segments[0].wavenumber_x
        wavenumber_y = segments[0].wavenumber_y
        mean_psd_2d = np.mean(np.stack([s.psd_2d for s in segments], axis=0), axis=0)
    else:
        wavenumber_x = np.array([])
        wavenumber_y = np.array([])
        mean_psd_2d = np.array([])

    return Pass2DSpectrumResult(
        swath=swath_name,
        wavenumber_x=wavenumber_x,
        wavenumber_y=wavenumber_y,
        mean_psd_2d=mean_psd_2d,
        n_segments_used=len(segments),
        n_segments_total=len(bounds),
        segments=segments,
        month=month,
    )


def compute_pass_spectra_2d(
    ssha: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    cross_track_distance: np.ndarray,
    segment_length_km: float,
    month: Optional[int] = None,
    along_track_spacing_km: Optional[float] = None,
    cross_track_spacing_km: Optional[float] = None,
    overlap: float = 0.0,
    max_gap_fraction: float = 0.25,
    max_column_gap_fraction: float = 0.5,
    detrend: str = "linear",
    detrend_cross_track: bool = True,
    window: str = "hann",
    cross_track_window: str = "hann",
):
    """
    Compute 2-D (along-track x cross-track) wavenumber spectra
    S_0(kx, ky) for BOTH swaths of a SWOT L2 LR Unsmoothed pass.

    Direct 2-D analog of compute_pass_spectra() in
    swot_alongtrack_spectra.py. The returned mean_psd_2d for each swath
    is the empirical S_0(kx, ky) to multiply by |H(kx, ky)|^2 (the 2-D
    Hamming-filter transfer function) and integrate over ky to obtain
    the filtered / no-aliasing-predicted along-track spectrum, per swath
    or per cross-track bin (recommended given the swath's cross-track
    noise inhomogeneity -- run this per cross-track subset if you need
    the Fig. 2-style binning rather than a full-swath average).

    Returns
    -------
    dict with keys "left" and "right", each a Pass2DSpectrumResult.
    """
    left_mask, right_mask = split_left_right_swaths(cross_track_distance)

    results = {}
    for name, mask in (("left", left_mask), ("right", right_mask)):
        results[name] = compute_swath_spectra_2d(
            ssha=ssha,
            latitude=latitude,
            longitude=longitude,
            cross_track_distance=cross_track_distance,
            swath_mask=mask,
            swath_name=name,
            segment_length_km=segment_length_km,
            month=month,
            along_track_spacing_km=along_track_spacing_km,
            cross_track_spacing_km=cross_track_spacing_km,
            overlap=overlap,
            max_gap_fraction=max_gap_fraction,
            max_column_gap_fraction=max_column_gap_fraction,
            detrend=detrend,
            detrend_cross_track=detrend_cross_track,
            window=window,
            cross_track_window=cross_track_window,
        )
    return results
