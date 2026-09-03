from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np
from scipy import signal
from pathlib import Path
try:
    import xarray as xr
except ImportError:  # xarray is optional - only needed for the file loader
    xr = None


EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class SegmentSpectrum:
    """Spectral estimate and metadata for a single along-track segment."""

    swath: str                  # "left" or "right"
    segment_index: int          # index of this segment along the pass
    wavenumber: np.ndarray      # cycles per km (1-D, length = nperseg//2+1)
    psd: np.ndarray             # PSD in (SSH units)^2 / (cycles/km), 1-D
    n_pixels_used: int          # number of cross-track pixels averaged
    lat_mean: float
    lat_min: float
    lat_max: float
    lon_mean: float
    along_track_distance_start_km: float
    along_track_distance_end_km: float
    valid_fraction: float       # fraction of non-NaN samples before fill
    gap_filled: bool            # whether interior NaNs were interpolated
    month: int                  # month associated with swath

    # --- Optional 2-D (along-track x cross-track), cross-track-integrated
    # spectrum -- only populated when compute_swath_spectra() /
    # compute_pass_spectra() is called with cross_track_integrated=True.
    # This is computed differently from `psd` above (which averages
    # independent 1-D column periodograms): here a single 2-D periodogram
    # is taken over the whole (along-track x cross-track) segment patch
    # and integrated over cross-track wavenumber, the same recipe used
    # for the "total" (direct + aliased) spectrum in
    # swot_alias_spectra.py, at native (undecimated) resolution -- so
    # this is the field to use when comparing against that script's
    # output rather than `psd`.
    wavenumber_2d: Optional[np.ndarray] = None
    psd_2d: Optional[np.ndarray] = None
    n_cols_2d: Optional[int] = None    # cross-track columns in the 2-D patch after gap-filling
    dx2_km_2d: Optional[float] = None  # cross-track spacing used for the 2-D patch


@dataclasses.dataclass
class PassSpectrumResult:
    """Full result for one pass: per-segment spectra + mean spectrum per swath."""

    swath: str
    wavenumber: np.ndarray              # common wavenumber axis (cycles/km)
    mean_psd: np.ndarray                # pass-mean PSD, averaged over segments
    n_segments_used: int
    n_segments_total: int
    segments: list  # list[SegmentSpectrum]
    month: int      # month associated with swath

    # --- Optional 2-D cross-track-integrated pass-mean spectrum; see
    # SegmentSpectrum.psd_2d. None unless cross_track_integrated=True was
    # passed to compute_swath_spectra() / compute_pass_spectra().
    wavenumber_2d: Optional[np.ndarray] = None
    mean_psd_2d: Optional[np.ndarray] = None
    n_segments_used_2d: int = 0

    def segment_latitudes(self) -> np.ndarray:
        """Convenience: array of mean latitude for each retained segment."""
        return np.array([s.lat_mean for s in self.segments])


# --------------------------------------------------------------------------- #
# Loading helper (optional convenience; not required if you already have
# numpy arrays of ssha/lat/lon/cross_track_distance)
# --------------------------------------------------------------------------- #

def load_swot_l2(filepath: str, ssh_var: str = "ssha_karin_2", hret: bool = True):
    """
    Load the variables needed for spectral analysis from a SWOT L2 LR
    Basic netCDF granule.

    Parameters
    ----------
    filepath : str
        Path to a SWOT_L2_LR_SSH_*_Basic*.nc file.
    ssh_var : str
        Name of the SSH(A) variable to extract. For the Expert product
        this is typically "ssha_karin_2" (KaRIn SSHA with the
        recommended editing/corrections already applied) or
        "ssh_karin_2". Use "ssha_karin" / "ssh_karin" for the
        non-default variants if preferred.
    hret : bool
        Include or exclude coherent internal tides from HRET. 
        Default is True to include internal tide (i.e. adding it to SWOT products).
        It only applies if ssh_var is ssha.
    Returns
    -------
    dict with keys:
        ssha : (num_lines, num_pixels) float array, NaN where invalid
               or flagged by quality flags / land.
        latitude, longitude : (num_lines, num_pixels) float arrays
        cross_track_distance : (num_pixels,) or (num_lines, num_pixels)
               float array, negative on the left swath, positive on the
               right swath, NaN/0 over the nadir gap.
    """
    if xr is None:
        raise ImportError("xarray is required for load_swot_l2_basic(); "
                           "install it, or build the input arrays yourself "
                           "and call compute_pass_spectra() directly.")

    ds = xr.open_dataset(filepath)

    ssha = ds[ssh_var]+ds['height_cor_xover']#.values.astype(float)
    if ssh_var=='ssha_karin_2':
        if hret==True:
            ssha = ssha + ds['internal_tide_hret']
    else:
        pass
    ssha = ssha.values.astype(float)
    # Mask out flagged data using the associated quality flag, if present.
    qual_var = ssh_var.replace("ssha", "ssha").replace("ssh_karin_2", "ssh_karin_2")
    for cand in (f"{ssh_var}_qual", "ssha_karin_2_qual", "ssh_karin_2_qual"):
        if cand in ds.variables:
            qual = ds[cand].values
            ssha = np.where(qual == 0, ssha, np.nan)
            break

    lat = ds["latitude"].values.astype(float)
    lon = ds["longitude"].values.astype(float)

    if "cross_track_distance" in ds.variables:
        xtrack = ds["cross_track_distance"].values.astype(float)
    else:
        raise KeyError("cross_track_distance variable not found; needed to "
                        "separate left/right swaths and exclude the nadir gap.")

    ds.close()
    return {
        "ssha": ssha,
        "latitude": lat,
        "longitude": lon,
        "cross_track_distance": xtrack,
    }


def load_swot_l2_expert(filepath, ssh_var='ssha_karin_2', HRET=True, swh_var=False):
    # NO type annotations — avoids Python 3.14 PEP 649 __annotate__ capture bug
    ds = xr.open_dataset(filepath)
    if HRET:
        #ssha   = np.array(ds[ssh_var] + ds['height_cor_xover'] + ds['internal_tide_hret'], dtype='float64')
        ssha = (ds[ssh_var] + ds['height_cor_xover'] + ds['internal_tide_hret']).values
        #ssha = (ds[ssh_var] + ds['height_cor_xover']+ ds['internal_tide_hret']).values#.copy()
    else:
        ssha   = np.array(ds[ssh_var] + ds['height_cor_xover'], dtype='float64')
        #ssha = (ds[ssh_var] + ds['height_cor_xover']).values#.copy()
    #ssha = ssha.astype('float64')   # string dtype, no builtins involved

    for qual_name in (f'{ssh_var}_qual', 'ssha_karin_2_qual', 'ssh_karin_2_qual'):
        if qual_name in ds.variables:
            ssha = np.where(np.array(ds[qual_name]) == 0, ssha, np.nan)
            #ssha = np.where(ds[qual_name].values == 0, ssha, np.nan)
            break
    for scf in ('ancillary_surface_classification_flag', 'surface_classification_flag'):
        if scf in ds.variables:
            ssha = np.where(np.array(ds[scf]) == 0, ssha, np.nan)
            break
    if 'surface_type' in ds.variables:
        ssha = np.where(np.array(ds['surface_type']) == 0, ssha, np.nan)
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    xtrack = ds['cross_track_distance'].values
    
    if swh_var:
        swh = np.where(np.array(ds['swh_karin_qual'])==0, np.array(ds['swh_karin']), np.nan)
        ssha = np.where(swh<=5,ssha,np.nan)
        
    ds.close()
    return {'ssha': ssha, 'latitude': lat, 'longitude': lon,
                'cross_track_distance': xtrack}
        

def load_swot_l2_unsmoothed(filepath, ssh_var='ssha_karin_2', HRET=True):
    """
    Load the variables needed for spectral analysis from a SWOT L2 LR
    Unsmoothed granule. Unlike the Expert product, Unsmoothed data is
    split across two netCDF groups ('left' and 'right'); this function
    merges them into the same flat (num_lines, num_pixels) layout used
    by load_swot_l2_expert, so downstream code (compute_pass_spectra,
    split_left_right_swaths) needs no changes.
    """
    def _load_group(group_name, flip):
        ds = xr.open_dataset(filepath, group=group_name)

        if HRET:
            ssha = (ds[ssh_var] + ds['height_cor_xover'] + ds['internal_tide_hret']).values
        else:
            ssha = np.array(ds[ssh_var] + ds['height_cor_xover'])

        for qual_name in (f'{ssh_var}_qual', 'ssha_karin_2_qual', 'ssh_karin_2_qual'):
            if qual_name in ds.variables:
                ssha = np.where(np.array(ds[qual_name]) == 0, ssha, np.nan)
                break
        for scf in ('ancillary_surface_classification_flag', 'surface_classification_flag'):
            if scf in ds.variables:
                ssha = np.where(np.array(ds[scf]) == 0, ssha, np.nan)
                break
        if 'surface_type' in ds.variables:
            ssha = np.where(np.array(ds['surface_type']) == 0, ssha, np.nan)

        lat = ds['latitude'].values
        lon = ds['longitude'].values
        xtrack = np.array(ds['cross_track_distance'], dtype='float64')
        # sign convention: negative = left swath, positive = right swath,
        # matching what split_left_right_swaths() expects downstream
        sign = -1.0 if group_name == 'left' else 1.0
        xtrack = sign * np.abs(xtrack)

        ds.close()

        if flip:
            ssha = ssha[:, ::-1]
            lat = lat[:, ::-1]
            lon = lon[:, ::-1]
            xtrack = xtrack[:, ::-1] if xtrack.ndim == 2 else xtrack[::-1]

        return ssha, lat, lon, xtrack

    # left group's cross-track index runs right-to-left -> flip it so both
    # groups increase with distance from nadir in the same direction
    ssha_l, lat_l, lon_l, xtrack_l = _load_group('left', flip=True)
    ssha_r, lat_r, lon_r, xtrack_r = _load_group('right', flip=False)

    ssha = np.concatenate([ssha_l, ssha_r], axis=1)
    lat = np.concatenate([lat_l, lat_r], axis=1)
    lon = np.concatenate([lon_l, lon_r], axis=1)
    xtrack = np.concatenate([xtrack_l, xtrack_r], axis=1) if xtrack_l.ndim == 2 \
        else np.concatenate([xtrack_l, xtrack_r])

    return {'ssha': ssha, 'latitude': lat, 'longitude': lon,
            'cross_track_distance': xtrack}
# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def along_track_distance_km(lat: np.ndarray, lon: np.ndarray,
                             axis: int = 0) -> np.ndarray:
    """
    Cumulative along-track distance (km) computed along `axis`, using a
    representative cross-track index (median pixel column) so that a
    single along-track distance axis can be shared by all pixel columns
    of a swath.

    lat, lon : (num_lines, num_pixels) arrays (NaNs allowed away from the
               reference column).
    """
    if lat.ndim == 1:
        ref_lat, ref_lon = lat, lon
    else:
        ref_col = lat.shape[1] // 2
        ref_lat = lat[:, ref_col]
        ref_lon = lon[:, ref_col]
        # fall back to nearest valid column if the reference is empty
        if np.all(np.isnan(ref_lat)):
            valid_cols = [c for c in range(lat.shape[1])
                          if not np.all(np.isnan(lat[:, c]))]
            if not valid_cols:
                raise ValueError("No valid latitude data found to build "
                                  "along-track distance axis.")
            ref_col = valid_cols[len(valid_cols) // 2]
            ref_lat = lat[:, ref_col]
            ref_lon = lon[:, ref_col]

    # interpolate over any internal NaNs in the reference track so the
    # distance axis itself is always well defined. Longitude uses a
    # wrap-safe fill (see _interp_nan_lon_1d) since SWOT longitudes are
    # 0-360 and a NaN can coincide with a prime-meridian crossing.
    ref_lat = _interp_nan_1d(ref_lat)
    ref_lon = _interp_nan_lon_1d(ref_lon)

    d = np.zeros_like(ref_lat)
    d[1:] = _haversine_km(ref_lat[:-1], ref_lon[:-1], ref_lat[1:], ref_lon[1:])
    return np.cumsum(d)


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate interior NaNs in a 1-D array; edge NaNs are
    filled with the nearest valid value (no extrapolation of slope)."""
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    idx = np.arange(n)
    good = ~np.isnan(x)
    if good.sum() == 0:
        return x
    if good.sum() < n:
        x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def _interp_nan_lon_1d(lon: np.ndarray) -> np.ndarray:
    """
    Same purpose as `_interp_nan_1d`, but wrap-safe for longitude.

    Longitude is circular (SWOT products use the 0-360 convention, so a
    swath crossing the prime meridian goes ...359.8, 359.9, 0.0, 0.1...).
    Plain `np.interp` on the raw values has no notion of that wrap: if a
    NaN happens to fall exactly at a 0/360 crossing, it interpolates the
    "long way around" through ~180 degrees instead of the true, short
    step through 0/360 -- injecting a large, spurious position error
    exactly at the crossing. Interpolating the (cos, sin) unit-circle
    representation instead sidesteps the wrap entirely (and works
    whether the input happens to use the 0-360 or -180-180 convention).
    """
    lon = np.asarray(lon, dtype=float)
    n = len(lon)
    idx = np.arange(n)
    good = ~np.isnan(lon)
    if good.sum() == 0 or good.sum() == n:
        return lon.copy()

    rad = np.radians(lon[good])
    c = np.interp(idx, idx[good], np.cos(rad))
    s = np.interp(idx, idx[good], np.sin(rad))

    filled = lon.copy()
    filled[~good] = np.degrees(np.arctan2(s[~good], c[~good])) % 360.0
    return filled


def _circular_mean_lon_deg(lon: np.ndarray) -> float:
    """
    Mean longitude, correctly handling the 0/360 wrap (average of unit
    vectors, i.e. circular mean) instead of a plain arithmetic mean.
    `np.nanmean([359.8, 0.2])` gives 180.0 -- exactly wrong, since the
    true midpoint is ~0.0/360.0. Any segment whose along-track pixels
    straddle the prime meridian gets a wildly wrong lon_mean under the
    naive mean, which is very likely what you're seeing as a "hole" at
    0 deg (those segments effectively vanish from their true location in
    anything binned/plotted by lon_mean) with data seemingly displaced
    elsewhere.
    """
    lon = np.asarray(lon, dtype=float)
    valid = lon[~np.isnan(lon)]
    if valid.size == 0:
        return float("nan")
    rad = np.radians(valid)
    mean_angle = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    return float(np.degrees(mean_angle) % 360.0)


# --------------------------------------------------------------------------- #
# Swath splitting
# --------------------------------------------------------------------------- #

def split_left_right_swaths(cross_track_distance: np.ndarray):
    """
    Return boolean column masks (left_mask, right_mask) over the
    cross-track (pixel) dimension, identifying which pixel columns
    belong to the left swath (negative cross-track distance), the right
    swath (positive), and implicitly excluding the nadir-gap columns
    (NaN or exactly 0 cross-track distance), which fall into neither
    mask.

    cross_track_distance may be 1-D (num_pixels,) or 2-D
    (num_lines, num_pixels); if 2-D, the per-pixel sign is taken from
    the median across lines (the swath geometry is essentially constant
    along track).
    """
    xt = np.asarray(cross_track_distance, dtype=float)
    if xt.ndim == 2:
        xt = np.nanmedian(xt, axis=0)

    left_mask = xt < 0
    right_mask = xt > 0
    return left_mask, right_mask


# --------------------------------------------------------------------------- #
# Core spectral computation
# --------------------------------------------------------------------------- #

def _segment_bounds(distance_km: np.ndarray, segment_length_km: float,
                     overlap: float = 0.0):
    """Yield (i_start, i_end) index pairs splitting `distance_km` into
    along-track segments of length `segment_length_km`, with optional
    fractional overlap (0 <= overlap < 1).

    A final segment is always anchored to the exact end of the track (in
    addition to the regularly-stepped segments), so that up to
    `segment_length_km` of data at the tail of every pass is not silently
    dropped. Without this, since essentially every SWOT granule starts its
    along-track distance count at very nearly the same orbital turning
    latitude, the always-discarded tail lands at the same absolute
    latitude for every file -- which shows up as a solid, uncovered
    latitude band once results are aggregated across many
    passes/granules, growing wider as segment_length_km increases.
    """
    if (0.0 <= overlap < 1.0):
        step_km = segment_length_km * (1.0 - overlap)
    else:
        raise ValueError("overlap must be in [0, 1).")
    total = distance_km[-1]
    starts_km = list(np.arange(distance_km[0], total - segment_length_km + 1e-9, step_km))
    if len(starts_km) == 0:
        # pass shorter than one segment: use whole pass as a single segment
        starts_km = [distance_km[0]]

    # Anchor one final segment to the literal end of the track, even if it
    # overlaps the previous segment more than the nominal step -- this
    # guarantees full start-to-end coverage regardless of segment_length_km.
    last_start_needed = total - segment_length_km
    if last_start_needed > starts_km[-1] + 1e-9:
        starts_km.append(last_start_needed)

    bounds = []
    seen_i1 = set()
    for s_km in starts_km:
        e_km = s_km + segment_length_km
        i0 = int(np.searchsorted(distance_km, s_km, side="left"))
        i1 = int(np.searchsorted(distance_km, e_km, side="right"))
        if i1 - i0 < 8:  # need a minimum number of samples to FFT meaningfully
            continue
        if i1 in seen_i1:  # avoid a near-duplicate segment when already reaching the end
            continue
        seen_i1.add(i1)
        bounds.append((i0, i1))
    return bounds


# --------------------------------------------------------------------------- #
# Optional: 2-D (along-track x cross-track), cross-track-integrated PSD
# --------------------------------------------------------------------------- #
#
# The default computation above (`psd` / `mean_psd`) estimates the
# along-track spectrum by taking an independent 1-D periodogram of each
# cross-track pixel column and averaging over columns. That is NOT the
# same estimator as swot_alias_spectra.py's `total_psd`, which instead
# takes a single 2-D periodogram of the whole (along-track x cross-track)
# patch and integrates over cross-track wavenumber. The two give similar
# but not identical results (the 2-D estimator, via the cross-track Hann
# taper, correlates information across columns rather than treating them
# as independent realizations), so a like-for-like comparison against the
# alias-decomposition script's spectra needs this second estimator, which
# these helpers add as an opt-in (see `cross_track_integrated` below).

def _fill_nan_2d(patch: np.ndarray, max_nan_fraction: float = 0.15):
    """
    Fill small 2-D NaN gaps using row-then-column linear interpolation.
    Returns None if the patch exceeds max_nan_fraction, or if any NaNs
    remain after filling (e.g. an entire row/column with no valid
    neighbours) -- mirrors `swot_alias_spectra._fill_nan_2d`.
    """
    patch = np.asarray(patch, dtype=float).copy()

    if np.isnan(patch).mean() > max_nan_fraction:
        return None

    for i in range(patch.shape[0]):
        row = patch[i]
        if np.isnan(row).any() and not np.isnan(row).all():
            patch[i] = _interp_nan_1d(row)

    for j in range(patch.shape[1]):
        col = patch[:, j]
        if np.isnan(col).any() and not np.isnan(col).all():
            patch[:, j] = _interp_nan_1d(col)

    if np.isnan(patch).any():
        return None

    return patch


def _plane_detrend_2d(patch: np.ndarray) -> np.ndarray:
    """Remove a best-fit 2-D plane (mirrors swot_alias_spectra.plane_detrend_2d)."""
    n1, n2 = patch.shape
    y, x = np.mgrid[0:n1, 0:n2]
    A = np.column_stack([np.ones(n1 * n2), x.ravel(), y.ravel()])
    coeffs, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
    return patch - (A @ coeffs).reshape(n1, n2)


def _periodogram_2d_cross_track_integrated(
    patch: np.ndarray,
    dx1_km: float,
    dx2_km: float,
    window: str = "hann",
):
    """
    2-D Hann(-or-other)-tapered periodogram of `patch`, integrated over
    cross-track wavenumber to give a 1-D along-track PSD. Same recipe as
    `swot_alias_spectra.periodogram_2d` followed by its cross-track
    integration step, without any Expert-resolution filtering/aliasing
    decomposition -- i.e. this is the native-resolution "total" spectrum
    swot_alias_spectra.py would call `total_psd`.

    Returns
    -------
    k1 : one-sided along-track wavenumber (cycles/km)
    psd_1d : cross-track-integrated along-track PSD, (SSH units)^2/(cycles/km)
    """
    n1, n2 = patch.shape

    w1 = signal.get_window(window, n1)
    w2 = signal.get_window(window, n2)
    win2d = np.outer(w1, w2)
    norm = np.mean(win2d ** 2)

    F = np.fft.fft2(patch * win2d)
    S2D = (dx1_km * dx2_km / (n1 * n2)) * np.abs(F) ** 2 / norm

    k1 = np.fft.fftfreq(n1, dx1_km)
    k2 = np.fft.fftfreq(n2, dx2_km)
    dk2 = np.abs(k2[1] - k2[0]) if n2 > 1 else 1.0

    total_1d = S2D.sum(axis=1) * dk2

    positive = k1 >= 0
    k = k1[positive]
    total = total_1d[positive].copy()

    # DC and Nyquist are not doubled when folding to one-sided.
    interior = (k > 0) & (k < np.max(np.abs(k1)))
    total[interior] *= 2.0

    return k, total


def _estimate_dx2_km(cross_track_distance: Optional[np.ndarray], cols: np.ndarray):
    """
    Estimate the (approximately uniform) cross-track pixel spacing in km
    for the given swath columns, from the cross_track_distance array.
    Returns None if it can't be estimated (caller should fall back to an
    explicit dx2_km or skip the 2-D computation).
    """
    if cross_track_distance is None:
        return None

    xt = np.asarray(cross_track_distance, dtype=float)
    if xt.ndim == 2:
        xt = np.nanmedian(xt, axis=0)

    sub = xt[cols]
    sub = sub[np.isfinite(sub)]
    if sub.size < 2:
        return None

    diffs = np.abs(np.diff(np.sort(sub)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None

    return float(np.median(diffs))


def compute_swath_spectra(
    ssha: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    swath_mask: np.ndarray,
    swath_name: str,
    segment_length_km: float,
    month: Optional[int] = None,
    along_track_spacing_km: Optional[float] = None,
    overlap: float = 0.0,
    max_gap_fraction: float = 0.25,
    detrend: str = "linear",
    window: str = "hann",
    min_pixels_per_segment: int = 3,
    cross_track_integrated: bool = False,
    cross_track_distance: Optional[np.ndarray] = None,
    dx2_km: Optional[float] = None,
    max_nan_fraction_2d: float = 0.15,
    remove_plane_2d: bool = True,
    window_2d: str = "hann",
) -> PassSpectrumResult:
    """
    Compute along-track wavenumber spectra for ONE swath (left or right)
    of a SWOT L2 LR Basic pass.

    Parameters
    ----------
    ssha : (num_lines, num_pixels) array
        SSH or SSHA field for the full pass (both swaths + gap), NaN
        where invalid/flagged.
    latitude, longitude : (num_lines, num_pixels) arrays
        Geolocation, same shape as ssha.
    swath_mask : (num_pixels,) boolean array
        Pixel columns belonging to this swath (output of
        split_left_right_swaths).
    month : (num_pixels,) int
    swath_name : str
        "left" or "right" (for bookkeeping only).
    segment_length_km : float
        Length of each along-track segment in km. This is the key
        resolution/statistics trade-off parameter: longer segments give
        better low-wavenumber resolution but fewer independent segments
        to average.
    along_track_spacing_km : float, optional
        Nominal along-track sample spacing in km. If None, it is
        estimated from the data (median spacing of the along-track
        distance axis). Needed to convert FFT bin index to a physical
        wavenumber (cycles/km) and to resample onto a uniform grid if
        the native sampling is irregular.
    overlap : float
        Fractional overlap between consecutive segments (0 <= overlap < 1).
    max_gap_fraction : float
        Maximum allowed fraction of NaN samples within a segment
        (per pixel column) before that column is excluded from the
        segment average; if too few columns remain valid
        (< min_pixels_per_segment) the whole segment is dropped. This is
        how land, the unsampled nadir-adjacent pixels, and isolated
        data gaps are handled.
    detrend : str
        Passed to scipy.signal detrending ("linear", "constant", or
        False).
    window : str
        Taper window name (passed to scipy.signal.get_window), default
        Hann, standard in the SWOT cal/val literature.
    min_pixels_per_segment : int
        Minimum number of valid cross-track pixel columns required to
        keep a segment.
    cross_track_integrated : bool
        If True, ALSO compute a 2-D (along-track x cross-track),
        cross-track-integrated PSD per segment (`SegmentSpectrum.psd_2d`
        / `PassSpectrumResult.mean_psd_2d`), using the same 2-D
        periodogram + cross-track-integration recipe as the "total"
        (native-resolution) spectrum in swot_alias_spectra.py -- this is
        the field to use for a like-for-like comparison against that
        script's output, rather than the default `psd`/`mean_psd`
        (which averages independent 1-D column periodograms and is a
        different, if related, estimator). Off by default: it is
        additional computation on top of the default along-track
        spectrum, not a replacement for it.
    cross_track_distance : array, optional
        Needed only when cross_track_integrated=True and dx2_km is not
        given explicitly -- used to estimate the physical cross-track
        pixel spacing for this swath.
    dx2_km : float, optional
        Cross-track pixel spacing in km, used only when
        cross_track_integrated=True. If None, estimated from
        cross_track_distance.
    max_nan_fraction_2d : float
        Total NaN-fraction threshold for the 2-D segment patch (analogous
        to swot_alias_spectra's max_nan_fraction); only used when
        cross_track_integrated=True. Segments exceeding this have no
        psd_2d (left as None) but are otherwise unaffected -- the default
        1-D `psd` for that segment is still computed as usual.
    remove_plane_2d : bool
        Remove a best-fit 2-D plane from the segment patch before the 2-D
        periodogram, matching swot_alias_spectra's default
        (remove_plane=True). Only used when cross_track_integrated=True.
    window_2d : str
        Taper window (both dimensions) for the 2-D periodogram. Only used
        when cross_track_integrated=True.

    Returns
    -------
    PassSpectrumResult
    """
    #ssha = np.asarray(ssha, dtype=np.float64)
    cols = np.where(swath_mask)[0]
    if cols.size == 0:
        raise ValueError(f"No pixels found for swath '{swath_name}'.")

    sub_ssha = ssha[:, cols]
    sub_lat = latitude[:, cols]
    sub_lon = longitude[:, cols]

    dx2_km_resolved = dx2_km
    if cross_track_integrated and dx2_km_resolved is None:
        dx2_km_resolved = _estimate_dx2_km(cross_track_distance, cols)
        if dx2_km_resolved is None:
            raise ValueError(
                "cross_track_integrated=True requires either dx2_km or a "
                "usable cross_track_distance to estimate it from, and "
                f"neither was usable for swath '{swath_name}'."
            )

    distance_km = along_track_distance_km(sub_lat, sub_lon)

    if along_track_spacing_km is None:
        diffs = np.diff(distance_km)
        diffs = diffs[diffs > 0]
        along_track_spacing_km = np.float64(np.median(diffs)) if diffs.size else 2.0

    bounds = _segment_bounds(distance_km, segment_length_km, overlap=overlap)

    # nperseg: convert segment length to sample count using the nominal
    # spacing, but in practice we just use the index range from
    # _segment_bounds and resample to a fixed length so all segments
    # share an identical, common wavenumber axis.
    nperseg = int(round(segment_length_km / along_track_spacing_km))
    nperseg = max(nperseg, 8)

    win = signal.get_window(window, nperseg)

    segments: list = []
    for seg_idx, (i0, i1) in enumerate(bounds):
        seg_ssha = sub_ssha[i0:i1, :]
        seg_lat = sub_lat[i0:i1, :]
        seg_lon = sub_lon[i0:i1, :]
        seg_dist = distance_km[i0:i1]

        if seg_ssha.shape[0] < 8:
            continue

        # Resample each column onto a uniform along-track grid of length
        # nperseg covering [seg_dist[0], seg_dist[0] + segment_length_km).
        uniform_dist = (seg_dist[0]
                         + np.arange(nperseg) * along_track_spacing_km)
        # discard segment if the uniform grid runs past available data
        #if uniform_dist[-1] > seg_dist[-1] + along_track_spacing_km:
        #    continue
        
        # was: along_track_spacing_km
        overrun_tol_km = max(along_track_spacing_km * 10, 0.01 * segment_length_km)
        if uniform_dist[-1] > seg_dist[-1] + overrun_tol_km:
            continue
        col_psds = []
        any_gap_filled = False
        valid_fracs = []

        patch_2d = (
            np.full((nperseg, seg_ssha.shape[1]), np.nan)
            if cross_track_integrated else None
        )

        for c in range(seg_ssha.shape[1]):
            col = seg_ssha[:, c]
            valid = ~np.isnan(col)
            valid_frac = float(valid.mean()) if len(col) else 0.0
            valid_fracs.append(valid_frac)

            if valid_frac < (1.0 - max_gap_fraction) or valid.sum() < 8:
                continue  # too many gaps / land in this column -> skip it

            # Interpolate this column (in native sampling) onto the uniform
            # grid. Interior gaps are filled by linear interpolation.
            # Edge gaps (e.g. a coastline sitting right at one end of a
            # long segment) are tolerated up to max_gap_fraction of the
            # segment length: rather than discarding the whole column (and
            # cascading to the whole segment) over a modest coastal margin,
            # the edge is filled with the nearest valid value. Without
            # this, large segment_length_km values make ANY segment that
            # touches land at its very edge unusable in its entirety, even
            # when the overwhelming majority of the segment is valid open
            # ocean -- this disproportionately removes data at latitudes
            # where coastlines commonly sit close to open ocean (e.g. the
            # subtropical desert coasts around 20-30 deg latitude), and
            # gets worse the longer the segment is.
            good_dist = seg_dist[valid]
            good_val = col[valid]
            edge_tol_km = max_gap_fraction * segment_length_km
            needs_left_km = max(0.0, good_dist[0] - uniform_dist[0])
            needs_right_km = max(0.0, uniform_dist[-1] - good_dist[-1])
            if needs_left_km > edge_tol_km or needs_right_km > edge_tol_km:
                # missing data at the edge exceeds the tolerance -> skip column
                continue

            #resampled = np.interp(uniform_dist, good_dist, good_val,
            #                       left=good_val[0], right=good_val[-1])
            resampled = np.interp(uniform_dist, good_dist, good_val,
                       left=good_val[0], right=good_val[-1])

            if patch_2d is not None:
                # Raw (non-detrended) resampled column -- the 2-D patch
                # gets a single plane-detrend applied once, over the
                # whole patch, after gap-filling (see below), rather
                # than per-column detrending, to match
                # swot_alias_spectra's convention.
                patch_2d[:, c] = resampled

            if valid_frac < 1.0:
                any_gap_filled = True

            resampled = signal.detrend(resampled, type=detrend) if detrend else resampled
            tapered = resampled * win

            freqs, pxx = signal.periodogram(
                tapered, fs=1.0 / along_track_spacing_km,
                window="boxcar",  # window already applied manually above
                detrend=False, scaling="density",
            )
            # correct for window power loss (since we applied `win`
            # ourselves rather than letting periodogram do it, so we can
            # reuse the exact same window energy normalisation here)
            win_norm = (win ** 2).mean()
            pxx = pxx / win_norm

            col_psds.append(pxx)

        n_pixels_used = len(col_psds)
        if n_pixels_used < min_pixels_per_segment:
            continue

        mean_pxx = np.mean(np.stack(col_psds, axis=0), axis=0)

        wavenumber_2d = None
        psd_2d = None
        n_cols_2d = None

        if cross_track_integrated:
            filled = _fill_nan_2d(patch_2d, max_nan_fraction=max_nan_fraction_2d)

            if filled is not None:
                if remove_plane_2d:
                    filled = _plane_detrend_2d(filled)

                wavenumber_2d, psd_2d = _periodogram_2d_cross_track_integrated(
                    filled,
                    dx1_km=along_track_spacing_km,
                    dx2_km=dx2_km_resolved,
                    window=window_2d,
                )
                n_cols_2d = filled.shape[1]
            # else: patch had too many/too large NaN gaps for the 2-D
            # estimator specifically -- psd_2d stays None for this
            # segment, but the default 1-D `psd` above is unaffected.

        segments.append(SegmentSpectrum(
            swath=swath_name,
            segment_index=seg_idx,
            wavenumber=freqs,
            psd=mean_pxx,
            n_pixels_used=n_pixels_used,
            #lat_mean=float(np.nanmean(seg_lat).item()),
            #lat_min=float(np.nanmin(seg_lat).item()),
            #lat_max=float(np.nanmax(seg_lat).item()),
            #lon_mean=float(np.nanmean(seg_lon).item()),
            #along_track_distance_start_km=float(seg_dist[0].item()),
            #along_track_distance_end_km=float(seg_dist[-1].item()),
            #valid_fraction=float(np.mean(valid_fracs)) if valid_fracs else 0.0,
            lat_mean=np.nanmean(seg_lat).item(),
            lat_min=np.nanmin(seg_lat).item(),
            lat_max=np.nanmax(seg_lat).item(),
            lon_mean=_circular_mean_lon_deg(seg_lon),
            along_track_distance_start_km=seg_dist[0].item(),
            along_track_distance_end_km=seg_dist[-1].item(),
            valid_fraction=np.mean(valid_fracs) if valid_fracs else 0.0,
            gap_filled=any_gap_filled,
            month=month,
            wavenumber_2d=wavenumber_2d,
            psd_2d=psd_2d,
            n_cols_2d=n_cols_2d,
            dx2_km_2d=dx2_km_resolved if cross_track_integrated else None,
        ))

    if segments:
        wavenumber = segments[0].wavenumber
        mean_psd = np.mean(np.stack([s.psd for s in segments], axis=0), axis=0)
    else:
        wavenumber = np.array([])
        mean_psd = np.array([])

    segments_2d = [s for s in segments if s.psd_2d is not None]
    if segments_2d:
        wavenumber_2d_out = segments_2d[0].wavenumber_2d
        mean_psd_2d = np.mean(np.stack([s.psd_2d for s in segments_2d], axis=0), axis=0)
    else:
        wavenumber_2d_out = np.array([]) if cross_track_integrated else None
        mean_psd_2d = np.array([]) if cross_track_integrated else None

    return PassSpectrumResult(
        swath=swath_name,
        wavenumber=wavenumber,
        mean_psd=mean_psd,
        n_segments_used=len(segments),
        n_segments_total=len(bounds),
        segments=segments,
        wavenumber_2d=wavenumber_2d_out,
        mean_psd_2d=mean_psd_2d,
        n_segments_used_2d=len(segments_2d),
        month=month,
    )


def compute_pass_spectra(
    ssha: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    cross_track_distance: np.ndarray,
    segment_length_km: float,
    month: Optional[int] = None,
    along_track_spacing_km: Optional[float] = None,
    overlap: float = 0.0,
    max_gap_fraction: float = 0.25,
    detrend: str = "linear",
    window: str = "hann",
    min_pixels_per_segment: int = 3,
    cross_track_integrated: bool = False,
    dx2_km: Optional[float] = None,
    max_nan_fraction_2d: float = 0.15,
    remove_plane_2d: bool = True,
    window_2d: str = "hann",
):
    """
    Compute along-track wavenumber
    spectra for BOTH swaths (left and right) of a SWOT L2 LR Basic pass,
    handling land/NaN gaps and the nadir gap automatically.

    Parameters mirror compute_swath_spectra(). In particular,
    cross_track_integrated=True additionally computes, per swath, a 2-D
    (along-track x cross-track) cross-track-integrated PSD
    (PassSpectrumResult.mean_psd_2d / .wavenumber_2d and each segment's
    .psd_2d), directly comparable to the native-resolution "total" PSD
    produced by swot_alias_spectra.py. It is off by default -- the
    default return value is unchanged from the existing along-track-only
    (per-column-averaged) computation.

    Returns
    -------
    dict with keys "left" and "right", each a PassSpectrumResult.
    """
    left_mask, right_mask = split_left_right_swaths(cross_track_distance)

    results = {}
    for name, mask in (("left", left_mask), ("right", right_mask)):
        results[name] = compute_swath_spectra(
            ssha=ssha,
            latitude=latitude,
            longitude=longitude,
            swath_mask=mask,
            month=month,
            swath_name=name,
            segment_length_km=segment_length_km,
            along_track_spacing_km=along_track_spacing_km,
            overlap=overlap,
            max_gap_fraction=max_gap_fraction,
            detrend=detrend,
            window=window,
            min_pixels_per_segment=min_pixels_per_segment,
            cross_track_integrated=cross_track_integrated,
            cross_track_distance=cross_track_distance,
            dx2_km=dx2_km,
            max_nan_fraction_2d=max_nan_fraction_2d,
            remove_plane_2d=remove_plane_2d,
            window_2d=window_2d,
        )
    return results
