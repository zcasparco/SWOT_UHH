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


@dataclasses.dataclass
class PassSpectrumResult:
    """Full result for one pass: per-segment spectra + mean spectrum per swath."""

    swath: str
    wavenumber: np.ndarray              # common wavenumber axis (cycles/km)
    mean_psd: np.ndarray                # pass-mean PSD, averaged over segments
    n_segments_used: int
    n_segments_total: int
    segments: list  # list[SegmentSpectrum]

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
    # distance axis itself is always well defined
    ref_lat = _interp_nan_1d(ref_lat)
    ref_lon = _interp_nan_1d(ref_lon)

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
    fractional overlap (0 <= overlap < 1)."""
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1).")
    step_km = segment_length_km * (1.0 - overlap)
    total = distance_km[-1]
    starts_km = np.arange(distance_km[0], total - segment_length_km + 1e-9, step_km)
    if len(starts_km) == 0:
        # pass shorter than one segment: use whole pass as a single segment
        starts_km = np.array([distance_km[0]])

    bounds = []
    for s_km in starts_km:
        e_km = s_km + segment_length_km
        i0 = int(np.searchsorted(distance_km, s_km, side="left"))
        i1 = int(np.searchsorted(distance_km, e_km, side="right"))
        if i1 - i0 < 8:  # need a minimum number of samples to FFT meaningfully
            continue
        bounds.append((i0, i1))
    return bounds


def compute_swath_spectra(
    ssha: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    swath_mask: np.ndarray,
    swath_name: str,
    segment_length_km: float,
    along_track_spacing_km: Optional[float] = None,
    overlap: float = 0.0,
    max_gap_fraction: float = 0.25,
    detrend: str = "linear",
    window: str = "hann",
    min_pixels_per_segment: int = 3,
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

    Returns
    -------
    PassSpectrumResult
    """
    ssha = np.asarray(ssha, dtype=float)
    cols = np.where(swath_mask)[0]
    if cols.size == 0:
        raise ValueError(f"No pixels found for swath '{swath_name}'.")

    sub_ssha = ssha[:, cols]
    sub_lat = latitude[:, cols]
    sub_lon = longitude[:, cols]

    distance_km = along_track_distance_km(sub_lat, sub_lon)

    if along_track_spacing_km is None:
        diffs = np.diff(distance_km)
        diffs = diffs[diffs > 0]
        along_track_spacing_km = float(np.median(diffs)) if diffs.size else 2.0

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
        if uniform_dist[-1] > seg_dist[-1] + along_track_spacing_km:
            continue

        col_psds = []
        any_gap_filled = False
        valid_fracs = []

        for c in range(seg_ssha.shape[1]):
            col = seg_ssha[:, c]
            valid = ~np.isnan(col)
            valid_frac = valid.mean() if len(col) else 0.0
            valid_fracs.append(valid_frac)

            if valid_frac < (1.0 - max_gap_fraction) or valid.sum() < 8:
                continue  # too many gaps / land in this column -> skip it

            # interpolate this column (in native sampling) onto the
            # uniform grid; NaNs interpolated only between valid points,
            # never extrapolated beyond the valid data range.
            good_dist = seg_dist[valid]
            good_val = col[valid]
            if good_dist[0] > uniform_dist[0] or good_dist[-1] < uniform_dist[-1]:
                # uniform grid would require extrapolation -> skip column
                continue

            resampled = np.interp(uniform_dist, good_dist, good_val)
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

        segments.append(SegmentSpectrum(
            swath=swath_name,
            segment_index=seg_idx,
            wavenumber=freqs,
            psd=mean_pxx,
            n_pixels_used=n_pixels_used,
            lat_mean=float(np.nanmean(seg_lat)),
            lat_min=float(np.nanmin(seg_lat)),
            lat_max=float(np.nanmax(seg_lat)),
            lon_mean=float(np.nanmean(seg_lon)),
            along_track_distance_start_km=float(seg_dist[0]),
            along_track_distance_end_km=float(seg_dist[-1]),
            valid_fraction=float(np.mean(valid_fracs)) if valid_fracs else 0.0,
            gap_filled=any_gap_filled,
        ))

    if segments:
        wavenumber = segments[0].wavenumber
        mean_psd = np.mean(np.stack([s.psd for s in segments], axis=0), axis=0)
    else:
        wavenumber = np.array([])
        mean_psd = np.array([])

    return PassSpectrumResult(
        swath=swath_name,
        wavenumber=wavenumber,
        mean_psd=mean_psd,
        n_segments_used=len(segments),
        n_segments_total=len(bounds),
        segments=segments,
    )


def compute_pass_spectra(
    ssha: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    cross_track_distance: np.ndarray,
    segment_length_km: float,
    along_track_spacing_km: Optional[float] = None,
    overlap: float = 0.0,
    max_gap_fraction: float = 0.25,
    detrend: str = "linear",
    window: str = "hann",
    min_pixels_per_segment: int = 3,
):
    """
    Top-level convenience function: compute along-track wavenumber
    spectra for BOTH swaths (left and right) of a SWOT L2 LR Basic pass,
    handling land/NaN gaps and the nadir gap automatically.

    Parameters mirror compute_swath_spectra(); see that function's
    docstring for full details.

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
            swath_name=name,
            segment_length_km=segment_length_km,
            along_track_spacing_km=along_track_spacing_km,
            overlap=overlap,
            max_gap_fraction=max_gap_fraction,
            detrend=detrend,
            window=window,
            min_pixels_per_segment=min_pixels_per_segment,
        )
    return results


