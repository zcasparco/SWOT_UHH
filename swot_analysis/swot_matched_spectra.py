"""
swot_matched_spectra.py
=======================

Segment-matched along-track wavenumber spectra from a SWOT L2 LR *Unsmoothed*
file and the corresponding *Expert* file of the same pass.

For every Unsmoothed segment that survives the usual NaN / gap screening, this
module

1. runs the alias decomposition (`alias_decompose_patch`) -> predicted Expert
   spectrum split into ``direct`` and ``aliased`` parts,
2. finds the Expert-grid rows covering the same footprint (matched on the
   segment centre, fixed number of Expert rows),
3. computes the *measured* Expert spectrum on those rows with exactly the same
   estimator conventions as the prediction (2-D Hann periodogram, integrated
   over cross-track wavenumber, one-sided),

and returns only the segments for which BOTH products are usable. Hence, for
every swath, predicted and measured spectra have

* the same number of segments, in the same order,
* the same wavenumber axis (cycles/km),
* the same central latitude / longitude (those of the Unsmoothed footprint;
  the actual Expert centre and the residual offset in km are also stored).

Main entry points
-----------------
process_matched_pair(unsmoothed_path, expert_path, output_path=None, ...)
    Load both files and return ``{"left": [pairs], "right": [pairs]}``.
compute_matched_spectra(data_unsmoothed, data_expert, ...)
    Same, but from already-loaded dicts (as returned by the loaders).
matched_pairs_to_dataset(pairs)
    Pack a list of pairs into an ``xarray.Dataset`` (segment x wavenumber).
"""

from __future__ import annotations

import contextlib
import dataclasses
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal

try:  # package layout used by the existing wrapper
    from swot_analysis import swot_alias_spectra as _alias_mod
    from swot_analysis.swot_alias_spectra import (
        compute_alias_pass_spectra,
        load_swot_l2_unsmoothed,
        periodogram_2d,
        plane_detrend_2d,
        split_left_right_swaths,
        _circular_mean_lon_deg,
        _fill_nan_2d,
        _haversine_km,
        _interp_nan_1d,
        _interp_nan_lon_1d,
        _max_contiguous_gap_fraction,
        _max_contiguous_gap_rows,
        _resolve_segment_sampling,
        _segment_bounds_fixed_samples,
        along_track_distance_km,
    )
except ImportError:  # flat layout (module sitting next to this file)
    import swot_alias_spectra as _alias_mod  # type: ignore
    from swot_alias_spectra import (  # type: ignore
        compute_alias_pass_spectra,
        load_swot_l2_unsmoothed,
        periodogram_2d,
        plane_detrend_2d,
        split_left_right_swaths,
        _circular_mean_lon_deg,
        _fill_nan_2d,
        _haversine_km,
        _interp_nan_1d,
        _interp_nan_lon_1d,
        _max_contiguous_gap_fraction,
        _max_contiguous_gap_rows,
        _resolve_segment_sampling,
        _segment_bounds_fixed_samples,
        along_track_distance_km,
    )

try:
    import xarray as xr
except ImportError:
    xr = None


# =============================================================================
# Default configuration (same defaults as swot_alias_wrapper.py)
# =============================================================================

SEGMENT_LENGTH_KM = 1000.0
OVERLAP = 0.5

MAX_NAN_FRACTION = 0.15
MAX_GAP_FRACTION = 0.1

N_TAPS = 17
DETREND = "plane"   # "plane" | "linear" | "constant" | None
REMOVE_EDGES_KM = None

DX1_NATIVE_KM = 0.25
DX2_NATIVE_KM = 0.25
DX1_EXPERT_KM = 2.0
DX2_EXPERT_KM = 2.0

SWATHS = ("left", "right")

# --- QC defaults (new) -------------------------------------------------------
# Row-level plausibility screening applied to BOTH products before segmenting.
QC_MASK_BAD_ROWS = True
QC_MAX_ABS_DEV_M = None        # None -> auto: max(5 m, 8 * robust sigma of row medians)
QC_MAX_STEP_M = 5.0            # max jump of the cross-track row median between 2 lines
QC_MAX_PIXEL_DEV_M = 10.0      # max |pixel - row median| (spikes)
QC_DILATE_KM = 10.0            # grow the masked rows by this distance each side

# Per-pair flags.
QC_MAX_HF_RATIO = 20.0         # predicted / expert PSD in the high-k band
QC_HF_BAND_MIN_K = 0.1         # cycles/km
QC_MAX_LENGTH_ERROR = 0.03     # |track length - expected| / expected
QC_MAX_STEP_FACTOR = 3.0       # max along-track step / nominal dx


# =============================================================================
# Result container
# =============================================================================

@dataclasses.dataclass
class MatchedSegmentPair:
    """
    One Unsmoothed segment and its matching Expert segment.

    All PSDs share ``wavenumber`` (Expert along-track axis, cycles/km) and have
    units of SSHA^2 / (cycles/km).

    ``predicted_psd`` = ``direct_psd`` + ``aliased_psd`` is the Expert spectrum
    predicted from the Unsmoothed data; ``expert_psd`` is the one measured on
    the Expert grid over the same footprint.
    """

    swath: str
    segment_index: int          # index of the Unsmoothed segment in its swath

    wavenumber: np.ndarray

    # Predicted from Unsmoothed
    direct_psd: np.ndarray
    aliased_psd: np.ndarray
    predicted_psd: np.ndarray

    # Measured on Expert grid
    expert_psd: np.ndarray

    # Common reference position (Unsmoothed footprint)
    lat_mean: float
    lon_mean: float
    lat_min: float
    lat_max: float

    # Actual Expert footprint centre and residual mismatch
    expert_lat_mean: float
    expert_lon_mean: float
    center_offset_km: float

    # Row bookkeeping
    unsmoothed_row_start: int
    unsmoothed_row_end: int     # exclusive
    expert_row_start: int
    expert_row_end: int         # exclusive

    along_track_distance_start_km: float
    along_track_distance_end_km: float

    # Data-quality diagnostics for both products
    unsmoothed_valid_fraction: float
    expert_valid_fraction: float
    unsmoothed_gap_filled: bool
    expert_gap_filled: bool
    unsmoothed_gap_fraction: Optional[float]
    expert_gap_fraction: float
    unsmoothed_n_pixels: int
    expert_n_pixels: int

    # QC flags (new). ``qc_ok`` is False if ANY check fails; ``qc_reasons`` is a
    # comma-separated list of the failed checks (empty string when ok):
    #   hf_ratio_high  predicted high-k PSD >> measured Expert high-k PSD
    #                  (the "flat predicted / aliased spectrum" symptom)
    #   hf_ratio_nan   high-k ratio could not be computed
    #   grid_length    along-track length of a segment != expected from dx
    #   grid_step      a gap / skipped lines in the along-track grid
    qc_ok: bool = True
    qc_reasons: str = ""
    hf_ratio: float = float("nan")
    unsmoothed_track_length_km: float = float("nan")
    expert_track_length_km: float = float("nan")
    unsmoothed_max_step_km: float = float("nan")
    expert_max_step_km: float = float("nan")
    unsmoothed_length_error: float = float("nan")
    expert_length_error: float = float("nan")
    unsmoothed_dx_actual_km: float = float("nan")
    expert_dx_actual_km: float = float("nan")


# =============================================================================
# Expert loader (single group, no left/right split)
# =============================================================================

def load_swot_l2_expert(filepath, ssh_var="ssha_karin_2", HRET=False):
    """
    Load an Expert (Basic-layout) SWOT L2 LR file.

    Mirrors ``load_swot_l2_unsmoothed`` so both products are treated
    identically: ``height_cor_xover`` and ``internal_tide_hret`` are added with
    a missing value meaning "no correction" (0.0) rather than propagating NaN,
    and the same quality / surface flags are applied.

    NOTE: pass the same ``HRET`` to both loaders, otherwise predicted and
    measured spectra differ by the internal-tide signal.
    """
    try:
        import xarray as _xr
    except ImportError as exc:
        raise ImportError("xarray is required to load SWOT files.") from exc

    ds = _xr.open_dataset(filepath)

    base = ds[ssh_var].values.astype(float)

    xover = ds["height_cor_xover"].values.astype(float)
    xover = np.where(np.isfinite(xover), xover, 0.0)
    ssha = base + xover

    if HRET:
        hret = ds["internal_tide_hret"].values.astype(float)
        hret = np.where(np.isfinite(hret), hret, 0.0)
        ssha = ssha + hret

    ssha = np.where(np.isfinite(base), ssha, np.nan)

    for q in (f"{ssh_var}_qual", "ssha_karin_2_qual", "ssh_karin_2_qual"):
        if q in ds.variables:
            ssha = np.where(np.asarray(ds[q]) == 0, ssha, np.nan)
            break

    for q in (
        "ancillary_surface_classification_flag",
        "surface_classification_flag",
    ):
        if q in ds.variables:
            ssha = np.where(np.asarray(ds[q]) == 0, ssha, np.nan)
            break

    if "surface_type" in ds.variables:
        ssha = np.where(np.asarray(ds["surface_type"]) == 0, ssha, np.nan)

    out = {
        "ssha": ssha,
        "latitude": ds["latitude"].values.astype(float),
        "longitude": ds["longitude"].values.astype(float),
        "cross_track_distance": ds["cross_track_distance"].values.astype(float),
    }
    ds.close()
    return out


# =============================================================================
# Helpers
# =============================================================================

DETREND_CHOICES = ("plane", "linear", "constant", "none")


def make_detrend_func(detrend):
    """
    Return a function ``f(patch2d) -> patch2d`` for the requested detrend.

    ``"plane"``    best-fit 2-D plane (a + b*cross + c*along); the default and
                   what the Unsmoothed pipeline did before (``remove_plane``).
    ``"linear"``   1-D linear trend along track, removed independently for
                   every cross-track column (``scipy.signal.detrend``).
    ``"constant"`` segment mean only.
    ``None`` / ``"none"`` / ``False``  no detrending.
    """
    if detrend is None or detrend is False:
        detrend = "none"
    if detrend is True:
        detrend = "plane"
    name = str(detrend).lower()
    if name not in DETREND_CHOICES:
        raise ValueError(f"detrend must be one of {DETREND_CHOICES} or None, got {detrend!r}")

    if name == "plane":
        return plane_detrend_2d
    if name == "linear":
        return lambda a: signal.detrend(a, axis=0, type="linear")
    if name == "constant":
        return lambda a: a - a.mean()
    return lambda a: a


@contextlib.contextmanager
def _unsmoothed_detrend(func):
    """
    ``compute_alias_swath_spectra`` only offers ``remove_plane`` and calls the
    module-level ``plane_detrend_2d`` when it is True. To apply the chosen
    detrend to the Unsmoothed segments without editing that module, swap the
    function it calls for the duration of the run and restore it afterwards.
    (Not thread-safe; fine for processes.)
    """
    original = _alias_mod.plane_detrend_2d
    _alias_mod.plane_detrend_2d = func
    try:
        yield
    finally:
        _alias_mod.plane_detrend_2d = original


def _edge_in_xt_units(cross_track_distance, remove_edges_km):
    """
    ``split_left_right_swaths`` compares ``remove_edges_km`` directly against
    the values of ``cross_track_distance``, which SWOT files store in METRES.
    Convert so that "km" really means km. Returns None if no edge removal.
    """
    if remove_edges_km is None:
        return None
    xt = np.asarray(cross_track_distance, dtype=float)
    if np.nanmax(np.abs(xt)) > 1000.0:  # values are metres
        return float(remove_edges_km) * 1000.0
    return float(remove_edges_km)


def _xt_1d(cross_track_distance):
    """Per-column cross-track distance (median over rows if 2-D)."""
    xt = np.asarray(cross_track_distance, dtype=float)
    return np.nanmedian(xt, axis=0) if xt.ndim == 2 else xt


def _reference_track(lat, lon, ref_col):
    """lat/lon of one reference column (NaN-interpolated, wrap-safe)."""
    return _interp_nan_1d(lat[:, ref_col]), _interp_nan_lon_1d(lon[:, ref_col])


def _good_columns(data, max_column_nan_fraction):
    """
    Columns that are not chronically invalid over the pass: NaN fraction,
    evaluated only on rows where the swath is mostly valid (so that land /
    whole-row dropouts do not make every column look bad), must not exceed
    ``max_column_nan_fraction``. Catches the all-NaN nadir-gap and flagged
    edge pixels that Expert files carry inside their left/right masks.
    """
    fin = np.isfinite(data["ssha"])
    row_valid = fin.mean(axis=1)
    if row_valid.max() <= 0:
        return np.zeros(fin.shape[1], dtype=bool)
    rows_ok = row_valid >= 0.5 * row_valid.max()
    col_nan = 1.0 - fin[rows_ok].mean(axis=0)
    return col_nan <= max_column_nan_fraction


def harmonize_columns(
    data_unsmoothed, data_expert, dx2_expert_km, max_column_nan_fraction=0.5
):
    """
    Drop chronically-invalid columns from both products and clip both to the
    cross-track range they have in common (per swath), so that every segment
    is built from the same physical swath strip in both products and no
    all-NaN column can make a whole segment unfillable.

    Returns new data dicts (inputs are not modified) and a per-swath summary.
    """
    out, info = [], {}
    xt_u = _xt_1d(data_unsmoothed["cross_track_distance"])
    xt_e = _xt_1d(data_expert["cross_track_distance"])
    scale = 1000.0 if max(np.nanmax(np.abs(xt_u)), np.nanmax(np.abs(xt_e))) > 1000.0 else 1.0
    tol = 0.5 * dx2_expert_km * scale

    good_u = _good_columns(data_unsmoothed, max_column_nan_fraction)
    good_e = _good_columns(data_expert, max_column_nan_fraction)
    keep_u = np.zeros_like(good_u)
    keep_e = np.zeros_like(good_e)

    for swath, sign in (("left", -1.0), ("right", 1.0)):
        su = good_u & (sign * xt_u > 0)
        se = good_e & (sign * xt_e > 0)
        info[swath] = dict(n_cols_unsmoothed=int(su.sum()), n_cols_expert=int(se.sum()))
        if not su.any() or not se.any():
            continue
        au, ae = np.abs(xt_u[su]), np.abs(xt_e[se])
        lo, hi = max(au.min(), ae.min()), min(au.max(), ae.max())
        keep_u |= su & (np.abs(xt_u) >= lo - tol) & (np.abs(xt_u) <= hi + tol)
        keep_e |= se & (np.abs(xt_e) >= lo - tol) & (np.abs(xt_e) <= hi + tol)
        info[swath].update(
            cross_track_range_km=(float(lo / scale), float(hi / scale)),
            n_cols_unsmoothed_kept=int((keep_u & (sign * xt_u > 0)).sum()),
            n_cols_expert_kept=int((keep_e & (sign * xt_e > 0)).sum()),
        )

    for data, keep in ((data_unsmoothed, keep_u), (data_expert, keep_e)):
        d = dict(data)
        for key in ("ssha", "latitude", "longitude"):
            d[key] = data[key][:, keep]
        xt = np.asarray(data["cross_track_distance"])
        d["cross_track_distance"] = xt[:, keep] if xt.ndim == 2 else xt[keep]
        out.append(d)
    return out[0], out[1], info


def _segment_center(lat_ref, lon_ref, i0, i1):
    """
    Geographic centre of rows [i0, i1) on the reference track: the middle row
    if the count is odd, the mean of the two middle rows if even.
    """
    lo = (i0 + i1 - 1) // 2
    hi = (i0 + i1) // 2
    lat = 0.5 * (lat_ref[lo] + lat_ref[hi])
    lon = _circular_mean_lon_deg(np.array([lon_ref[lo], lon_ref[hi]]))
    return float(lat), float(lon)


def _nanmedian_axis1(a):
    """Row-wise nanmedian without the all-NaN RuntimeWarning (-> NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=1)


def mask_implausible_rows(
    data,
    dx_km,
    *,
    max_abs_dev_m=QC_MAX_ABS_DEV_M,
    max_step_m=QC_MAX_STEP_M,
    max_pixel_dev_m=QC_MAX_PIXEL_DEV_M,
    dilate_km=QC_DILATE_KM,
):
    """
    Set to NaN rows (and spikes) whose SSH(A) values are not physical.

    Why: ``qual == 0`` / surface flags do NOT catch everything. Near coasts,
    ice, and in some dropouts the Unsmoothed ``ssh_karin_2`` contains smooth
    ramps and plateaus of hundreds-thousands of metres with quality 0. Such
    rows contain no NaN, so they pass every NaN / gap screening, and the 250 m
    -> 2 km alias decomposition then turns the step/ramp (and the tail of the
    low-pass filter applied to it) into a big, flat (white-like) aliased and
    predicted PSD, while the real 2 km Expert data (where those rows were
    flagged) show a normal red spectrum.

    Tests, all on the cross-track row median (robust against noisy pixels):
      * |row_median - pass median| > ``max_abs_dev_m`` (auto if None)
      * |diff(row_median)| > ``max_step_m`` (ramps, jumps)
      * per pixel |value - row_median| > ``max_pixel_dev_m`` (spikes)
    Flagged rows are dilated by ``dilate_km`` on each side.

    Returns ``(new_data, info)``; inputs are not modified.
    """
    ssha = np.array(data["ssha"], dtype=float, copy=True)
    n_rows = ssha.shape[0]
    rmed = _nanmedian_axis1(ssha)
    fin = np.isfinite(rmed)
    info = dict(n_rows=int(n_rows), n_rows_masked=0, n_pixels_spike=0,
                max_abs_dev_m=None)
    if fin.sum() < 3:
        return dict(data, ssha=ssha), info

    ref = float(np.median(rmed[fin]))
    if max_abs_dev_m is None:
        mad = float(np.median(np.abs(rmed[fin] - ref)))
        max_abs_dev_m = max(5.0, 8.0 * 1.4826 * mad)
    info["max_abs_dev_m"] = float(max_abs_dev_m)

    bad = np.zeros(n_rows, dtype=bool)
    bad[fin] |= np.abs(rmed[fin] - ref) > max_abs_dev_m

    # step test on the valid rows only (so NaN rows do not hide a jump)
    idx = np.where(fin)[0]
    dstep = np.abs(np.diff(rmed[idx]))
    jump = dstep > max_step_m
    bad[idx[:-1][jump]] = True
    bad[idx[1:][jump]] = True

    nd = int(np.ceil(dilate_km / dx_km))
    if nd > 0 and bad.any():
        bad = np.convolve(bad.astype(float), np.ones(2 * nd + 1), mode="same") > 0

    ssha[bad, :] = np.nan

    with np.errstate(invalid="ignore"):
        spike = np.abs(ssha - rmed[:, None]) > max_pixel_dev_m
    info["n_pixels_spike"] = int(spike.sum())
    ssha[spike] = np.nan

    info["n_rows_masked"] = int(bad.sum())
    return dict(data, ssha=ssha), info


def _actual_dx_km(lat_ref, lon_ref):
    """Median along-track step of a whole reference track (km)."""
    st = np.atleast_1d(_haversine_km(lat_ref[:-1], lon_ref[:-1], lat_ref[1:], lon_ref[1:]))
    st = st[np.isfinite(st) & (st > 0)]
    return float(np.median(st)) if st.size else np.nan


def _track_stats(lat_ref, lon_ref, i0, i1, dx_actual_km):
    """
    Along-track length of rows [i0, i1) on the reference track, its largest
    single step, and the relative length error against ``(n-1) * dx_actual_km``.

    ``dx_actual_km`` is the MEASURED median line spacing of the whole pass,
    not the nominal one: the real spacing differs slightly from 0.25 / 2 km
    (cross-track offset of the reference column, orbit geometry), and a check
    against the nominal value flags every segment. Against the measured
    spacing, only genuinely irregular stretches (missing / duplicated lines)
    deviate. Returns ``(length_km, max_step_km, length_error_frac)``; NaN
    entries mean "could not be evaluated" and are not treated as a failure.
    """
    la, lo = lat_ref[i0:i1], lon_ref[i0:i1]
    if la.size < 2 or not np.isfinite(dx_actual_km) or not (
        np.all(np.isfinite(la)) and np.all(np.isfinite(lo))
    ):
        return np.nan, np.nan, np.nan
    steps = np.atleast_1d(_haversine_km(la[:-1], lo[:-1], la[1:], lo[1:]))
    length = float(np.sum(steps))
    expected = (la.size - 1) * dx_actual_km
    return length, float(np.max(steps)), abs(length - expected) / expected


def _hf_ratio(k, predicted, expert, k_min):
    """Mean predicted / mean expert PSD for k >= k_min (NaN if undefined)."""
    m = (k >= k_min) & np.isfinite(predicted) & np.isfinite(expert)
    if m.sum() < 3:
        return np.nan
    e = float(np.mean(expert[m]))
    return float(np.mean(predicted[m]) / e) if e > 0 else np.nan


def expert_segment_psd(patch, dx1_expert_km, dx2_expert_km):
    """
    Measured Expert along-track PSD of one (already filled / detrended)
    2-D patch.

    Uses the same estimator as ``alias_decompose_patch``'s *total* spectrum:
    2-D Hann-tapered periodogram, integrated over cross-track wavenumber,
    folded to a one-sided spectrum (interior bins doubled). Output axis is
    identical to the decomposition's: ``fftfreq(n_rows, dx1_expert_km) >= 0``.
    """
    n1, n2 = patch.shape
    if n1 < 4 or n2 < 4:
        return None

    k1, k2, s2d = periodogram_2d(patch, dx1_expert_km, dx2_expert_km)

    dk2 = np.abs(k2[1] - k2[0])
    psd_1d = s2d.sum(axis=1) * dk2

    positive = k1 >= 0
    k = k1[positive]
    psd = psd_1d[positive].copy()

    interior = (k > 0) & (k < np.max(np.abs(k1)))
    psd[interior] *= 2.0

    return k, psd


def _screen_and_fill(
    patch,
    dx_km,
    max_nan_fraction,
    max_gap_fraction,
    max_gap_km,
    detrend_func,
):
    """
    Same NaN gate as ``compute_alias_swath_spectra``. Returns
    ``(filled, info)`` on success, or ``(None, reason)`` with reason in
    {"nan_fraction", "gap", "fill_failed"}.
    """
    valid_fraction = float(np.isfinite(patch).mean())
    if valid_fraction < 1.0 - max_nan_fraction:
        return None, "nan_fraction"

    gap_fraction = _max_contiguous_gap_fraction(patch, axis=0)
    if gap_fraction > max_gap_fraction:
        return None, "gap"

    if max_gap_km is not None:
        if _max_contiguous_gap_rows(patch, axis=0) * dx_km > max_gap_km:
            return None, "gap"

    filled = _fill_nan_2d(patch, max_nan_fraction=max_nan_fraction)
    if filled is None:
        return None, "fill_failed"

    gap_filled = bool(np.isnan(patch).any())
    filled = detrend_func(filled)

    return filled, dict(
        valid_fraction=valid_fraction,
        gap_fraction=gap_fraction,
        gap_filled=gap_filled,
    )


def _best_expert_window(
    lat_ref_e, lon_ref_e, n_rows_e, n1e, target_lat, target_lon
):
    """
    Choose the Expert window of exactly ``n1e`` rows whose centre is closest
    to (target_lat, target_lon). Returns ``(j0, j1, offset_km, lat_c, lon_c)``
    or None when the window would leave the granule.
    """
    d = _haversine_km(target_lat, target_lon, lat_ref_e, lon_ref_e)
    if not np.any(np.isfinite(d)):
        return None
    jc = int(np.nanargmin(d))

    best = None
    for shift in (-1, 0, 1):
        j0 = jc - n1e // 2 + shift
        j1 = j0 + n1e
        if j0 < 0 or j1 > n_rows_e:
            continue
        lat_c, lon_c = _segment_center(lat_ref_e, lon_ref_e, j0, j1)
        off = float(_haversine_km(target_lat, target_lon, lat_c, lon_c))
        if best is None or off < best[2]:
            best = (j0, j1, off, lat_c, lon_c)
    return best


# =============================================================================
# Core: matched spectra from already-loaded data
# =============================================================================

def compute_matched_spectra(
    data_unsmoothed: dict,
    data_expert: dict,
    *,
    segment_length_km: float = SEGMENT_LENGTH_KM,
    overlap: float = OVERLAP,
    max_nan_fraction: float = MAX_NAN_FRACTION,
    max_gap_fraction: float = MAX_GAP_FRACTION,
    max_gap_km: Optional[float] = None,
    n_taps: int = N_TAPS,
    dx1_native_km: float = DX1_NATIVE_KM,
    dx2_native_km: float = DX2_NATIVE_KM,
    dx1_expert_km: float = DX1_EXPERT_KM,
    dx2_expert_km: float = DX2_EXPERT_KM,
    detrend=DETREND,
    remove_edges_km: Optional[float] = REMOVE_EDGES_KM,
    max_center_offset_km: Optional[float] = None,
    harmonize: bool = True,
    max_column_nan_fraction: float = 0.5,
    qc_mask_bad_rows: bool = QC_MASK_BAD_ROWS,
    qc_max_abs_dev_m: Optional[float] = QC_MAX_ABS_DEV_M,
    qc_max_step_m: float = QC_MAX_STEP_M,
    qc_max_pixel_dev_m: float = QC_MAX_PIXEL_DEV_M,
    qc_dilate_km: float = QC_DILATE_KM,
    qc_max_hf_ratio: Optional[float] = QC_MAX_HF_RATIO,
    qc_hf_band_min_k: float = QC_HF_BAND_MIN_K,
    qc_max_length_error: Optional[float] = QC_MAX_LENGTH_ERROR,
    qc_max_step_factor: Optional[float] = QC_MAX_STEP_FACTOR,
    drop_flagged: bool = False,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    verbose: bool = False,
):
    """
    Matched Unsmoothed-predicted / Expert-measured spectra for both swaths.

    Parameters (beyond the usual segmentation/NaN ones)
    ---------------------------------------------------
    detrend : {"plane", "linear", "constant", None}
        Trend removed from every segment, Unsmoothed AND Expert, after gap
        filling and before the spectrum is computed. See
        :func:`make_detrend_func`. Default ``"plane"``.
    max_center_offset_km : float, optional
        Reject a pair if the best Expert window centre is further than this
        from the Unsmoothed segment centre. Default ``2 * dx1_expert_km``.
        This is also the guard against pairing files from different
        passes/cycles (they would be hundreds of km apart).
    harmonize : bool
        Before anything else, drop chronically-invalid cross-track columns
        (e.g. the all-NaN nadir gap and flagged edge pixels that Expert files
        keep inside their left/right masks) from both products and clip both
        to their common cross-track range. Without this, one all-NaN column
        makes every Expert segment unfillable (-> empty output). See
        :func:`harmonize_columns`. ``max_column_nan_fraction`` is the NaN
        fraction above which a column counts as invalid.
    qc_mask_bad_rows : bool
        NaN-out non-physical rows/spikes in BOTH products before segmenting
        (:func:`mask_implausible_rows`). This is the actual FIX for flat
        predicted/aliased spectra caused by huge-valued rows that carry
        ``qual == 0``. The masked rows then become gaps and the existing
        ``max_gap_fraction`` / ``max_nan_fraction`` screening drops the
        affected segments.
    qc_max_hf_ratio, qc_hf_band_min_k : float
        Flag (``qc_ok=False``) when mean predicted / mean measured PSD for
        ``k >= qc_hf_band_min_k`` exceeds ``qc_max_hf_ratio``. Catches whatever
        the row masking misses. ``None`` disables.
    qc_max_length_error, qc_max_step_factor : float
        Irregular-grid flags: relative error of a segment's along-track length
        vs ``(n_rows-1)*dx``, and largest along-track step vs ``dx``.
    drop_flagged : bool
        If True, flagged pairs are removed (counted in ``n_dropped_qc``);
        if False (default) they are kept with ``qc_ok=False`` / ``qc_reasons``.
    lat_min/lat_max/lon_min/lon_max : optional
        Geographic selection applied on the common ``lat_mean``/``lon_mean``
        (after pairing, so both products are always filtered identically).
        Longitudes follow the 0-360 convention used by SWOT.

    Returns
    -------
    result : dict
        ``{"left": [MatchedSegmentPair, ...], "right": [...]}``
    counts : dict
        Per swath, how many candidates were kept/dropped and why.
    """
    if max_center_offset_km is None:
        max_center_offset_km = 2.0 * dx1_expert_km

    detrend_func = make_detrend_func(detrend)

    qc_info = None
    if qc_mask_bad_rows:
        kw = dict(max_abs_dev_m=qc_max_abs_dev_m, max_step_m=qc_max_step_m,
                  max_pixel_dev_m=qc_max_pixel_dev_m, dilate_km=qc_dilate_km)
        data_unsmoothed, qc_u = mask_implausible_rows(data_unsmoothed, dx1_native_km, **kw)
        data_expert, qc_e = mask_implausible_rows(data_expert, dx1_expert_km, **kw)
        qc_info = dict(unsmoothed=qc_u, expert=qc_e)
        if verbose:
            print(f"row QC: Unsmoothed {qc_u['n_rows_masked']}/{qc_u['n_rows']} rows, "
                  f"Expert {qc_e['n_rows_masked']}/{qc_e['n_rows']} rows masked")

    column_info = None
    if harmonize:
        data_unsmoothed, data_expert, column_info = harmonize_columns(
            data_unsmoothed, data_expert, dx2_expert_km, max_column_nan_fraction)

    n_lines_patch, _Ma, n1e = _resolve_segment_sampling(
        segment_length_km, dx1_native_km, dx1_expert_km
    )

    edge_u = _edge_in_xt_units(data_unsmoothed["cross_track_distance"], remove_edges_km)
    edge_e = _edge_in_xt_units(data_expert["cross_track_distance"], remove_edges_km)

    # ---- 1. Unsmoothed alias decomposition (unchanged pipeline) -------------
    with _unsmoothed_detrend(detrend_func):
        by_swath = compute_alias_pass_spectra(
            data=data_unsmoothed,
            segment_length_km=segment_length_km,
            dx1_native_km=dx1_native_km,
            dx2_native_km=dx2_native_km,
            dx1_expert_km=dx1_expert_km,
            dx2_expert_km=dx2_expert_km,
            overlap=overlap,
            max_nan_fraction=max_nan_fraction,
            max_gap_fraction=max_gap_fraction,
            max_gap_km=max_gap_km,
            n_taps=n_taps,
            remove_plane=True,  # always calls the (swapped-in) detrend function
            remove_edges_km=edge_u,
        )

    mask_u = dict(zip(SWATHS, split_left_right_swaths(
        data_unsmoothed["cross_track_distance"], remove_edges_km=edge_u)))
    mask_e = dict(zip(SWATHS, split_left_right_swaths(
        data_expert["cross_track_distance"], remove_edges_km=edge_e)))

    xt_u = _xt_1d(data_unsmoothed["cross_track_distance"])
    xt_e = _xt_1d(data_expert["cross_track_distance"])

    ssha_e = data_expert["ssha"]
    n_rows_u = data_unsmoothed["ssha"].shape[0]
    n_rows_e = ssha_e.shape[0]

    # Segment bounds are deterministic given (n_rows, n_lines_patch, overlap);
    # recompute them to recover each segment's row range.
    bounds_u = _segment_bounds_fixed_samples(n_rows_u, n_lines_patch, overlap)

    result = {s: [] for s in SWATHS}
    counts = {}

    for swath in SWATHS:
        seg_list = by_swath[swath].segments

        u = by_swath[swath]
        cnt = dict(
            n_unsmoothed_total=u.n_segments_total,
            n_unsmoothed_dropped_nan_fraction=u.n_segments_dropped_nan_fraction,
            n_unsmoothed_dropped_gap=u.n_segments_dropped_gap,
            n_unsmoothed_dropped_fill_failed=u.n_segments_dropped_fill_failed,
            n_unsmoothed_dropped_decompose_failed=u.n_segments_dropped_decompose_failed,
            n_unsmoothed_used=len(seg_list),
            n_pairs=0,
            n_dropped_no_expert_match=0,
            n_dropped_expert_nan_fraction=0,
            n_dropped_expert_gap=0,
            n_dropped_expert_fill_failed=0,
            n_dropped_expert_failed=0,
            n_dropped_k_mismatch=0,
            n_dropped_geo_filter=0,
            n_flagged_qc=0,
            n_dropped_qc=0,
        )
        if qc_info is not None:
            cnt["row_qc"] = qc_info
        if column_info is not None:
            cnt["columns"] = column_info[swath]
        counts[swath] = cnt

        cols_u = np.where(mask_u[swath])[0]
        cols_e = np.where(mask_e[swath])[0]
        if not len(seg_list) or cols_e.size == 0:
            if len(seg_list) and cols_e.size == 0:
                cnt["n_dropped_no_expert_match"] = len(seg_list)
            continue

        # Reference tracks at the SAME cross-track position in both products
        # (the Unsmoothed mid-swath column, as used for its segment bounds,
        # and the Expert column nearest to it), so the centre offset does not
        # contain a cross-track component.
        ref_u = cols_u[len(cols_u) // 2]
        ref_e = cols_e[np.argmin(np.abs(xt_e[cols_e] - xt_u[ref_u]))]
        lat_ref_u, lon_ref_u = _reference_track(
            data_unsmoothed["latitude"], data_unsmoothed["longitude"], ref_u)
        lat_ref_e, lon_ref_e = _reference_track(
            data_expert["latitude"], data_expert["longitude"], ref_e)

        dx_act_u = _actual_dx_km(lat_ref_u, lon_ref_u)
        dx_act_e = _actual_dx_km(lat_ref_e, lon_ref_e)
        cnt["dx_actual_km"] = dict(unsmoothed=dx_act_u, expert=dx_act_e)
        if verbose:
            print(f"{swath:5s}: measured line spacing  Unsmoothed {dx_act_u:.4f} km "
                  f"(nominal {dx1_native_km}), Expert {dx_act_e:.4f} km (nominal {dx1_expert_km})")

        sub_lat_e = data_expert["latitude"][:, cols_e]
        sub_lon_e = data_expert["longitude"][:, cols_e]
        sub_ssha_e = ssha_e[:, cols_e]

        for seg in seg_list:
            i0, i1 = bounds_u[seg.segment_index]

            # ---- 2. Locate the matching Expert rows -----------------------
            t_lat, t_lon = _segment_center(lat_ref_u, lon_ref_u, i0, i1)
            win = _best_expert_window(
                lat_ref_e, lon_ref_e, n_rows_e, n1e, t_lat, t_lon)
            if win is None or win[2] > max_center_offset_km:
                cnt["n_dropped_no_expert_match"] += 1
                continue
            j0, j1, offset_km, e_lat_c, e_lon_c = win

            # ---- 3. Measured Expert spectrum ------------------------------
            patch_e = sub_ssha_e[j0:j1, :]
            filled, info = _screen_and_fill(
                patch_e, dx1_expert_km, max_nan_fraction,
                max_gap_fraction, max_gap_km, detrend_func)
            if filled is None:
                cnt[f"n_dropped_expert_{info}"] += 1
                continue

            out = expert_segment_psd(filled, dx1_expert_km, dx2_expert_km)
            if out is None:
                cnt["n_dropped_expert_failed"] += 1
                continue
            k_e, psd_e = out

            if len(k_e) != len(seg.wavenumber) or not np.allclose(k_e, seg.wavenumber):
                cnt["n_dropped_k_mismatch"] += 1
                continue

            # ---- 4. Common geographic selection ---------------------------
            if (
                (lat_min is not None and seg.lat_mean < lat_min)
                or (lat_max is not None and seg.lat_mean > lat_max)
                or (lon_min is not None and seg.lon_mean < lon_min)
                or (lon_max is not None and seg.lon_mean > lon_max)
            ):
                cnt["n_dropped_geo_filter"] += 1
                continue

            # ---- 5. QC flags (flat-spectrum symptom + irregular grid) -----
            reasons = []
            ratio = _hf_ratio(seg.wavenumber, seg.total_psd, psd_e, qc_hf_band_min_k)
            if qc_max_hf_ratio is not None:
                if not np.isfinite(ratio):
                    reasons.append("hf_ratio_nan")
                elif ratio > qc_max_hf_ratio:
                    reasons.append("hf_ratio_high")
            len_u, step_u, err_u = _track_stats(lat_ref_u, lon_ref_u, i0, i1, dx_act_u)
            len_e, step_e, err_e = _track_stats(lat_ref_e, lon_ref_e, j0, j1, dx_act_e)
            # NaN (not evaluable) is never a failure: use "not (x > tol)".
            if qc_max_length_error is not None and (
                err_u > qc_max_length_error or err_e > qc_max_length_error
            ):
                reasons.append("grid_length")
            if qc_max_step_factor is not None and (
                step_u > qc_max_step_factor * dx_act_u
                or step_e > qc_max_step_factor * dx_act_e
            ):
                reasons.append("grid_step")
            if reasons:
                cnt["n_flagged_qc"] += 1
                if drop_flagged:
                    cnt["n_dropped_qc"] += 1
                    continue

            result[swath].append(
                MatchedSegmentPair(
                    swath=swath,
                    segment_index=seg.segment_index,
                    wavenumber=seg.wavenumber,
                    direct_psd=seg.direct_psd,
                    aliased_psd=seg.aliased_psd,
                    predicted_psd=seg.total_psd,
                    expert_psd=psd_e,
                    lat_mean=seg.lat_mean,
                    lon_mean=seg.lon_mean,
                    lat_min=seg.lat_min,
                    lat_max=seg.lat_max,
                    expert_lat_mean=float(np.nanmean(sub_lat_e[j0:j1])),
                    expert_lon_mean=_circular_mean_lon_deg(sub_lon_e[j0:j1]),
                    center_offset_km=offset_km,
                    unsmoothed_row_start=int(i0),
                    unsmoothed_row_end=int(i1),
                    expert_row_start=int(j0),
                    expert_row_end=int(j1),
                    along_track_distance_start_km=seg.along_track_distance_start_km,
                    along_track_distance_end_km=seg.along_track_distance_end_km,
                    unsmoothed_valid_fraction=seg.valid_fraction,
                    expert_valid_fraction=info["valid_fraction"],
                    unsmoothed_gap_filled=bool(seg.gap_filled),
                    expert_gap_filled=info["gap_filled"],
                    unsmoothed_gap_fraction=seg.gap_fraction,
                    expert_gap_fraction=info["gap_fraction"],
                    unsmoothed_n_pixels=seg.n_pixels,
                    expert_n_pixels=filled.shape[1],
                    qc_ok=not reasons,
                    qc_reasons=",".join(reasons),
                    hf_ratio=ratio,
                    unsmoothed_track_length_km=len_u,
                    expert_track_length_km=len_e,
                    unsmoothed_max_step_km=step_u,
                    expert_max_step_km=step_e,
                    unsmoothed_length_error=err_u,
                    expert_length_error=err_e,
                    unsmoothed_dx_actual_km=dx_act_u,
                    expert_dx_actual_km=dx_act_e,
                )
            )
            cnt["n_pairs"] += 1

    if verbose:
        for swath in SWATHS:
            c = counts[swath]
            print(
                f"{swath:5s}: {c['n_pairs']} pairs | Unsmoothed kept "
                f"{c['n_unsmoothed_used']}/{c['n_unsmoothed_total']} "
                f"(nan={c['n_unsmoothed_dropped_nan_fraction']}, "
                f"gap={c['n_unsmoothed_dropped_gap']}, "
                f"fill={c['n_unsmoothed_dropped_fill_failed']}, "
                f"decomp={c['n_unsmoothed_dropped_decompose_failed']}) | "
                f"Expert-side drops: no_match={c['n_dropped_no_expert_match']}, "
                f"nan={c['n_dropped_expert_nan_fraction']}, "
                f"gap={c['n_dropped_expert_gap']}, "
                f"fill={c['n_dropped_expert_fill_failed']}, "
                f"failed={c['n_dropped_expert_failed']}, "
                f"k={c['n_dropped_k_mismatch']}, geo={c['n_dropped_geo_filter']} | "
                f"QC flagged={c['n_flagged_qc']} dropped={c['n_dropped_qc']}"
            )
            if "columns" in c:
                print(f"       columns: {c['columns']}")

    if not any(result[s] for s in SWATHS):
        warnings.warn(
            "No matched segments were produced. Check `counts` (or "
            "verbose=True): n_unsmoothed_used==0 means the Unsmoothed side "
            "already fails; n_dropped_no_expert_match means the files do not "
            "overlap / are different passes; n_dropped_expert_* means the "
            "Expert patches fail the NaN screening.",
            RuntimeWarning,
        )

    return result, counts


# =============================================================================
# Packing
# =============================================================================

def matched_pairs_to_dataset(pairs):
    """
    Pack a flat list of ``MatchedSegmentPair`` into an ``xarray.Dataset`` with
    dims (segment, wavenumber). Predicted and measured spectra, and the common
    lat/lon, are all aligned on the same ``segment`` index.
    """
    if xr is None:
        raise ImportError("xarray is required for matched_pairs_to_dataset().")
    if not pairs:
        raise ValueError("No pairs to pack.")

    if len({len(p.wavenumber) for p in pairs}) > 1:
        raise ValueError(
            "Pairs have different wavenumber lengths; they must come from "
            "identical segment_length_km / dx1_native_km / dx1_expert_km."
        )

    def col(name):
        return [getattr(p, name) for p in pairs]

    def stack(name):
        return np.stack([getattr(p, name) for p in pairs], axis=0)

    sk = ["segment", "wavenumber"]
    psd_units = {"units": "m^2 / (cycles/km)"}

    return xr.Dataset(
        data_vars=dict(
            predicted_psd=(sk, stack("predicted_psd"), {
                **psd_units,
                "long_name": "Expert PSD predicted from Unsmoothed (direct + aliased)"}),
            direct_psd=(sk, stack("direct_psd"), {
                **psd_units, "long_name": "Direct (unaliased) component"}),
            aliased_psd=(sk, stack("aliased_psd"), {
                **psd_units, "long_name": "Aliased component"}),
            expert_psd=(sk, stack("expert_psd"), {
                **psd_units, "long_name": "Measured Expert PSD"}),
            lat_mean=(["segment"], col("lat_mean"), {"units": "degrees_north"}),
            lon_mean=(["segment"], col("lon_mean"), {"units": "degrees_east"}),
            lat_min=(["segment"], col("lat_min"), {"units": "degrees_north"}),
            lat_max=(["segment"], col("lat_max"), {"units": "degrees_north"}),
            expert_lat_mean=(["segment"], col("expert_lat_mean"), {"units": "degrees_north"}),
            expert_lon_mean=(["segment"], col("expert_lon_mean"), {"units": "degrees_east"}),
            center_offset_km=(["segment"], col("center_offset_km"), {
                "units": "km",
                "long_name": "Distance between Unsmoothed and Expert segment centres"}),
            along_track_distance_start_km=(["segment"], col("along_track_distance_start_km"), {"units": "km"}),
            along_track_distance_end_km=(["segment"], col("along_track_distance_end_km"), {"units": "km"}),
            unsmoothed_row_start=(["segment"], col("unsmoothed_row_start")),
            unsmoothed_row_end=(["segment"], col("unsmoothed_row_end")),
            expert_row_start=(["segment"], col("expert_row_start")),
            expert_row_end=(["segment"], col("expert_row_end")),
            unsmoothed_valid_fraction=(["segment"], col("unsmoothed_valid_fraction")),
            expert_valid_fraction=(["segment"], col("expert_valid_fraction")),
            unsmoothed_gap_filled=(["segment"], [bool(v) for v in col("unsmoothed_gap_filled")]),
            expert_gap_filled=(["segment"], [bool(v) for v in col("expert_gap_filled")]),
            unsmoothed_n_pixels=(["segment"], col("unsmoothed_n_pixels")),
            expert_n_pixels=(["segment"], col("expert_n_pixels")),
            qc_ok=(["segment"], [bool(v) for v in col("qc_ok")]),
            qc_reasons=(["segment"], col("qc_reasons")),
            hf_ratio=(["segment"], col("hf_ratio"), {
                "long_name": "mean predicted / mean expert PSD for k >= QC_HF_BAND_MIN_K"}),
            unsmoothed_track_length_km=(["segment"], col("unsmoothed_track_length_km"), {"units": "km"}),
            expert_track_length_km=(["segment"], col("expert_track_length_km"), {"units": "km"}),
            unsmoothed_max_step_km=(["segment"], col("unsmoothed_max_step_km"), {"units": "km"}),
            expert_max_step_km=(["segment"], col("expert_max_step_km"), {"units": "km"}),
            unsmoothed_length_error=(["segment"], col("unsmoothed_length_error")),
            expert_length_error=(["segment"], col("expert_length_error")),
            swath=(["segment"], col("swath")),
            segment_index=(["segment"], col("segment_index")),
        ),
        coords=dict(
            wavenumber=("wavenumber", pairs[0].wavenumber, {"units": "cycles/km"}),
            segment=("segment", np.arange(len(pairs))),
        ),
    )


def flatten_matched_pairs(result: dict):
    """``{"left": [...], "right": [...]}`` -> flat list (left first)."""
    return [p for swath in SWATHS for p in result[swath]]


# =============================================================================
# File-level entry point
# =============================================================================

def process_matched_pair(
    unsmoothed_path,
    expert_path,
    output_path=None,
    *,
    ssh_var: str = "ssha_karin_2",
    hret: bool = True,
    return_counts: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """
    Load one Unsmoothed + one Expert file of the same pass and return
    segment-matched predicted/measured spectra.

    ``kwargs`` are forwarded to :func:`compute_matched_spectra`
    (segment_length_km, overlap, max_nan_fraction, max_gap_fraction,
    max_gap_km, n_taps, dx*_km, detrend, remove_edges_km,
    max_center_offset_km, lat_min/lat_max/lon_min/lon_max).

    ``hret`` is applied to BOTH products (so that they contain the same
    signal).

    Returns ``{"left": [MatchedSegmentPair], "right": [...]}``, or
    ``(result, counts)`` if ``return_counts``. If ``output_path`` is given the
    result, a packed ``xarray.Dataset``, the counts and the metadata are
    pickled there.
    """
    unsmoothed_path = Path(unsmoothed_path)
    expert_path = Path(expert_path)

    data_u = load_swot_l2_unsmoothed(unsmoothed_path, ssh_var=ssh_var, HRET=hret)
    data_e = load_swot_l2_expert(expert_path, ssh_var=ssh_var, HRET=hret)

    result, counts = compute_matched_spectra(
        data_u, data_e, verbose=verbose, **kwargs
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        flat = flatten_matched_pairs(result)
        dataset = None
        if flat:
            try:
                dataset = matched_pairs_to_dataset(flat)
            except Exception:
                dataset = None

        payload = {
            "pairs": flat,
            "by_swath": result,
            "dataset": dataset,
            "segment_counts": counts,
            "metadata": {
                "unsmoothed_file": str(unsmoothed_path),
                "expert_file": str(expert_path),
                "ssh_var": ssh_var,
                "hret": hret,
                **kwargs,
            },
        }
        with open(output_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return (result, counts) if return_counts else result
