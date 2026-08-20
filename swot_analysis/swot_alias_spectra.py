from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import signal

try:
    import xarray as xr
except ImportError:
    xr = None


EARTH_RADIUS_KM = 6371.0088


# ============================================================================
# Result containers
# ============================================================================

@dataclasses.dataclass
class AliasSegmentSpectrum:
    """
    Alias-decomposition result for one along-track/cross-track Unsmoothed
    SWOT segment.

    All PSDs use the same along-track wavenumber axis and have units of
    SSH^2 / (cycles/km).

    total_psd = direct_psd + aliased_psd
    """

    swath: str
    segment_index: int

    wavenumber: np.ndarray

    direct_psd: np.ndarray
    aliased_psd: np.ndarray
    total_psd: np.ndarray

    lat_mean: float
    lat_min: float
    lat_max: float
    lon_mean: float

    along_track_distance_start_km: float
    along_track_distance_end_km: float

    n_lines: int
    n_pixels: int

    valid_fraction: float
    gap_filled: bool

    # Useful diagnostics
    direct_energy: Optional[float] = None
    aliased_energy: Optional[float] = None
    aliased_fraction: Optional[float] = None
    gap_fraction: Optional[float] = None


@dataclasses.dataclass
class AliasPassSpectrumResult:
    """All alias-decomposition segments for one SWOT swath."""

    swath: str

    wavenumber: np.ndarray

    mean_direct_psd: np.ndarray
    mean_aliased_psd: np.ndarray
    mean_total_psd: np.ndarray

    segments: list[AliasSegmentSpectrum]

    n_segments_used: int
    n_segments_total: int

    month: Optional[int] = None

    # Diagnostics: how many candidate segments were dropped, and why.
    # Populated by compute_alias_swath_spectra; useful for sanity-checking
    # NaN/gap handling across a large batch of files without having to
    # inspect every segment.
    n_segments_dropped_nan_fraction: int = 0
    n_segments_dropped_gap: int = 0
    n_segments_dropped_fill_failed: int = 0
    n_segments_dropped_decompose_failed: int = 0

    def segment_latitudes(self) -> np.ndarray:
        return np.array([s.lat_mean for s in self.segments])

    def segment_longitudes(self) -> np.ndarray:
        return np.array([s.lon_mean for s in self.segments])


# ============================================================================
# Basic utilities
# ============================================================================

def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()

    good = np.isfinite(x)
    if good.sum() == 0:
        return x

    if good.sum() < len(x):
        idx = np.arange(len(x))
        x[~good] = np.interp(idx[~good], idx[good], x[good])

    return x


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        np.radians, (lat1, lon1, lat2, lon2)
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(np.minimum(1.0, a)))
    )


# ============================================================================
# SWOT geometry
# ============================================================================

def along_track_distance_km(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """
    Build a single along-track distance axis from the middle valid
    cross-track column.
    """

    if latitude.ndim == 1:
        lat = latitude
        lon = longitude
    else:
        ref_col = latitude.shape[1] // 2

        lat = latitude[:, ref_col]
        lon = longitude[:, ref_col]

        if np.all(~np.isfinite(lat)):
            valid_cols = [
                c for c in range(latitude.shape[1])
                if np.any(np.isfinite(latitude[:, c]))
            ]

            if not valid_cols:
                raise ValueError("No valid latitude track found.")

            ref_col = valid_cols[len(valid_cols) // 2]

            lat = latitude[:, ref_col]
            lon = longitude[:, ref_col]

    lat = _interp_nan_1d(lat)
    lon = _interp_nan_1d(lon)

    d = np.zeros(len(lat))

    d[1:] = _haversine_km(
        lat[:-1],
        lon[:-1],
        lat[1:],
        lon[1:],
    )

    return np.cumsum(d)


def split_left_right_swaths(
    cross_track_distance: np.ndarray,
    remove_edges_km: Optional[float] = None,
):
    """
    Return boolean masks for the two SWOT swaths.

    The masks are column masks.
    """

    xt = np.asarray(cross_track_distance, dtype=float)

    if xt.ndim == 2:
        xt = np.nanmedian(xt, axis=0)

    left = xt < 0
    right = xt > 0

    if remove_edges_km is not None:
        for name, mask in (("left", left), ("right", right)):
            idx = np.where(mask)[0]

            if idx.size < 2:
                continue

            x = np.abs(xt[idx])

            # Remove approximately the requested amount from the
            # outer cross-track edge.
            if x.size > 1:
                spacing = np.nanmedian(np.diff(np.sort(x)))
            else:
                spacing = 0.0

            cutoff = np.nanmax(x) - remove_edges_km

            if name == "left":
                left = left & (np.abs(xt) <= cutoff)
            else:
                right = right & (np.abs(xt) <= cutoff)

    return left, right


# ============================================================================
# Segment selection
# ============================================================================
#
# Segments are sliced by a *fixed number of native-grid rows*, not by a
# distance (km) window. Distance-based windows (the previous approach)
# produce a slightly different row count per segment, because along-track
# sample spacing is never perfectly uniform -- and since the alias
# decomposition's output wavenumber axis length is exactly
# (patch_rows // Ma), a varying row count means a varying wavenumber-axis
# length per segment, which breaks any attempt to stack segments together
# (np.stack, xarray, np.mean over segments, ...). Fixing the row count
# instead guarantees every segment produced with the same
# (segment_length_km, dx1_native_km, dx1_expert_km) -- across swaths, and
# across every file in a batch run -- has an identical wavenumber axis.

def _resolve_segment_sampling(
    segment_length_km: float,
    dx1_native_km: float,
    dx1_expert_km: float,
):
    """
    Turn a segment length in km into an exact, fixed number of native-grid
    rows per segment.

    Returns
    -------
    n_lines_patch : int
        Exact number of native along-track rows per segment. Every segment
        bounds tuple (i0, i1) produced downstream satisfies
        i1 - i0 == n_lines_patch.
    Ma : int
        Native -> Expert along-track decimation factor
        (round(dx1_expert_km / dx1_native_km)).
    n1e_target : int
        Number of Expert-grid along-track samples per segment
        (= n_lines_patch // Ma). This is the guaranteed length of the
        one-sided wavenumber / direct_psd / aliased_psd / total_psd arrays
        `alias_decompose_patch` will return for every segment produced with
        these parameters.
    """
    Ma = max(1, int(round(dx1_expert_km / dx1_native_km)))

    n1e_target = max(4, int(round(segment_length_km / dx1_expert_km)))

    n_lines_patch = n1e_target * Ma

    return n_lines_patch, Ma, n1e_target


def _segment_bounds_fixed_samples(
    n_rows: int,
    n_lines_patch: int,
    overlap: float = 0.0,
):
    """
    Fixed-length (row-index) segment bounds.

    Every returned (i0, i1) satisfies i1 - i0 == n_lines_patch exactly, so
    every patch sliced from these bounds decimates onto the same number of
    Expert-grid wavenumber bins (see `_resolve_segment_sampling`).
    """

    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1).")

    if n_lines_patch < 1:
        raise ValueError("n_lines_patch must be >= 1.")

    if n_rows < n_lines_patch:
        return []

    step = max(1, int(round(n_lines_patch * (1.0 - overlap))))

    last_start = n_rows - n_lines_patch

    starts = list(range(0, last_start + 1, step))

    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    return [(s, s + n_lines_patch) for s in starts]


def _segment_bounds_at_latitudes(
    latitude: np.ndarray,
    target_latitudes: Sequence[float],
    n_lines_patch: int,
):
    """
    Create fixed-length (`n_lines_patch` rows) segments centered on
    specified latitudes -- e.g. to align Unsmoothed segments with an
    already-computed Expert product's segment centroids. Every returned
    bound has exactly n_lines_patch rows, same guarantee as
    `_segment_bounds_fixed_samples`.
    """

    latitude = _interp_nan_1d(latitude)

    n_rows = len(latitude)
    half = n_lines_patch // 2

    bounds = []

    for target_lat in target_latitudes:

        center_idx = int(
            np.argmin(np.abs(latitude - target_lat))
        )

        i0 = center_idx - half
        i1 = i0 + n_lines_patch

        if i0 < 0 or i1 > n_rows:
            continue

        bounds.append((i0, i1))

    return bounds


# ============================================================================
# Gap handling
# ============================================================================

def _max_contiguous_gap_fraction(
    patch: np.ndarray,
    axis: int = 0,
    row_bad_threshold: float = 0.5,
) -> float:
    """
    Longest contiguous run of "mostly missing" rows (or columns, if
    axis=1), expressed as a fraction of that axis's length.

    A row is "mostly missing" when more than `row_bad_threshold` of its
    entries (across the *other* dimension) are NaN. This is a majority
    vote across the cross-track width, not a per-pixel check -- so a
    small number of chronically-flagged pixels (e.g. a persistently-bad
    near-nadir or far-swath edge column, common in real quality-flagged
    altimetry data) do not, by themselves, poison the metric. What this
    is meant to catch is the case where the swath is *genuinely* blocked
    for a stretch of along-track samples -- a coastline cutting across
    most of the swath, an island, an orbit/instrument dropout -- which is
    exactly the situation where linear interpolation (`_fill_nan_2d`) is
    least trustworthy: it silently draws a straight line across a real
    physical gap, suppressing genuine small-scale variance there and
    potentially injecting spurious low-wavenumber ramp energy. Segments
    are screened on this metric (via `max_gap_fraction`) *before* any
    filling happens, so land/coastal voids and large gaps get dropped
    rather than interpolated over.

    Using a single chronically-bad pixel's run length here (rather than a
    row-majority vote) was tried first and turned out to be far too
    aggressive in practice: a single always-NaN column anywhere in the
    patch (again, a routine occurrence, not a real gap) drives that
    column's run length to 100% of the segment, which rejects every
    segment in the file regardless of the actual data quality.

    axis=0 (default) checks along-track runs of mostly-blocked rows (the
    along-track spectrum is what this codebase estimates, so this is the
    metric that matters most for spectral bias); axis=1 checks
    cross-track runs of mostly-blocked columns.
    """

    mask = np.isnan(patch)

    if axis == 1:
        mask = mask.T

    n, width = mask.shape

    if n == 0 or width == 0:
        return 0.0

    row_nan_fraction = mask.mean(axis=1)
    row_bad = row_nan_fraction > row_bad_threshold

    if not row_bad.any():
        return 0.0

    # Longest contiguous run of True in row_bad (1-D run-length via a
    # reset-on-False cumulative counter).
    b = row_bad.astype(np.int64)
    run = np.empty_like(b)
    run[0] = b[0]

    for i in range(1, n):
        run[i] = (run[i - 1] + b[i]) * b[i]

    return float(run.max()) / n


def _fill_nan_2d(
    patch: np.ndarray,
    max_nan_fraction: float = 0.15,
):
    """
    Fill small 2-D NaN gaps using row then column interpolation.

    This performs *linear* interpolation across gaps, which is only a
    reasonable approximation for short, scattered gaps (typical KaRIn
    per-pixel dropouts). It is not, by itself, gap-size aware -- callers
    are expected to have already screened out patches with large
    contiguous voids (land, coastlines, orbit gaps) via
    `_max_contiguous_gap_fraction` / `max_gap_fraction` before calling
    this, since interpolating across a large real gap silently damps the
    small-scale spectral content that this codebase is trying to measure.
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


# ============================================================================
# Plane detrending
# ============================================================================

def plane_detrend_2d(patch: np.ndarray) -> np.ndarray:
    """
    Remove a best-fit 2-D plane.

    This is preferable to independent 1-D detrending here because SWOT
    swath geometry can introduce a broad cross-track bowl.
    """

    n1, n2 = patch.shape

    y, x = np.mgrid[0:n1, 0:n2]

    A = np.column_stack(
        [
            np.ones(n1 * n2),
            x.ravel(),
            y.ravel(),
        ]
    )

    coeffs, *_ = np.linalg.lstsq(
        A,
        patch.ravel(),
        rcond=None,
    )

    return patch - (A @ coeffs).reshape(n1, n2)


# ============================================================================
# Native 2-D spectrum
# ============================================================================

def periodogram_2d(
    patch: np.ndarray,
    dx1_km: float,
    dx2_km: float,
):
    """
    Two-sided 2-D Hann-tapered PSD.

    Returns
    -------
    k1, k2, S2D
    """

    n1, n2 = patch.shape

    window = np.outer(
        np.hanning(n1),
        np.hanning(n2),
    )

    norm = np.mean(window ** 2)

    F = np.fft.fft2(patch * window)

    S2D = (
        dx1_km * dx2_km
        / (n1 * n2)
        * np.abs(F) ** 2
        / norm
    )

    k1 = np.fft.fftfreq(n1, dx1_km)
    k2 = np.fft.fftfreq(n2, dx2_km)

    return k1, k2, S2D


# ============================================================================
# Expert filtering
# ============================================================================

def hamming_transfer_function(
    wavenumber: np.ndarray,
    dx_km: float,
    n_taps: int = 9,
):
    """
    Frequency response of a centered Hamming moving-average filter.

    Replace this function if your exact Expert filtering kernel is known
    and differs from this definition.
    """

    n_taps = int(n_taps)

    if n_taps < 1:
        return np.ones_like(wavenumber)

    if n_taps == 1:
        return np.ones_like(wavenumber)

    window = np.hamming(n_taps)
    window = window / window.sum()

    # FIR frequency response.
    omega = 2.0 * np.pi * wavenumber * dx_km

    n = np.arange(n_taps)

    H = np.sum(
        window[None, :]
        * np.exp(-1j * omega[:, None] * n[None, :]),
        axis=1,
    )

    return np.abs(H) ** 2


# ============================================================================
# Alias decomposition
# ============================================================================

def alias_decompose_patch(
    patch: np.ndarray,
    dx1_native_km: float,
    dx2_native_km: float,
    dx1_expert_km: float,
    dx2_expert_km: float,
    n_taps: int = 9,
):
    """
    Compute direct and aliased Expert-resolution along-track spectra.

    The input is an Unsmoothed 2-D patch.

    Returns
    -------
    k
        One-sided Expert along-track wavenumber.

    direct
        Direct contribution.

    aliased
        Aliased contribution.

    total = direct + aliased.
    """

    Ma = int(round(dx1_expert_km / dx1_native_km))
    Mc = int(round(dx2_expert_km / dx2_native_km))

    Ma = max(Ma, 1)
    Mc = max(Mc, 1)

    n1, n2 = patch.shape

    n1_trim = (n1 // Ma) * Ma
    n2_trim = (n2 // Mc) * Mc

    patch = patch[:n1_trim, :n2_trim]

    n1, n2 = patch.shape

    if n1 < 4 * Ma or n2 < 4 * Mc:
        return None

    k1, k2, S_native = periodogram_2d(
        patch,
        dx1_native_km,
        dx2_native_km,
    )

    H1 = hamming_transfer_function(
        k1,
        dx1_native_km,
        n_taps,
    )

    H2 = hamming_transfer_function(
        k2,
        dx2_native_km,
        n_taps,
    )

    S_filtered = (
        S_native
        * H1[:, None]
        * H2[None, :]
    )

    n1e = n1 // Ma
    n2e = n2 // Mc

    # ---------------------------------------------------------------
    # Fold native spectral replicas onto the Expert grid.
    # ---------------------------------------------------------------

    folded = S_filtered.reshape(
        Ma,
        n1e,
        n2,
    ).sum(axis=0)

    folded = folded.reshape(
        n1e,
        Mc,
        n2e,
    ).sum(axis=1)

    # r=0, s=0 = direct contribution.
    direct_2d = S_filtered[:n1e, :n2e]

    # ---------------------------------------------------------------
    # Integrate cross-track wavenumber.
    # ---------------------------------------------------------------

    dk2 = np.abs(k2[1] - k2[0])

    total_1d = folded.sum(axis=1) * dk2
    direct_1d = direct_2d.sum(axis=1) * dk2

    aliased_1d = np.maximum(
        total_1d - direct_1d,
        0.0,
    )

    k_expert = np.fft.fftfreq(
        n1e,
        dx1_expert_km,
    )

    # ---------------------------------------------------------------
    # Convert two-sided spectrum to one-sided spectrum.
    # ---------------------------------------------------------------

    positive = k_expert >= 0

    k = k_expert[positive]

    direct = direct_1d[positive].copy()
    total = total_1d[positive].copy()

    # DC and Nyquist are not doubled.
    interior = (
        (k > 0)
        & (k < np.max(np.abs(k_expert)))
    )

    direct[interior] *= 2.0
    total[interior] *= 2.0

    aliased = np.maximum(
        total - direct,
        0.0,
    )

    return k, direct, aliased, total


# ============================================================================
# Segment extraction + alias spectrum
# ============================================================================

def compute_alias_swath_spectra(
    ssha: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    swath_mask: np.ndarray,
    swath_name: str,
    segment_length_km: float,
    dx1_native_km: float,
    dx2_native_km: float,
    dx1_expert_km: float,
    dx2_expert_km: float,
    month: Optional[int] = None,
    overlap: float = 0.0,
    max_nan_fraction: float = 0.15,
    max_gap_fraction: float = 0.1,
    n_taps: int = 9,
    remove_plane: bool = True,
    center_latitudes: Optional[Sequence[float]] = None,
):
    """
    Compute AliasSegmentSpectrum objects for one SWOT swath.

    All returned segments share the same wavenumber-axis length: segments
    are sliced to a fixed number of native-grid rows (derived from
    segment_length_km, dx1_native_km, dx1_expert_km; see
    `_resolve_segment_sampling`), not a distance window, so the alias
    decomposition's output axis length is identical for every segment --
    including segments from other swaths/files processed with the same
    parameters.

    NaN handling (land, coastlines, gaps)
    --------------------------------------
    Each candidate segment is screened by two independent, complementary
    checks before any gap-filling happens:

    - `max_nan_fraction`: total fraction of missing samples in the
      segment. Segments over land, or mostly over land, have a high total
      NaN fraction and are dropped here regardless of the gap's shape.
    - `max_gap_fraction`: the length of the single longest *contiguous*
      along-track run of missing samples, as a fraction of the segment
      length. This catches the case `max_nan_fraction` alone cannot: a
      segment that is well under the total-NaN budget but contains one
      large, spatially-contiguous void (a coastline cutting across the
      swath, an island, an orbit/instrument dropout). Such a void, if
      left to `_fill_nan_2d`'s linear interpolation, would silently draw
      a straight line across a real physical gap -- damping genuine
      small-scale variance there and biasing exactly the along-track
      wavenumber spectrum this function is trying to estimate.

    Only segments passing *both* checks reach `_fill_nan_2d`, so the
    linear interpolation it performs is only ever applied to short,
    scattered gaps (typical KaRIn per-pixel dropouts) -- the case it's
    actually a reasonable approximation for.
    """

    n_lines_patch, Ma, n1e_target = _resolve_segment_sampling(
        segment_length_km,
        dx1_native_km,
        dx1_expert_km,
    )

    cols = np.where(swath_mask)[0]

    if cols.size == 0:
        return AliasPassSpectrumResult(
            swath=swath_name,
            wavenumber=np.array([]),
            mean_direct_psd=np.array([]),
            mean_aliased_psd=np.array([]),
            mean_total_psd=np.array([]),
            segments=[],
            n_segments_used=0,
            n_segments_total=0,
            month=month,
        )

    sub_ssha = ssha[:, cols]
    sub_lat = latitude[:, cols]
    sub_lon = longitude[:, cols]

    distance = along_track_distance_km(
        sub_lat,
        sub_lon,
    )

    ref_lat = sub_lat[:, sub_lat.shape[1] // 2]

    if center_latitudes is not None:

        bounds = _segment_bounds_at_latitudes(
            ref_lat,
            center_latitudes,
            n_lines_patch,
        )

    else:

        bounds = _segment_bounds_fixed_samples(
            sub_ssha.shape[0],
            n_lines_patch,
            overlap,
        )

    segments = []

    n_dropped_nan_fraction = 0
    n_dropped_gap = 0
    n_dropped_fill_failed = 0
    n_dropped_decompose_failed = 0

    for segment_index, (i0, i1) in enumerate(bounds):

        patch = sub_ssha[i0:i1, :]

        valid_fraction = float(
            np.isfinite(patch).mean()
        )

        if valid_fraction < 1.0 - max_nan_fraction:
            n_dropped_nan_fraction += 1
            continue

        # Gate on the largest contiguous along-track gap BEFORE any
        # filling: this is what actually catches land/coastal voids and
        # large data gaps that would otherwise pass the total-fraction
        # check above and get silently interpolated over. See the
        # NaN-handling note in this function's docstring.
        gap_fraction = _max_contiguous_gap_fraction(patch, axis=0)

        if gap_fraction > max_gap_fraction:
            n_dropped_gap += 1
            continue

        filled = _fill_nan_2d(
            patch,
            max_nan_fraction=max_nan_fraction,
        )

        if filled is None:
            n_dropped_fill_failed += 1
            continue

        gap_filled = np.isnan(patch).any()

        if remove_plane:
            filled = plane_detrend_2d(filled)

        result = alias_decompose_patch(
            filled,
            dx1_native_km,
            dx2_native_km,
            dx1_expert_km,
            dx2_expert_km,
            n_taps=n_taps,
        )

        if result is None:
            n_dropped_decompose_failed += 1
            continue

        k, direct, aliased, total = result

        lat_chunk = sub_lat[i0:i1]
        lon_chunk = sub_lon[i0:i1]

        lat_mean = float(np.nanmean(lat_chunk))
        lat_min = float(np.nanmin(lat_chunk))
        lat_max = float(np.nanmax(lat_chunk))
        lon_mean = float(np.nanmean(lon_chunk))

        segments.append(
            AliasSegmentSpectrum(
                swath=swath_name,
                segment_index=segment_index,

                wavenumber=k,

                direct_psd=direct,
                aliased_psd=aliased,
                total_psd=total,

                lat_mean=lat_mean,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_mean=lon_mean,

                along_track_distance_start_km=float(
                    distance[i0]
                ),
                along_track_distance_end_km=float(
                    distance[i1 - 1]
                ),

                n_lines=i1 - i0,
                n_pixels=filled.shape[1],

                valid_fraction=valid_fraction,
                gap_filled=gap_filled,
                gap_fraction=gap_fraction,
            )
        )

    # Defensive check: with fixed-sample segmentation every segment should
    # already share the same wavenumber-axis length (== n1e_target). This
    # guards against that invariant being silently broken by a future code
    # change, rather than letting a mismatched-length segment reach
    # np.stack()/segments_to_dataset() and fail there instead.
    if segments:

        lengths = {len(s.wavenumber) for s in segments}

        if len(lengths) > 1:

            counts = {
                length: sum(1 for s in segments if len(s.wavenumber) == length)
                for length in lengths
            }

            majority_length = max(counts, key=counts.get)

            warnings.warn(
                f"[{swath_name}] Segments with inconsistent wavenumber "
                f"lengths {sorted(lengths)} were produced (expected a "
                f"single length of {n1e_target}); keeping only the "
                f"{counts[majority_length]} segments of length "
                f"{majority_length} and dropping the rest. This should "
                "not normally happen -- please report it.",
                RuntimeWarning,
            )

            segments = [
                s for s in segments if len(s.wavenumber) == majority_length
            ]

    if not segments:

        return AliasPassSpectrumResult(
            swath=swath_name,
            wavenumber=np.array([]),
            mean_direct_psd=np.array([]),
            mean_aliased_psd=np.array([]),
            mean_total_psd=np.array([]),
            segments=[],
            n_segments_used=0,
            n_segments_total=len(bounds),
            month=month,
            n_segments_dropped_nan_fraction=n_dropped_nan_fraction,
            n_segments_dropped_gap=n_dropped_gap,
            n_segments_dropped_fill_failed=n_dropped_fill_failed,
            n_segments_dropped_decompose_failed=n_dropped_decompose_failed,
        )

    k = segments[0].wavenumber

    direct = np.mean(
        np.stack([s.direct_psd for s in segments]),
        axis=0,
    )

    aliased = np.mean(
        np.stack([s.aliased_psd for s in segments]),
        axis=0,
    )

    total = direct + aliased

    return AliasPassSpectrumResult(
        swath=swath_name,

        wavenumber=k,

        mean_direct_psd=direct,
        mean_aliased_psd=aliased,
        mean_total_psd=total,

        segments=segments,

        n_segments_used=len(segments),
        n_segments_total=len(bounds),

        month=month,

        n_segments_dropped_nan_fraction=n_dropped_nan_fraction,
        n_segments_dropped_gap=n_dropped_gap,
        n_segments_dropped_fill_failed=n_dropped_fill_failed,
        n_segments_dropped_decompose_failed=n_dropped_decompose_failed,
    )


# ============================================================================
# Full pass
# ============================================================================

def compute_alias_pass_spectra(
    data: dict,
    segment_length_km: float,
    dx1_native_km: float,
    dx2_native_km: float,
    dx1_expert_km: float,
    dx2_expert_km: float,
    month: Optional[int] = None,
    overlap: float = 0.0,
    max_nan_fraction: float = 0.15,
    max_gap_fraction: float = 0.1,
    n_taps: int = 9,
    remove_plane: bool = True,
    remove_edges_km: Optional[float] = None,
    center_latitudes: Optional[dict] = None,
):
    """
    Compute direct, aliased and total spectra for both SWOT swaths.

    See `compute_alias_swath_spectra` for the NaN/gap-handling contract:
    `max_nan_fraction` bounds the total missing-data fraction per segment
    (drops segments mostly/entirely over land), and `max_gap_fraction`
    separately bounds the longest *contiguous* along-track gap (drops
    segments with a large coastal/orbit void even when the total NaN
    fraction is otherwise fine). Only segments passing both are gap-filled
    and used.
    """

    left_mask, right_mask = split_left_right_swaths(
        data["cross_track_distance"],
        remove_edges_km=remove_edges_km,
    )

    results = {}

    for swath, mask in (
        ("left", left_mask),
        ("right", right_mask),
    ):

        targets = (
            center_latitudes.get(swath)
            if center_latitudes is not None
            else None
        )

        results[swath] = compute_alias_swath_spectra(
            ssha=data["ssha"],
            latitude=data["latitude"],
            longitude=data["longitude"],
            swath_mask=mask,
            swath_name=swath,

            segment_length_km=segment_length_km,

            dx1_native_km=dx1_native_km,
            dx2_native_km=dx2_native_km,
            dx1_expert_km=dx1_expert_km,
            dx2_expert_km=dx2_expert_km,

            month=month,
            overlap=overlap,
            max_nan_fraction=max_nan_fraction,
            max_gap_fraction=max_gap_fraction,
            n_taps=n_taps,

            remove_plane=remove_plane,

            center_latitudes=targets,
        )

    return results


# ============================================================================
# Optional simple file loader
# ============================================================================

def load_swot_l2_unsmoothed(
    filepath,
    ssh_var="ssha_karin_2",
    HRET=True,
):
    """
    Load SWOT L2 LR Unsmoothed data from the left/right groups.
    """

    if xr is None:
        raise ImportError(
            "xarray is required to load SWOT NetCDF files."
        )

    def load_group(group_name, flip):

        ds = xr.open_dataset(
            filepath,
            group=group_name,
        )

        if HRET:
            ssha = (
                ds[ssh_var]
                + ds["height_cor_xover"]
                + ds["internal_tide_hret"]
            ).values
        else:
            ssha = (
                ds[ssh_var]
                + ds["height_cor_xover"]
            ).values

        for q in (
            f"{ssh_var}_qual",
            "ssha_karin_2_qual",
            "ssh_karin_2_qual",
        ):
            if q in ds:
                ssha = np.where(
                    np.asarray(ds[q]) == 0,
                    ssha,
                    np.nan,
                )
                break

        for q in (
            "ancillary_surface_classification_flag",
            "surface_classification_flag",
        ):
            if q in ds:
                ssha = np.where(
                    np.asarray(ds[q]) == 0,
                    ssha,
                    np.nan,
                )
                break

        if "surface_type" in ds:
            ssha = np.where(
                np.asarray(ds["surface_type"]) == 0,
                ssha,
                np.nan,
            )

        lat = ds["latitude"].values.astype(float)
        lon = ds["longitude"].values.astype(float)
        xt = ds["cross_track_distance"].values.astype(float)

        sign = -1.0 if group_name == "left" else 1.0
        xt = sign * np.abs(xt)

        ds.close()

        if flip:
            ssha = ssha[:, ::-1]
            lat = lat[:, ::-1]
            lon = lon[:, ::-1]

            if xt.ndim == 2:
                xt = xt[:, ::-1]
            else:
                xt = xt[::-1]

        return ssha, lat, lon, xt

    left = load_group("left", flip=True)
    right = load_group("right", flip=False)

    ssha = np.concatenate(
        [left[0], right[0]],
        axis=1,
    )

    lat = np.concatenate(
        [left[1], right[1]],
        axis=1,
    )

    lon = np.concatenate(
        [left[2], right[2]],
        axis=1,
    )

    if left[3].ndim == 2:
        xt = np.concatenate(
            [left[3], right[3]],
            axis=1,
        )
    else:
        xt = np.concatenate(
            [left[3], right[3]]
        )

    return {
        "ssha": ssha,
        "latitude": lat,
        "longitude": lon,
        "cross_track_distance": xt,
    }


# ============================================================================
# Convenience: flatten results like all_segments_unsmooth
# ============================================================================

def flatten_alias_segments(
    results: dict,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
):
    """
    Return a simple list of AliasSegmentSpectrum objects.

    This is the alias equivalent of your `all_segments_unsmooth`.
    """

    output = []

    for result in results.values():

        for segment in result.segments:

            if lat_min is not None and segment.lat_mean < lat_min:
                continue

            if lat_max is not None and segment.lat_mean > lat_max:
                continue

            if lon_min is not None and segment.lon_mean < lon_min:
                continue

            if lon_max is not None and segment.lon_mean > lon_max:
                continue

            output.append(segment)

    return output


# ============================================================================
# Packing segments into an xarray.Dataset
# ============================================================================

def segments_to_dataset(
    segments: Sequence[AliasSegmentSpectrum],
) -> "xr.Dataset":
    """
    Pack a flat list of AliasSegmentSpectrum objects (e.g. from
    `flatten_alias_segments`) into a single xarray.Dataset with dimensions
    ("segment", "wavenumber").

    Every segment must share the same wavenumber-axis length. With
    `compute_alias_pass_spectra` / `compute_alias_swath_spectra` this is
    guaranteed for every segment produced with the same
    (segment_length_km, dx1_native_km, dx1_expert_km) -- including
    segments from different swaths and different files -- so results from
    many granules can be concatenated directly with this function (e.g.
    `xr.concat([segments_to_dataset(f) for f in per_file_segments],
    dim="segment")`, or simply passing the combined segment list from all
    files at once, as done here).

    Raises
    ------
    ValueError
        If the segments do not all share the same wavenumber-axis length
        (this would mean they came from calls with different
        segment_length_km / dx1_native_km / dx1_expert_km).
    """

    if xr is None:
        raise ImportError(
            "xarray is required for segments_to_dataset()."
        )

    if not segments:
        raise ValueError("No segments to pack.")

    lengths = {len(s.wavenumber) for s in segments}

    if len(lengths) > 1:
        raise ValueError(
            f"Segments have inconsistent wavenumber-axis lengths "
            f"{sorted(lengths)}; they must all come from calls with "
            "identical segment_length_km, dx1_native_km and "
            "dx1_expert_km."
        )

    wavenumber = segments[0].wavenumber

    direct_psd = np.stack([s.direct_psd for s in segments], axis=0)
    aliased_psd = np.stack([s.aliased_psd for s in segments], axis=0)
    total_psd = np.stack([s.total_psd for s in segments], axis=0)

    return xr.Dataset(
        data_vars=dict(
            direct_psd=(
                ["segment", "wavenumber"],
                direct_psd,
                {
                    "long_name": "Direct (unaliased) along-track SSHA PSD",
                    "units": "m^2 / (cycles/km)",
                },
            ),
            aliased_psd=(
                ["segment", "wavenumber"],
                aliased_psd,
                {
                    "long_name": "Aliased along-track SSHA PSD",
                    "units": "m^2 / (cycles/km)",
                },
            ),
            total_psd=(
                ["segment", "wavenumber"],
                total_psd,
                {
                    "long_name": "Total (direct + aliased) along-track SSHA PSD",
                    "units": "m^2 / (cycles/km)",
                },
            ),
            lat_mean=(
                ["segment"],
                [s.lat_mean for s in segments],
                {"units": "degrees_north"},
            ),
            lat_min=(
                ["segment"],
                [s.lat_min for s in segments],
                {"units": "degrees_north"},
            ),
            lat_max=(
                ["segment"],
                [s.lat_max for s in segments],
                {"units": "degrees_north"},
            ),
            lon_mean=(
                ["segment"],
                [s.lon_mean for s in segments],
                {"units": "degrees_east"},
            ),
            along_track_distance_start_km=(
                ["segment"],
                [s.along_track_distance_start_km for s in segments],
                {"units": "km"},
            ),
            along_track_distance_end_km=(
                ["segment"],
                [s.along_track_distance_end_km for s in segments],
                {"units": "km"},
            ),
            n_lines=(["segment"], [s.n_lines for s in segments]),
            n_pixels=(["segment"], [s.n_pixels for s in segments]),
            valid_fraction=(["segment"], [s.valid_fraction for s in segments]),
            gap_filled=(["segment"], [bool(s.gap_filled) for s in segments]),
            gap_fraction=(
                ["segment"],
                [
                    s.gap_fraction if s.gap_fraction is not None else np.nan
                    for s in segments
                ],
                {
                    "long_name": "Longest contiguous along-track NaN run, "
                                  "as a fraction of the segment length",
                },
            ),
            swath=(["segment"], [s.swath for s in segments]),
            segment_index=(["segment"], [s.segment_index for s in segments]),
        ),
        coords=dict(
            wavenumber=("wavenumber", wavenumber, {"units": "cycles/km"}),
            segment=("segment", np.arange(len(segments))),
        ),
    )