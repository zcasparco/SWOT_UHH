from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from swot_analysis.swot_alias_spectra import (
    load_swot_l2_unsmoothed,
    compute_alias_pass_spectra,
    flatten_alias_segments,
    segments_to_dataset,
)


# =============================================================================
# Default configuration (used only when this file is run as a script)
# =============================================================================

#INPUT_FILE = Path(
#    "/path/to/SWOT_L2_LR_SSH_..._Unsmoothed.nc"
#)

#OUTPUT_FILE = Path(
#    "/path/to/swot_alias_spectra.pkl"
#)


# -----------------------------------------------------------------------------
# Segment configuration
# -----------------------------------------------------------------------------

SEGMENT_LENGTH_KM = 250.0
OVERLAP = 0.0

MAX_NAN_FRACTION = 0.15
MAX_GAP_FRACTION = 0.1

N_TAPS = 9

REMOVE_PLANE = True#


REMOVE_EDGES_KM = None


# -----------------------------------------------------------------------------
# Native / Expert sampling
#
# Replace these with the values appropriate for your SWOT products.
# -----------------------------------------------------------------------------

DX1_NATIVE_KM = 0.25
DX2_NATIVE_KM = 0.25#

DX1_EXPERT_KM = 2.0
DX2_EXPERT_KM = 2.0


# -----------------------------------------------------------------------------
# Optional geographic selection
# -----------------------------------------------------------------------------

LAT_MIN = -70.0
LAT_MAX = 70.0

LON_MIN = 0.0
LON_MAX = 360.0


# =============================================================================
# Per-file processing entry point
#
# This is what run_aliased.ipynb (and anything else that wants to process a
# single Unsmoothed file, e.g. from a ProcessPoolExecutor worker) should call.
# =============================================================================

def process_one_file(
    unsmoothed_path,
    output_path=None,
    *,
    segment_length_km: float = SEGMENT_LENGTH_KM,
    overlap: float = OVERLAP,
    max_nan_fraction: float = MAX_NAN_FRACTION,
    max_gap_fraction: float = MAX_GAP_FRACTION,
    max_gap_km=None,
    n_taps: int = N_TAPS,
    dx1_native_km: float = DX1_NATIVE_KM,
    dx2_native_km: float = DX2_NATIVE_KM,
    dx1_expert_km: float = DX1_EXPERT_KM,
    dx2_expert_km: float = DX2_EXPERT_KM,
    remove_plane: bool = REMOVE_PLANE,
    remove_edges_km=REMOVE_EDGES_KM,
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
    lon_min: float = LON_MIN,
    lon_max: float = LON_MAX,
    hret: bool = True,
    ssh_var: str = "ssha_karin_2",
    verbose: bool = False,
    return_counts: bool = False,
    full_direct: bool = False,
    # Accepted for backwards compatibility with older notebook cells; not
    # currently used by the alias-decomposition itself.
    max_patches=None,
    band_km=None,
):
    """
    Run the full alias-decomposition pipeline on a single SWOT L2 LR
    Unsmoothed file.

    Returns a dict with keys "left" and "right", each mapping to a list of
    ``AliasSegmentSpectrum`` objects (i.e. directly usable with
    ``all_segments[swath].extend(result[swath])``).

    Every segment -- across both swaths, and across every file processed
    with the same segment_length_km / dx1_native_km / dx1_expert_km -- has
    an identical wavenumber-axis length (see
    ``swot_alias_spectra._resolve_segment_sampling``), so segments from
    many files can always be stacked/concatenated together, e.g. via
    ``swot_alias_spectra.segments_to_dataset``.

    NaN handling (land, coastlines, gaps)
    --------------------------------------
    Two independent thresholds gate which candidate segments are kept, so
    land, coastal voids, and large data gaps can't sneak past a single
    check and get linearly interpolated over:

    - ``max_nan_fraction``: drops a segment if its *total* fraction of
      missing samples is too high (segments over/mostly-over land).
    - ``max_gap_fraction``: drops a segment if its longest single
      *contiguous* along-track run of missing samples is too large, even
      if the total fraction is fine -- this is what catches a coastline
      or orbit gap that would otherwise pass ``max_nan_fraction`` and get
      silently smoothed over by linear interpolation, damping small-scale
      spectral content. See ``compute_alias_swath_spectra`` for details.

    Only segments passing both checks are gap-filled (short, scattered
    per-pixel dropouts only) and used.

    If ``verbose`` is True, a one-line-per-swath summary of how many
    candidate segments were kept/dropped (and why) is printed -- the same
    counts are always included in the pickled payload's
    ``"segment_counts"`` when ``output_path`` is given.

    If ``return_counts`` is True, returns ``(result, segment_counts)``
    instead of just ``result`` -- useful when a caller wants the
    kept/dropped diagnostics per file (e.g. for a batch-run manifest)
    without also having to write/re-read a pickle.

    If ``output_path`` is given, the full result (including the per-swath
    ``AliasPassSpectrumResult`` objects, a packed ``xarray.Dataset`` of the
    kept segments, and metadata) is also pickled there.
    """

    unsmoothed_path = Path(unsmoothed_path)

    data = load_swot_l2_unsmoothed(
        unsmoothed_path,
        ssh_var=ssh_var,
        HRET=hret,
    )
    if full_direct:
        by_swath, by_swath_full = compute_alias_pass_spectra(
            data=data,
    
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
    
            remove_plane=remove_plane,
    
            remove_edges_km=remove_edges_km,
    
            full_direct=full_direct
        )
    
        all_segments = flatten_alias_segments(
            by_swath,
    
            lat_min=lat_min,
            lat_max=lat_max,
    
            lon_min=lon_min,
            lon_max=lon_max,
        )
        result = {
            swath: by_swath[swath].segments
            for swath in ("left", "right")
        }
        result_full = {
            swath: by_swath_full[swath].segments
            for swath in ("left", "right")
        }
    else:
        by_swath = compute_alias_pass_spectra(
            data=data,
    
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
    
            remove_plane=remove_plane,
    
            remove_edges_km=remove_edges_km,
    
            full_direct=full_direct
        )
        all_segments = flatten_alias_segments(
            by_swath,
    
            lat_min=lat_min,
            lat_max=lat_max,
    
            lon_min=lon_min,
            lon_max=lon_max,
        )
        result = {
            swath: by_swath[swath].segments
            for swath in ("left", "right")
        }

    # Segment counts/diagnostics per swath, mainly useful for QC across a
    # large batch of files: how many candidate segments were dropped, and
    # for which of the two independent NaN-related reasons (total fraction
    # vs. a large contiguous gap -- see compute_alias_swath_spectra).
    segment_counts = {
            swath: {
                "n_segments_total": by_swath[swath].n_segments_total,
                "n_segments_used": by_swath[swath].n_segments_used,
                "n_segments_dropped_nan_fraction": by_swath[swath].n_segments_dropped_nan_fraction,
                "n_segments_dropped_gap": by_swath[swath].n_segments_dropped_gap,
                "n_segments_dropped_fill_failed": by_swath[swath].n_segments_dropped_fill_failed,
                "n_segments_dropped_decompose_failed": by_swath[swath].n_segments_dropped_decompose_failed,
            }
            for swath in ("left", "right")
        }
    if verbose:
        for swath in ("left", "right"):
            c = segment_counts[swath]
            print(
                f"{swath:5s}: kept {c['n_segments_used']}/{c['n_segments_total']}  "
                f"(dropped: nan_fraction={c['n_segments_dropped_nan_fraction']}, "
                f"gap={c['n_segments_dropped_gap']}, "
                f"fill_failed={c['n_segments_dropped_fill_failed']}, "
                f"decompose_failed={c['n_segments_dropped_decompose_failed']})"
            )

    if output_path is not None:

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = None
        if all_segments:
            try:
                dataset = segments_to_dataset(all_segments)
            except Exception:
                # Packing is a convenience, not required for the pickle to
                # be useful; fall back to the raw segment list if it fails
                # for any reason (e.g. xarray not installed).
                dataset = None

        payload = {
            "segments": all_segments,
            "dataset": dataset,
            "by_swath": by_swath,
            "segment_counts": segment_counts,
            "metadata": {
                "input_file": str(unsmoothed_path),
                "segment_length_km": segment_length_km,
                "overlap": overlap,
                "dx1_native_km": dx1_native_km,
                "dx2_native_km": dx2_native_km,
                "dx1_expert_km": dx1_expert_km,
                "dx2_expert_km": dx2_expert_km,
                "n_taps": n_taps,
                "max_nan_fraction": max_nan_fraction,
                "max_gap_fraction": max_gap_fraction,
                "max_gap_km": max_gap_km,
                "remove_plane": remove_plane,
            },
        }

        with open(output_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    if return_counts:
        return result, segment_counts
    if full_direct:
        return result, result_full
    else:
        return result


# =============================================================================
# Script entry point (single fixed file, defined via the module-level
# constants above)
# =============================================================================

def main():

    print(f"Loading:\n  {INPUT_FILE}")
    print("Computing alias decomposition...")

    result = process_one_file(
        INPUT_FILE,
        OUTPUT_FILE,

        segment_length_km=SEGMENT_LENGTH_KM,
        overlap=OVERLAP,

        max_nan_fraction=MAX_NAN_FRACTION,
        max_gap_fraction=MAX_GAP_FRACTION,
        n_taps=N_TAPS,

        dx1_native_km=DX1_NATIVE_KM,
        dx2_native_km=DX2_NATIVE_KM,
        dx1_expert_km=DX1_EXPERT_KM,
        dx2_expert_km=DX2_EXPERT_KM,

        remove_plane=REMOVE_PLANE,
        remove_edges_km=REMOVE_EDGES_KM,

        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        lon_min=LON_MIN,
        lon_max=LON_MAX,

        verbose=True,
    )

    print()

    for swath in ("left", "right"):
        segments = result[swath]
        print(f"{swath:5s}: {len(segments)} segments retained")

    total = len(result["left"]) + len(result["right"])
    print()
    print(f"Total retained segments: {total}")

    if total:
        first_swath = "left" if result["left"] else "right"
        s = result[first_swath][0]

        print("\nFirst segment")
        print("-----------------------------")
        print(f"swath       : {s.swath}")
        print(f"segment     : {s.segment_index}")
        print(f"latitude    : {s.lat_mean:.4f}")
        print(f"longitude   : {s.lon_mean:.4f}")
        print(
            f"track range: "
            f"{s.along_track_distance_start_km:.2f} - "
            f"{s.along_track_distance_end_km:.2f} km"
        )
        print(f"n lines     : {s.n_lines}")
        print(f"n pixels    : {s.n_pixels}")
        print(f"valid       : {s.valid_fraction:.3f}")

        print()
        print(f"k bins      : {len(s.wavenumber)}")
        print(
            f"k range     : "
            f"{s.wavenumber[1]:.5f} - "
            f"{s.wavenumber[-1]:.5f} cycles/km"
        )

    print()
    print(f"Saved:\n  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
