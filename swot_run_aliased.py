#!/usr/bin/env python
"""
SWOT SSH spectra — batch (sbatch) version.

Converted from run_aliased.ipynb. Differences
from the notebook, and why:

  * No interactive-only cells (progress(), the bare `client` widget
    display, plt.show()). Instead: plain logging + figures saved to disk
    with matplotlib's non-interactive 'Agg' backend.
  * Processes one month (one DATA_SUBDIR) at a time, and skips a month
    entirely if its output file already exists. This is what makes a
    5h30 job resumable: if the job dies partway (node failure, walltime
    hit, anything), just resubmit the same sbatch script and it picks
    up where it left off instead of starting over.
  * Runs fully detached from any client connection, so brief network/
    SSH/JupyterHub disconnects — the likely cause of the previous
    failure — can no longer kill it. That's the main point of moving
    this to sbatch.
  * Uses load_swot_l2_unsmoothed (adjust back to load_swot_l2_expert
    below if you actually want the Expert product on Levante instead).
  * Pins ALONG_TRACK_SPACING_KM explicitly (see earlier ValueError:
    "all input arrays must have the same shape" — caused by nperseg
    drifting across files/swaths when spacing was auto-estimated).

USAGE
-----
    python -u run_swot_spectra.py

(the -u is also set in the sbatch script; harmless to have twice)
"""
import traceback
import glob
import os
from pathlib import Path
import functools
import logging
import re
import sys
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")  # no display available on a compute node
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
                        
from dask.distributed import Client, LocalCluster, as_completed

from swot_analysis.swot_alias_wrapper import process_one_file
from swot_analysis.swot_alias_spectra import segments_to_dataset


from dask.distributed import Client, LocalCluster, wait


# ═══════════════════════════════════════════════════════════════════════════
# 1. Configuration — edit this block for each run
# ═══════════════════════════════════════════════════════════════════════════

# ── Paths (Levante) ─────────────────────────────────────────────────────────
DATA_DIR = "/work/uo0122/u241387/SWOT_SSH/SSH_Unsmoothed/"
SAVE_DIR = "/work/uo0122/u241387/SWOT_SSH/diags_2024/"
LOG_FILE = os.path.join(SAVE_DIR, "run_swot_spectra.log")

# One sub-directory per month of SWOT data.
DATA_SUBDIRS = [
    "SWOT_L2_LR_SSH_Unsmoothed_0124/",
    "SWOT_L2_LR_SSH_Unsmoothed_0224/",
    "SWOT_L2_LR_SSH_Unsmoothed_0324/",
    "SWOT_L2_LR_SSH_Unsmoothed_0424/",
    "SWOT_L2_LR_SSH_Unsmoothed_0524/",
    "SWOT_L2_LR_SSH_Unsmoothed_0624/",
    "SWOT_L2_LR_SSH_Unsmoothed_0724/",
    "SWOT_L2_LR_SSH_Unsmoothed_0824/",
    "SWOT_L2_LR_SSH_Unsmoothed_0924/",
    "SWOT_L2_LR_SSH_Unsmoothed_1024/",
    "SWOT_L2_LR_SSH_Unsmoothed_1124/",
    "SWOT_L2_LR_SSH_Unsmoothed_1224/",
]
FILE_PATTERN = "SWOT_L2_LR_SSH_Unsmoothed_*.nc"
DATA_SUBDIR_OVERRIDES = {}  # e.g. {'my_weird_folder_name': (2025, 7)}

# ── Region of interest ───────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -60.0, 60.0
LON_MIN, LON_MAX = 0.0, 360.0

# ── Spectral parameters ──────────────────────────────────────────────────────
SEGMENT_LENGTH_KM = 1000.0
OVERLAP = 0.5
DETREND = "constant"

SEGMENT_LENGTH_KM = 1000.0
OVERLAP = 0.0

MAX_NAN_FRACTION = 0.15
MAX_GAP_FRACTION = 0.1
N_TAPS = 17

DX1_NATIVE_KM = 0.25
DX2_NATIVE_KM = 0.25

DX1_EXPERT_KM = 2.0
DX2_EXPERT_KM = 2.0

HRET = False
SSH_VAR = "ssha_karin_2"

N_WORKERS = 1  # max(1, (os.cpu_count() or 2) - 1)
THREADS_PER_WORKER = 1
MEMORY_LIMIT = "5GB"

# ═══════════════════════════════════════════════════════════════════════════
# 2. Logging setup
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(SAVE_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("swot_spectra")


def parse_year_month(subdir_name, overrides=None):
    if overrides and subdir_name in overrides:
        return overrides[subdir_name]
    name = subdir_name.rstrip("/").rstrip("\\")
    match = re.search(r"(\d{2})(\d{2})$", name)
    if not match:
        raise ValueError(
            f"Could not infer year/month from subdirectory name '{subdir_name}'. "
            f"Add an explicit entry to DATA_SUBDIR_OVERRIDES instead."
        )
    mm, yy = match.groups()
    month = int(mm)
    year = 2000 + int(yy)
    if not (1 <= month <= 12):
        raise ValueError(f"Parsed invalid month={month} from '{subdir_name}'.")
    return year, month

# ═══════════════════════════════════════════════════════════════════════════
# 3. Worker function (unchanged logic from the notebook, unsmoothed loader)
# ═══════════════════════════════════════════════════════════════════════════

def run_one_file(filepath):
    filepath = Path(filepath)
    if OUTPUT_DIR is not None:
        output_file = OUTPUT_DIR / f"{filepath.stem}_alias.pkl"
    else:
        output_file=None
    try:
        result = process_one_file(
            filepath,
            output_file,

            segment_length_km=SEGMENT_LENGTH_KM,
            overlap=OVERLAP,

            max_nan_fraction=MAX_NAN_FRACTION,
            max_gap_fraction=MAX_GAP_FRACTION,
            n_taps=N_TAPS,

            dx1_native_km=DX1_NATIVE_KM,
            dx2_native_km=DX2_NATIVE_KM,
            dx1_expert_km=DX1_EXPERT_KM,
            dx2_expert_km=DX2_EXPERT_KM,

            hret=HRET,
            ssh_var=SSH_VAR,
        )

        return {
            "file": str(filepath),
            "result": result,
            "error": None,
        }

    except Exception:
        return {
            "file": str(filepath),
            "result": None,
            "error": traceback.format_exc(),
        }

# ═══════════════════════════════════════════════════════════════════════════
# 4. Main — per-month processing with checkpointing
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log = setup_logging()
    t_start = time.time()

    log.info("Starting Dask LocalCluster: n_workers=%d threads_per_worker=%d memory_limit=%s",
              N_WORKERS, THREADS_PER_WORKER, MEMORY_LIMIT)
    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        memory_limit=MEMORY_LIMIT,
        dashboard_address=":0",  # random free port; avoids clashing with other jobs
    )
    client = Client(cluster)
    log.info("Dask dashboard (tunnel if needed): %s", client.dashboard_link)

    worker_fn = functools.partial(
            run_one_file,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    monthly_segment_files = []

    for subdir in DATA_SUBDIRS:
        year, month = parse_year_month(subdir, overrides=DATA_SUBDIR_OVERRIDES)
        out_path = os.path.join(SAVE_DIR, f"swot_segments_{year}{month:02d}.nc")
        monthly_segment_files.append(out_path)

        if os.path.exists(out_path):
            log.info("[%d-%02d] output already exists, skipping: %s", year, month, out_path)
            continue

        filepaths = sorted(glob.glob(os.path.join(DATA_DIR, subdir, FILE_PATTERN)))[:]
        log.info("[%d-%02d] %d files found in %s", year, month, len(filepaths), subdir)
        if not filepaths:
            log.warning("[%d-%02d] no files found, skipping.", year, month)
            continue

        t_month = time.time()
        futures = client.map(worker_fn, filepaths)
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            filepath = Path(result["file"])

            if result["error"] is None:
                continue
        #print(f"[{i}/{len(filepaths)}] OK   {filepath.name}")
            else:
                print(f"[{i}/{len(filepath)}] FAIL {filepath.name}")
                print(result["error"])
        all_segments = {
            "left": [],
            "right": [],
        }

    for entry in results:
        if entry["error"] is not None:
            continue

        result = entry["result"]

        for swath in ("left", "right"):
            all_segments[swath].extend(result[swath])

#        log.info("[%d-%02d] %d segments retained, %d file(s) skipped (%.1f min)",
#                  year, month, len(all_segments), len(skipped),
#                  (time.time() - t_month) / 60)
#        for name, err in skipped:
#            log.warning("[%d-%02d] skipped %s:\n%s", year, month, name, err)

        if not all_segments:
            log.warning("[%d-%02d] no segments retained — nothing to save for this month.",
                        year, month)
            continue

        ds_month = segments_to_dataset(all_segments["left"] + all_segments["right"])
        ds_month.to_netcdf(out_path, mode="w")
        log.info("[%d-%02d] saved %s", year, month, out_path)

    client.close()
    cluster.close()
    log.info("Dask cluster closed.")

    # ── Combine all available monthly files into the final datasets ─────────
    existing = [f for f in monthly_segment_files if os.path.exists(f)]
    if not existing:
        log.error("No monthly segment files were produced — nothing to combine. Exiting.")
        return

    log.info("Combining %d monthly segment file(s) into final dataset...", len(existing))
    ds_segments = xr.concat([xr.open_dataset(f) for f in existing], dim="segment")
    log.info("Combined segment dataset: %s", dict(ds_segments.sizes))

    ds_segments.to_netcdf(os.path.join(SAVE_DIR, "swot_segments_1000km_notide_demean.nc"), mode="w")
    log.info("Saved combined segment dataset to %s", SAVE_DIR)

    # ── Diagnostics ───────────────────────────────────────────────────────
    log.info("Segments per swath: %s",
              {sw: int((ds_segments["swath"] == sw).sum()) for sw in ("left", "right")})
    log.info("Unique granules: %d", len(np.unique(ds_segments["granule"].values)))

