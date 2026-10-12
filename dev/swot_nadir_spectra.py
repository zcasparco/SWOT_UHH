"""
Along-track SSHA wavenumber spectra from SWOT nadir (20 Hz) granules, parallelised with dask.

Output (one NetCDF per batch of files + optional combined dataset):
    dims:   segment, wavenumber
    vars:   psd(segment, wavenumber)         [m^2 / (cycles/km)]
            lat_mean(segment), lon_mean(segment), time_mean(segment), file_index(segment)
"""
import glob
import os
import traceback

import dask
import numpy as np
import xarray as xr
from scipy import signal

# >>> import your own helpers (the ones used in your original script) <<<
from your_module import along_track_distance_km, _segment_bounds  # noqa: E402


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
PARAMS = dict(
    # MUST be fixed across files so every spectrum has the same wavenumber axis.
    # Check the median spacing of one granule (~0.33-0.34 km at 20 Hz) and set it here.
    along_track_spacing_km=0.34,
    segment_length_km=1000.0,
    overlap=0.5,
    window="hann",
    max_gap_fraction=0.25,
    detrend="constant",
)


def circular_mean_lon(lon):
    """Mean longitude that is safe across the 0/360 or +-180 wrap."""
    r = np.deg2rad(lon[np.isfinite(lon)])
    if r.size == 0:
        return np.nan
    m = np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    return m % 360 if np.nanmin(lon) >= 0 else (m + 180) % 360 - 180


def wavenumber_axis(p=PARAMS):
    nperseg = max(int(round(p["segment_length_km"] / p["along_track_spacing_km"])), 8)
    return np.fft.rfftfreq(nperseg, d=p["along_track_spacing_km"])  # cycles / km


# --------------------------------------------------------------------------- #
# Core: spectra for a single track (your loop, wrapped in a function)
# --------------------------------------------------------------------------- #
def track_spectra(lat, lon, ssha, time, p=PARAMS):
    """Returns dict of arrays with a leading 'segment' axis (possibly of length 0)."""
    dx = p["along_track_spacing_km"]
    L = p["segment_length_km"]
    gap = p["max_gap_fraction"]
    nperseg = max(int(round(L / dx)), 8)
    win = signal.get_window(p["window"], nperseg)
    win_norm = (win ** 2).mean()
    nk = nperseg // 2 + 1

    out = dict(psd=[], lat_mean=[], lon_mean=[], time_mean=[])

    def _empty():
        return dict(psd=np.empty((0, nk), np.float32),
                    lat_mean=np.empty(0), lon_mean=np.empty(0),
                    time_mean=np.empty(0, "datetime64[ns]"))

    # drop samples with unusable coordinates (distance computation needs them)
    keep = np.isfinite(lat) & np.isfinite(lon)
    lat, lon, ssha, time = lat[keep], lon[keep], ssha[keep], time[keep]
    if lat.size < 8:
        return _empty()

    distance_km = along_track_distance_km(lat, lon)
    bounds = _segment_bounds(distance_km, L, overlap=p["overlap"],
                             center_distances_km=None)

    overrun_tol_km = max(dx * 10, 0.01 * L)
    edge_tol_km = gap * L

    for i0, i1 in bounds:
        col = ssha[i0:i1]
        seg_dist = distance_km[i0:i1]
        if col.shape[0] < 8:
            continue

        uniform_dist = seg_dist[0] + np.arange(nperseg) * dx
        if uniform_dist[-1] > seg_dist[-1] + overrun_tol_km:
            continue

        valid = ~np.isnan(col)
        valid_frac = float(valid.mean())
        if valid_frac < (1.0 - gap) or valid.sum() < 8:
            continue

        good_dist, good_val = seg_dist[valid], col[valid]
        if (max(0.0, good_dist[0] - uniform_dist[0]) > edge_tol_km or
                max(0.0, uniform_dist[-1] - good_dist[-1]) > edge_tol_km):
            continue

        resampled = np.interp(uniform_dist, good_dist, good_val,
                              left=good_val[0], right=good_val[-1])
        if p["detrend"]:
            resampled = signal.detrend(resampled, type=p["detrend"])

        _, pxx = signal.periodogram(resampled * win, fs=1.0 / dx, window="boxcar",
                                    detrend=False, scaling="density")
        out["psd"].append((pxx / win_norm).astype(np.float32))
        out["lat_mean"].append(np.nanmean(lat[i0:i1]))
        out["lon_mean"].append(circular_mean_lon(lon[i0:i1]))

        t = time[i0:i1]
        t = t[~np.isnat(t)]
        out["time_mean"].append(
            t.astype("datetime64[ns]").astype("int64").mean().astype("int64").astype("datetime64[ns]")
            if t.size else np.datetime64("NaT", "ns"))

    if not out["psd"]:
        return _empty()
    return dict(psd=np.stack(out["psd"]),
                lat_mean=np.asarray(out["lat_mean"]),
                lon_mean=np.asarray(out["lon_mean"]),
                time_mean=np.asarray(out["time_mean"], dtype="datetime64[ns]"))


# --------------------------------------------------------------------------- #
# One granule -> spectra
# --------------------------------------------------------------------------- #
def process_file(path, p=PARAMS):
    """Safe for dask: never raises, returns (result_dict | None, error_str | None)."""
    try:
        # coordinates live in data_20, ssha in the data_20/ku subgroup
        with xr.open_dataset(path, group="data_20") as ds_c, \
             xr.open_dataset(path, group="data_20/ku") as ds_k:
            lat = ds_c["latitude"].values.astype("float64")
            lon = ds_c["longitude"].values.astype("float64")
            time = ds_c["time"].values
            ssha = ds_k["ssha"].values.astype("float64")
            # TODO: apply quality flags here, e.g.
            # ssha = np.where(ds_k["ssha_qual"].values == 0, ssha, np.nan)
        return track_spectra(lat, lon, ssha, time, p), None
    except Exception:
        return None, f"{path}\n{traceback.format_exc()}"


# --------------------------------------------------------------------------- #
# Driver: batches of files -> NetCDF
# --------------------------------------------------------------------------- #
def run_batch(files, start_index, p=PARAMS):
    tasks = [dask.delayed(process_file)(f, p) for f in files]
    results = dask.compute(*tasks)

    psd, lat_m, lon_m, t_m, fidx, errors = [], [], [], [], [], []
    for i, (res, err) in enumerate(results):
        if err:
            errors.append(err)
            continue
        n = res["psd"].shape[0]
        if n == 0:
            continue
        psd.append(res["psd"]); lat_m.append(res["lat_mean"])
        lon_m.append(res["lon_mean"]); t_m.append(res["time_mean"])
        fidx.append(np.full(n, start_index + i))

    k = wavenumber_axis(p)
    if psd:
        ds = xr.Dataset(
            {"psd": (("segment", "wavenumber"), np.concatenate(psd))},
            coords={
                "wavenumber": ("wavenumber", k, {"units": "cycles/km"}),
                "lat_mean": ("segment", np.concatenate(lat_m)),
                "lon_mean": ("segment", np.concatenate(lon_m)),
                "time_mean": ("segment", np.concatenate(t_m)),
                "file_index": ("segment", np.concatenate(fidx)),
            },
        )
        ds["psd"].attrs.update(units="m^2 / (cycles/km)", long_name="SSHA PSD")
    else:
        ds = None
    return ds, errors


def main(file_glob, outdir, batch_size=500, p=PARAMS):
    files = sorted(glob.glob(file_glob))
    os.makedirs(outdir, exist_ok=True)

    # persist the file list so 'file_index' can be mapped back to a filename
    with open(os.path.join(outdir, "files.txt"), "w") as fh:
        fh.write("\n".join(files))

    all_errors = []
    for b in range(0, len(files), batch_size):
        target = os.path.join(outdir, f"spectra_{b:06d}.nc")
        if os.path.exists(target):          # allows restarting an interrupted run
            continue
        ds, errs = run_batch(files[b:b + batch_size], b, p)
        all_errors += errs
        if ds is not None:
            ds.to_netcdf(target)
        print(f"batch {b}-{b + batch_size}: "
              f"{0 if ds is None else ds.sizes['segment']} segments, {len(errs)} errors")

    if all_errors:
        with open(os.path.join(outdir, "errors.log"), "w") as fh:
            fh.write("\n\n".join(all_errors))

    # combined view (lazy)
    return xr.open_mfdataset(os.path.join(outdir, "spectra_*.nc"),
                             concat_dim="segment", combine="nested")


if __name__ == "__main__":
    # Optional: use a distributed cluster (local or SLURM/PBS, etc.)
    # from dask.distributed import Client
    # client = Client(n_workers=16, threads_per_worker=1)   # 1 thread: scipy/numpy are GIL-friendly but memory-heavy
    ds = main("/path/to/SWOT_L2_LR_NALT*/*.nc", "spectra_out", batch_size=500)
    print(ds)
