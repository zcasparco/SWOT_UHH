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

import glob
import os

import numpy as np
import xarray as xr


class SWOTGridAccumulator:
    """
    Incrementally compute gridded means from SWOT L2 LR Expert files.

    Each variable is accumulated independently while sharing the same file read.
    """

    def __init__(
        self,
        variables,
        dlon=0.25,
        dlat=0.25,
        lon_range=(-180, 180),
        lat_range=(-60, 60),
        HRET=True,
        mean_sq = True
    ):

        if isinstance(variables, str):
            variables = [variables]

        self.variables = variables
        self.HRET = HRET
        self.mean_sq = mean_sq

        self.dlon = dlon
        self.dlat = dlat

        self.lon_edges = np.arange(lon_range[0], lon_range[1] + dlon, dlon)
        self.lat_edges = np.arange(lat_range[0], lat_range[1] + dlat, dlat)

        self.lon_centers = 0.5 * (
            self.lon_edges[:-1] + self.lon_edges[1:]
        )
        self.lat_centers = 0.5 * (
            self.lat_edges[:-1] + self.lat_edges[1:]
        )

        self.nlon = len(self.lon_centers)
        self.nlat = len(self.lat_centers)

        self.sum = {
            var: np.zeros((self.nlat, self.nlon), dtype=np.float64)
            for var in self.variables
        }

        self.count = {
            var: np.zeros((self.nlat, self.nlon), dtype=np.int64)
            for var in self.variables
        }

    def _load_variable(self, ds, variable):

        if variable in (
            "ssha_karin",
            "ssha_karin_2",
            "ssh_karin",
            "ssh_karin_2",
        ):

            if self.HRET:

                values = (
                    ds[variable]
                    + ds["height_cor_xover"]
                    + ds["internal_tide_hret"]
                ).values

            else:

                values = (
                    ds[variable]
                    + ds["height_cor_xover"]
                ).values

        else:

            values = ds[variable].values

        # Variable quality flag
        for qual in (
            f"{variable}_qual",
            "ssha_karin_qual",
            "ssh_karin_2_qual",
        ):

            if qual in ds:

                values = np.where(
                    ds[qual].values == 0,
                    values,
                    np.nan,
                )

                break
            

        return values

    def add_file(self, filepath):

        with xr.open_dataset(filepath) as ds:

            lat = ds.latitude.values
            lon = ((ds.longitude.values + 180) % 360) - 180

            # Ocean mask

            ocean = np.ones(lat.shape, dtype=bool)

            for scf in (
                "ancillary_surface_classification_flag",
                "surface_classification_flag",
            ):

                if scf in ds:
                    ocean &= ds[scf].values == 0
                    break

            if "surface_type" in ds:
                ocean &= ds["surface_type"].values == 0

            finite_ll = (
                np.isfinite(lat)
                & np.isfinite(lon)
            )

            base_mask = ocean & finite_ll

            if not np.any(base_mask):
                return

            lat = lat[base_mask]
            lon = lon[base_mask]

            ilat = np.searchsorted(
                self.lat_edges,
                lat,
                side="right",
            ) - 1

            ilon = np.searchsorted(
                self.lon_edges,
                lon,
                side="right",
            ) - 1

            inside = (
                (ilat >= 0)
                & (ilat < self.nlat)
                & (ilon >= 0)
                & (ilon < self.nlon)
            )

            ilat = ilat[inside]
            ilon = ilon[inside]

            flat = ilat * self.nlon + ilon

            for variable in self.variables:

                values = self._load_variable(ds, variable)

                values = values[base_mask][inside]

                valid = np.isfinite(values)

                if not np.any(valid):
                    continue

                    
                flat_valid = flat[valid]
                
                self.sum[variable] += np.bincount(
                    flat_valid,
                    weights=values[valid],
                    minlength=self.nlat * self.nlon,
                ).reshape(self.nlat, self.nlon)

                self.count[variable] += np.bincount(
                    flat_valid,
                    minlength=self.nlat * self.nlon,
                ).reshape(self.nlat, self.nlon)

    def add_files(self, filepaths):

        for filepath in filepaths:
            self.add_file(filepath)

    def merge(self, other):

        for variable in self.variables:

            self.sum[variable] += other.sum[variable]
            self.count[variable] += other.count[variable]

        return self

    def __iadd__(self, other):

        return self.merge(other)

    def __add__(self, other):

        new = SWOTGridAccumulator(
            variables=self.variables,
            dlon=self.dlon,
            dlat=self.dlat,
            lon_range=(
                self.lon_edges[0],
                self.lon_edges[-1],
            ),
            lat_range=(
                self.lat_edges[0],
                self.lat_edges[-1],
            ),
            HRET=self.HRET,
        )

        for variable in self.variables:

            new.sum[variable] = (
                self.sum[variable]
                + other.sum[variable]
            )

            new.count[variable] = (
                self.count[variable]
                + other.count[variable]
            )

        return new

    def to_dataset(self):

        data_vars = {}

        for variable in self.variables:

            mean = np.full(
                (self.nlat, self.nlon),
                np.nan,
                dtype=np.float64,
            )

            mask = self.count[variable] > 0

            mean[mask] = (
                self.sum[variable][mask]
                / self.count[variable][mask]
            )

            data_vars[f"mean_{variable}"] = (
                ("latitude", "longitude"),
                mean,
            )

            data_vars[f"nobs_{variable}"] = (
                ("latitude", "longitude"),
                self.count[variable],
            )

        return xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                latitude=self.lat_centers,
                longitude=self.lon_centers,
            ),
        )
def load_swot_l2_expert(filepath : str, ssh_var : str='ssha_karin_2', HRET : bool=True, other_var=None):
    # NO type annotations — avoids Python 3.14 PEP 649 __annotate__ capture bug
    ds = xr.open_dataset(filepath)
    if HRET:
        ssha = (ds[ssh_var] + ds['height_cor_xover'] + ds['internal_tide_hret']).values
    else:
        ssha   = np.array(ds[ssh_var] + ds['height_cor_xover'], dtype='float64')
        
    for qual_name in (f'{ssh_var}_qual', 'ssha_karin_qual', 'ssh_karin_2_qual'):
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
    xtrack = np.array(ds['cross_track_distance'], dtype='float64')
    xtrack = ds['cross_track_distance'].values
    
    ds.close()
    if other_var:
        dict_others = {other_name : ds[other_name].values for other_name in other_var}
        dict_out = {'ssha': ssha, 'latitude': lat, 'longitude': lon,'cross_track_distance': xtrack}
        dict_out.update(dict_others)
        return dict_out
    else:
        return {'ssha': ssha, 'latitude': lat, 'longitude': lon,'cross_track_distance': xtrack}


def build_box_mean(dict_var: dict,
                   var: np.array,
                   box_deg: float = 2.0,
                   min_sampling_fraction: float = 0.5) -> xr.Dataset:

    from collections import defaultdict

    lat     = dict_var['latitude'].values
    lon     = dict_var['longitude'].values
    lat_idx = np.floor(lat / box_deg).astype(int)
    lon_idx = np.floor(lon / box_deg).astype(int)

    box_to_segs = defaultdict(list)
    for i, key in enumerate(zip(lat_idx, lon_idx)):
        box_to_segs[key].append(i)

    var_all    = dict_var[var].values
    rows_lat, rows_lon, rows_n, rows_var = [], [], [], []

    for (li, loi), idxs in sorted(box_to_segs.items()):
        rows_psd.append(np.nanmean(var_all[idxs]))
        rows_n.append(len(idxs))
        rows_lat.append((li  + 0.5) * box_deg)
        rows_lon.append((loi + 0.5) * box_deg)

    n_points = np.array(rows_n)
    n_max      = n_points.max()
    keep       = n_points >= min_sampling_fraction * n_max

    var_stack  = np.stack(rows_var, axis=0)
    lat_center = np.array(rows_lat)
    lon_center = np.array(rows_lon)

    return xr.Dataset(
        data_vars=dict(
            mean_var=(['box'], var_stack[keep],
                      {'long_name': 'Box-mean'}),
            n_points=(['box'], n_points[keep]),
            sampling_fraction=(['box'], n_points[keep] / n_max),
            lat_center=(['box'], lat_center[keep], {'units': 'degrees_north'}),
            lon_center=(['box'], lon_center[keep], {'units': 'degrees_east'}),
        ),
        coords=dict(box=('box', np.arange(keep.sum())),
        ),
        attrs=dict(box_size_deg=box_deg,
                   min_sampling_fraction=min_sampling_fraction),
    )