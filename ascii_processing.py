#!/usr/bin/env python3
"""Spatial CIMIS ASCII processing with selectable target grid.

This script converts Spatial CIMIS ASCII grids to NetCDF and supports two
target grid modes:

* "gridmet"  – reproject to the GridMET 4 km grid in EPSG:4326
* "native"   – keep the native footprint / resolution of the ASCII files

The target mode is controlled via the config file (key: ``target_mode``).
Other options mirror the classic ``spatial_cimis_processing_gridmet.py``
script, including optional mask creation, 2D output, and climatologies.

Example config snippet::

    target_mode = native  # or gridmet (default)

Author: UC Davis Global Environmental Change Lab (adapted)
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import rasterio
import geopandas as gpd
from netCDF4 import Dataset, date2num
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers and configuration
# ---------------------------------------------------------------------------


GRIDMET_TARGET = {
    "ncols": 247,
    "nrows": 227,
    "lon_min": -124.39166663333334,
    "lon_max": -114.14166663333334,
    "lat_min": 32.56666666666667,
    "lat_max": 41.983333333333334,
    "lon_res": 0.041666666666666664,
    "lat_res": 0.04166666666666666,
    "crs": "EPSG:4326",
}

DEFAULT_ASCII_CRS = "EPSG:3310"  # California Albers – fallback when ASC has no CRS

NATIVE_TARGET_PARAMS = {
    "ncols": 500,
    "nrows": 552,
    "xllcorner": -400000.0,
    "yllcorner": -650000.0,
    "cellsize": 2000.0,
    "crs": "EPSG:3310",
}


@dataclass
class TargetGrid:
    """Container describing the target grid/transform/coordinates."""

    name: str
    crs: str
    transform: Affine
    shape: Tuple[int, int]
    coord_names: Tuple[str, str]
    coord_arrays: Dict[str, np.ndarray]


def join_path(base: str, filename: str) -> str:
    return os.path.join(base, filename)


def read_asc_header(filename: str) -> Dict[str, float]:
    header: Dict[str, float] = {}
    with open(filename, "r") as f:
        for _ in range(6):
            parts = f.readline().strip().split()
            if len(parts) >= 2:
                key = parts[0].lower()
                value = (
                    float(parts[1]) if ("." in parts[1] or "e" in parts[1].lower()) else int(parts[1])
                )
                header[key] = value
    return header


def dates_daily(y0: int, m0: int, d0: int, mtot: int, noleap: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    day = np.zeros((mtot))
    month = np.zeros((mtot))
    year = np.zeros((mtot))
    start_date = date(y0, m0, d0)
    delta = timedelta(days=1)
    current = start_date
    idx = 0
    while idx < mtot:
        day[idx] = current.day
        month[idx] = current.month
        year[idx] = current.year
        if noleap == 0:
            current = current + delta
        else:
            next_day = current + delta
            if next_day.month == 2 and next_day.day == 29:
                next_day = next_day + delta
            current = next_day
        idx += 1
    return day, month, year


# ---------------------------------------------------------------------------
# Target grid resolution helpers
# ---------------------------------------------------------------------------


def build_gridmet_target() -> TargetGrid:
    transform = Affine.translation(GRIDMET_TARGET["lon_min"], GRIDMET_TARGET["lat_max"]) * Affine.scale(
        GRIDMET_TARGET["lon_res"], -GRIDMET_TARGET["lat_res"]
    )

    lon = np.linspace(
        GRIDMET_TARGET["lon_min"],
        GRIDMET_TARGET["lon_max"],
        GRIDMET_TARGET["ncols"],
    )
    lat = np.linspace(
        GRIDMET_TARGET["lat_max"],
        GRIDMET_TARGET["lat_min"],
        GRIDMET_TARGET["nrows"],
    )

    return TargetGrid(
        name="gridmet",
        crs=GRIDMET_TARGET["crs"],
        transform=transform,
        shape=(GRIDMET_TARGET["nrows"], GRIDMET_TARGET["ncols"]),
        coord_names=("lat", "lon"),
        coord_arrays={"lat": lat, "lon": lon},
    )


def build_native_target(example_ascii: Optional[str] = None) -> TargetGrid:
    params = NATIVE_TARGET_PARAMS.copy()

    if example_ascii is not None:
        try:
            header = read_asc_header(example_ascii)
            mismatches = []
            for key in ("ncols", "nrows"):
                if key in header and int(header[key]) != int(params[key]):
                    mismatches.append(f"{key}: header={header[key]} config={params[key]}")
            if "cellsize" in header and float(header["cellsize"]) != float(params["cellsize"]):
                mismatches.append(
                    f"cellsize: header={header['cellsize']} config={params['cellsize']}"
                )
            if mismatches:
                print(
                    "  Warning: native target differs from ASCII header – using configured grid.\n"
                    f"    Differences: {', '.join(mismatches)}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not read ASCII header for native grid validation: {exc}")

    ncols = int(params["ncols"])
    nrows = int(params["nrows"])
    cellsize = float(params["cellsize"])
    x_origin = float(params["xllcorner"])
    y_origin = float(params["yllcorner"])

    transform = from_origin(x_origin, y_origin + cellsize * nrows, cellsize, cellsize)

    xs = x_origin + cellsize * (np.arange(ncols) + 0.5)
    ys = y_origin + cellsize * (np.arange(nrows) + 0.5)
    ys = ys[::-1]

    return TargetGrid(
        name="native",
        crs=params["crs"],
        transform=transform,
        shape=(nrows, ncols),
        coord_names=("y", "x"),
        coord_arrays={"x": xs, "y": ys},
    )


def iterate_possible_files(
    data_path: str, variable: str, start_year: int, end_year: int
) -> Iterable[str]:
    for yr in range(start_year, end_year + 1):
        ndays = 365 + calendar.isleap(yr)
        day_arr, month_arr, year_arr = dates_daily(yr, 1, 1, ndays, 0)
        for d, m, y in zip(day_arr.astype(int), month_arr.astype(int), year_arr.astype(int)):
            yield join_path(
                data_path,
                f"{variable}.{y:04d}-{m:02d}-{d:02d}.asc",
            )


def find_first_ascii_file(
    data_path: str, variable: str, start_year: int, end_year: int
) -> Optional[str]:
    for candidate in iterate_possible_files(data_path, variable, start_year, end_year):
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_target_grid(
    mode: str, data_path: str, variable: str, start_year: int, end_year: int
) -> TargetGrid:
    mode = mode.lower()
    if mode == "native":
        print("  Using configured native grid parameters")
        return build_native_target()

    print("  Using GridMET 4 km target grid")
    return build_gridmet_target()


def build_state_mask(shapefile_path: str, target: TargetGrid) -> Optional[np.ndarray]:
    if not shapefile_path:
        return None
    resolved_path = os.path.abspath(shapefile_path)
    if not os.path.exists(resolved_path):
        print(f"  Warning: shapefile {resolved_path} not found; skipping state mask.")
        return None

    try:
        gdf = gpd.read_file(resolved_path)
        if gdf.empty:
            print(f"  Warning: shapefile {resolved_path} is empty; skipping state mask.")
            return None
        gdf = gdf.to_crs(target.crs)
        shapes = [(geom, 1) for geom in gdf.geometry if geom and not geom.is_empty]
        if not shapes:
            print(f"  Warning: shapefile {resolved_path} has no valid geometries; skipping state mask.")
            return None

        mask = rasterize(
            shapes,
            out_shape=target.shape,
            transform=target.transform,
            fill=0,
            all_touched=True,  # Include cells that touch boundary (exclude only cells completely outside)
            dtype=np.uint8,
        )
        return mask
    except Exception as exc:  # noqa: BLE001
        print(f"  Warning: could not rasterize shapefile {resolved_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Reprojection helpers
# ---------------------------------------------------------------------------


def read_and_reproject_asc(filename: str, target: TargetGrid) -> Tuple[np.ndarray, bool, int]:
    """
    Read ASCII file, zero negative values (excluding -9999), then reproject if needed.
    Negative values are zeroed BEFORE reprojection.
    
    Returns:
        Tuple of (raster_data, was_reprojected, negative_cells_count)
    """
    target_shape = target.shape
    try:
        with rasterio.open(filename) as src:
            data = src.read(1)
            
            # Zero negative values (excluding -9999) BEFORE reprojection
            neg_mask = (data < 0) & (data != -9999)
            negative_count = int(np.count_nonzero(neg_mask))
            if np.any(neg_mask):
                data[neg_mask] = 0.0
            
            src_transform = src.transform
            src_crs = src.crs if src.crs else DEFAULT_ASCII_CRS

            needs_reproject = (
                data.shape != target_shape or src_transform != target.transform or src_crs != target.crs
            )

            if not needs_reproject:
                return data, False, negative_count

            target_data = np.full(target_shape, -9999, dtype=np.float32)

            reproject(
                source=data,
                destination=target_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=target.transform,
                dst_crs=target.crs,
                resampling=Resampling.average,
                src_nodata=-9999,
                dst_nodata=-9999,
            )

            return target_data, True, negative_count

    except Exception as exc:  # noqa: BLE001
        print(f"Error reading {filename}: {exc}")
        return np.full(target_shape, -9999, dtype=np.float32), False, 0


def process_variable_data(
    data_path: str,
    netcdf_path: str,
    variable: str,
    target: TargetGrid,
    yr_start: int,
    yr_end: int,
    state_mask: Optional[np.ndarray] = None,
) -> None:
    print(f"Processing {variable} ({target.name})...")

    os.makedirs(netcdf_path, exist_ok=True)

    coord_y, coord_x = target.coord_names
    y_array = target.coord_arrays[coord_y]
    x_array = target.coord_arrays[coord_x]

    negative_cells_total = 0
    negative_days_total = 0
    forward_fill_cells_total = 0
    forward_fill_days_total = 0
    native_total_files = 0
    native_reprojected_files = 0

    years = range(yr_start, yr_end + 1)
    for yr in years:
        print(f"  Year {yr}")
        ndays = 365 + calendar.isleap(yr)
        day_arr, month_arr, year_arr = dates_daily(yr, 1, 1, ndays, 0)

        data = np.full((ndays, *target.shape), -9999, dtype=np.float32)
        days_with_negative = np.zeros(ndays, dtype=bool)
        negative_cells_per_day = np.zeros(ndays, dtype=np.int32)
        days_with_reprojection = np.zeros(ndays, dtype=bool)
        found_any = False

        for t in range(ndays):
            path = join_path(data_path, f"{variable}.{int(year_arr[t]):04d}-{int(month_arr[t]):02d}-{int(day_arr[t]):02d}.asc")
            if not os.path.isfile(path):
                continue

            found_any = True
            # Read and reproject (negative values are zeroed inside read_and_reproject_asc BEFORE reprojection)
            raster, was_reprojected, neg_cells = read_and_reproject_asc(path, target)
            if was_reprojected:
                days_with_reprojection[t] = True
            if target.name == "native":
                native_total_files += 1
                if was_reprojected:
                    native_reprojected_files += 1

            # Track negative values that were zeroed before reprojection
            if neg_cells > 0:
                negative_cells_total += neg_cells
                negative_days_total += 1
                days_with_negative[t] = True
                negative_cells_per_day[t] = neg_cells

            # Check for any remaining negative values after reprojection (should be rare)
            # This handles edge cases where reprojection might introduce new negatives
            neg_mask = (raster < 0) & (raster != -9999)
            if np.any(neg_mask):
                additional_neg_cells = int(np.count_nonzero(neg_mask))
                negative_cells_total += additional_neg_cells
                if neg_cells == 0:  # Only increment day count if no negatives were found before reprojection
                    negative_days_total += 1
                    days_with_negative[t] = True
                negative_cells_per_day[t] += additional_neg_cells
                raster[neg_mask] = 0.0

            if state_mask is not None:
                valid_before_mask = np.sum(raster != -9999)
                raster[state_mask == 0] = -9999
                valid_after_mask = np.sum(raster != -9999)
                if t == 0:  # Print only for first file to avoid spam
                    print(f"    First file: {valid_before_mask} valid before mask, {valid_after_mask} valid after mask")

            data[t, :, :] = raster

        if not found_any:
            print(f"    Warning: no ASCII files found for {yr}")
            continue

        dates = [
            datetime(int(year_arr[i]), int(month_arr[i]), int(day_arr[i])) for i in range(ndays)
        ]
        times = date2num(dates, "days since 1900-01-01")

        coords = {
            "time": ("time", times),
            coord_y: (coord_y, y_array),
            coord_x: (coord_x, x_array),
        }

        xrds = xr.Dataset(
            data_vars={variable: (("time",) + target.coord_names, data)},
            coords=coords,
            attrs={
                "title": f"Spatial CIMIS {variable} ({target.name} grid)",
                "grid_mode": target.name,
                "grid_crs": target.crs,
            },
        )

        data_array = xrds[variable].where(xrds[variable] != -9999)
        
        # Debug: check valid data before forward fill
        valid_count_before_ffill = int(data_array.count().item())
        print(f"    Valid cells before forward fill: {valid_count_before_ffill}")

        missing_before = data_array.isnull()
        forward_filled = data_array.ffill("time")

        filled_mask = missing_before & ~forward_filled.isnull()
        try:
            filled_cells_year = int(filled_mask.sum().item())
        except ValueError:
            filled_cells_year = 0
        try:
            filled_days_year = int(filled_mask.any(dim=target.coord_names).sum().item())
        except ValueError:
            filled_days_year = 0

        # Track which days have forward-filled values
        days_with_forward_fill = filled_mask.any(dim=target.coord_names).values
        forward_fill_cells_per_day = filled_mask.sum(dim=target.coord_names).values.astype(np.int32)

        if filled_cells_year > 0:
            print(f"    Forward-filled {filled_cells_year} cells across {filled_days_year} timesteps")

        forward_fill_cells_total += filled_cells_year
        forward_fill_days_total += filled_days_year
        
        # Print summary of days with issues
        negative_day_indices = np.where(days_with_negative)[0]
        forward_fill_day_indices = np.where(days_with_forward_fill)[0]
        reprojection_day_indices = np.where(days_with_reprojection)[0]
        
        if len(reprojection_day_indices) > 0:
            print(f"    Days with reprojection: {len(reprojection_day_indices)} days")
            if len(reprojection_day_indices) <= 10:
                print(f"      Day indices: {reprojection_day_indices.tolist()}")
            else:
                print(f"      Day indices: {reprojection_day_indices[:10].tolist()} ... ({len(reprojection_day_indices)} total)")
        
        if len(negative_day_indices) > 0:
            print(f"    Days with negative values zeroed: {len(negative_day_indices)} days")
            if len(negative_day_indices) <= 10:
                print(f"      Day indices: {negative_day_indices.tolist()}")
            else:
                print(f"      Day indices: {negative_day_indices[:10].tolist()} ... ({len(negative_day_indices)} total)")
        
        if len(forward_fill_day_indices) > 0:
            print(f"    Days with forward-filled values: {len(forward_fill_day_indices)} days")
            if len(forward_fill_day_indices) <= 10:
                print(f"      Day indices: {forward_fill_day_indices.tolist()}")
            else:
                print(f"      Day indices: {forward_fill_day_indices[:10].tolist()} ... ({len(forward_fill_day_indices)} total)")

        xrds[variable] = forward_filled.fillna(-9999).astype(np.float32)
        xrds[variable].encoding.update({"_FillValue": -9999.0, "dtype": "float32"})
        
        # Set time coordinate attributes
        xrds["time"].attrs = {
            "units": "days since 1900-01-01",
            "long_name": "time",
            "standard_name": "time",
            "calendar": "gregorian"
        }
        
        # Add tracking variables to the dataset
        xrds["days_with_negative"] = (("time",), days_with_negative)
        xrds["days_with_negative"].attrs = {
            "long_name": "Days with negative values that were zeroed",
            "description": "Boolean array indicating which days had negative values (excluding -9999) that were set to zero"
        }
        
        xrds["negative_cells_per_day"] = (("time",), negative_cells_per_day)
        xrds["negative_cells_per_day"].attrs = {
            "long_name": "Number of negative cells zeroed per day",
            "description": "Count of cells with negative values (excluding -9999) that were set to zero for each day"
        }
        
        xrds["days_with_forward_fill"] = (("time",), days_with_forward_fill)
        xrds["days_with_forward_fill"].attrs = {
            "long_name": "Days with forward-filled values",
            "description": "Boolean array indicating which days had missing values filled using forward fill"
        }
        
        xrds["forward_fill_cells_per_day"] = (("time",), forward_fill_cells_per_day)
        xrds["forward_fill_cells_per_day"].attrs = {
            "long_name": "Number of forward-filled cells per day",
            "description": "Count of cells that were forward-filled for each day"
        }
        
        xrds["days_with_reprojection"] = (("time",), days_with_reprojection)
        xrds["days_with_reprojection"].attrs = {
            "long_name": "Days with reprojected data",
            "description": "Boolean array indicating which days had ASCII files that required reprojection to match the target grid"
        }

        yearly_filename = join_path(
            netcdf_path,
            f"spatial_cimis_{variable.lower()}_{yr}_" + ("4km" if target.name == "gridmet" else "2km") + ".nc",
        )
        if os.path.exists(yearly_filename):
            print(f"    Warning: Overwriting existing file {yearly_filename}")
        xrds.to_netcdf(yearly_filename)
        xrds.close()
        print(f"    Saved {yearly_filename}")

    if negative_cells_total > 0:
        print(f"  Negative values zeroed: {negative_cells_total} cells across {negative_days_total} timesteps")

    if forward_fill_cells_total > 0:
        print(
            f"  Forward-fill summary: {forward_fill_cells_total} cells across {forward_fill_days_total} timesteps filled using last valid value"
        )

    if target.name == "native":
        print("  Native processing summary:")
        print(f"    Files processed: {native_total_files}")
        print(f"    Files requiring reprojection: {native_reprojected_files}")

    print(f"Finished processing {variable} ({target.name})")


# ---------------------------------------------------------------------------
# Ancillary outputs (2D grid/time, daily & monthly climatologies)
# ---------------------------------------------------------------------------


def open_yearly_files(
    netcdf_path: str,
    variable: str,
    target: TargetGrid,
    start_year: int,
    end_year: int,
) -> Tuple[list[str], list[int]]:
    files: list[str] = []
    years: list[int] = []
    for yr in range(start_year, end_year + 1):
        path = join_path(netcdf_path, f"spatial_cimis_{variable.lower()}_{yr}_{target.name}.nc")
        if os.path.exists(path):
            files.append(path)
            years.append(yr)
    return files, years


def create_2d_files(
    netcdf_path: str,
    output_path: str,
    variable: str,
    target: TargetGrid,
    start_year: int,
    end_year: int,
) -> None:
    files, years = open_yearly_files(netcdf_path, variable, target, start_year, end_year)
    if not files:
        print("No yearly files found for 2D conversion.")
        return

    os.makedirs(output_path, exist_ok=True)

    for path, yr in zip(files, years):
        ds = xr.open_dataset(path)
        var_data = ds[variable]

        non_time_dims = var_data.dims[1:]
        stacked = var_data.stack(grid=non_time_dims).transpose("grid", "time")

        suffix = "4km" if target.name == "gridmet" else "2km"
        output_filename = join_path(output_path, f"spatial_cimis_{variable.lower()}_2d_{yr}_{suffix}.nc")

        if os.path.exists(output_filename):
            print(f"  Warning: Overwriting existing file {output_filename}")

        ncfile = Dataset(output_filename, mode="w", format="NETCDF4_CLASSIC")

        grid_dim = ncfile.createDimension("grid", stacked.shape[0])
        time_dim = ncfile.createDimension("time", stacked.shape[1])

        grid_var = ncfile.createVariable("grid", np.int32, ("grid",))
        time_var = ncfile.createVariable("time", np.float64, ("time",))
        time_var.units = "days since 1900-01-01"
        time_var.long_name = "time"

        data_var = ncfile.createVariable(variable, np.float32, ("grid", "time"), fill_value=-9999)

        grid_var[:] = np.arange(stacked.shape[0])
        time_var[:] = ds["time"].values
        data_var[:, :] = stacked.values

        data_var.units = ds[variable].attrs.get("units", "unknown")
        data_var.standard_name = ds[variable].attrs.get("standard_name", variable.lower())
        data_var.description = ds[variable].attrs.get("long_name", f"Daily {variable}")

        ncfile.title = f"Spatial CIMIS {variable} ({target.name} grid)"
        ncfile.history = "Created " + datetime.today().strftime("%Y-%m-%d")
        ncfile.grid_mode = target.name
        ncfile.grid_crs = target.crs

        ncfile.close()
        ds.close()
        print(f"Saved 2D file {output_filename}")


def create_daily_climatology(
    netcdf_path: str,
    output_path: str,
    variable: str,
    target: TargetGrid,
) -> None:
    files = [
        join_path(netcdf_path, f)
        for f in os.listdir(netcdf_path)
        if f.startswith(f"spatial_cimis_{variable.lower()}_") and f.endswith(f"_{target.name}.nc")
    ]

    if not files:
        print("No files found for daily climatology.")
        return

    ds = xr.open_mfdataset(files, combine="by_coords")
    var_data = ds[variable]
    clim = var_data.groupby("time.dayofyear").mean(dim="time")

    os.makedirs(output_path, exist_ok=True)
    suffix = "4km" if target.name == "gridmet" else "2km"
    filename = join_path(output_path, f"spatial_cimis_{variable.lower()}_daily_clim_2004-2023_{suffix}.nc")

    clim_ds = clim.to_dataset(name=variable)
    clim_ds[variable].attrs = var_data.attrs
    clim_ds.attrs["title"] = f"Daily climatology of {variable} ({target.name} grid)"
    clim_ds.attrs["grid_mode"] = target.name
    clim_ds.attrs["grid_crs"] = target.crs

    if os.path.exists(filename):
        print(f"  Warning: Overwriting existing file {filename}")

    clim_ds.to_netcdf(filename)
    ds.close()
    print(f"Saved daily climatology to {filename}")


def create_monthly_climatology(
    netcdf_path: str,
    output_path: str,
    variable: str,
    target: TargetGrid,
) -> None:
    files = [
        join_path(netcdf_path, f)
        for f in os.listdir(netcdf_path)
        if f.startswith(f"spatial_cimis_{variable.lower()}_") and f.endswith(f"_{target.name}.nc")
    ]

    if not files:
        print("No files found for monthly climatology.")
        return

    ds = xr.open_mfdataset(files, combine="by_coords")
    var_data = ds[variable]
    clim = var_data.groupby("time.month").mean(dim="time")

    os.makedirs(output_path, exist_ok=True)
    suffix = "4km" if target.name == "gridmet" else "2km"
    filename = join_path(output_path, f"spatial_cimis_{variable.lower()}_monthly_clim_2004-2023_{suffix}.nc")

    clim_ds = clim.to_dataset(name=variable)
    clim_ds[variable].attrs = var_data.attrs
    clim_ds.attrs["title"] = f"Monthly climatology of {variable} ({target.name} grid)"
    clim_ds.attrs["grid_mode"] = target.name
    clim_ds.attrs["grid_crs"] = target.crs

    if os.path.exists(filename):
        print(f"  Warning: Overwriting existing file {filename}")

    clim_ds.to_netcdf(filename)
    ds.close()
    print(f"Saved monthly climatology to {filename}")


# ---------------------------------------------------------------------------
# Config / main entry point
# ---------------------------------------------------------------------------


def load_config(config_file: str) -> Dict[str, str]:
    config: Dict[str, str] = {}
    try:
        with open(config_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Config file {config_file} not found; using defaults")
    return config


def bool_from_config(config: Dict[str, str], key: str, default: bool = False) -> bool:
    if key not in config:
        return default
    return config[key].lower() in {"true", "1", "yes"}


def int_from_config(config: Dict[str, str], key: str, default: int) -> int:
    return int(config.get(key, default))


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial CIMIS ASCII processing")
    parser.add_argument("--config", type=str, default="spatial_cimis_config.txt", help="Config file")
    args = parser.parse_args()

    config = load_config(args.config)

    variable = config.get("variable", "Tx")
    data_path = config.get("data_path", "/group/moniergrp/SpatialCIMIS/ascii/")
    netcdf_path = config.get("netcdf_path", "/group/moniergrp/SpatialCIMIS/netcdf/")
    output_path = config.get("output_path", "/home/salba/SpatialCIMIS/data/")
    start_year = int_from_config(config, "start_year", 2004)
    end_year = int_from_config(config, "end_year", 2023)
    target_mode = config.get("target_mode", "native").lower()
    shapefile_setting = config.get("shapefile_path", "CA_State.shp")
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if shapefile_setting:
        shapefile_path = (
            shapefile_setting
            if os.path.isabs(shapefile_setting)
            else os.path.abspath(os.path.join(config_dir, shapefile_setting))
        )
    else:
        shapefile_path = ""

    process_data_flag = bool_from_config(config, "process_data", True)
    process_2d = bool_from_config(config, "process_2d", False)
    process_daily = bool_from_config(config, "process_daily_clim", False)
    process_monthly = bool_from_config(config, "process_monthly_clim", False)
    process_all = bool_from_config(config, "process_all", False)

    if process_all:
        process_data_flag = process_2d = process_daily = process_monthly = True

    print("=" * 70)
    print("Spatial CIMIS ASCII Processing")
    print("=" * 70)
    print(f"Variable: {variable}")
    print(f"Data path: {data_path}")
    print(f"NetCDF path: {netcdf_path}")
    print(f"Target mode: {target_mode}")
    print(f"Years: {start_year}-{end_year}")
    print(f"State shapefile: {shapefile_path or 'None'}")
    print()

    target_grid = resolve_target_grid(target_mode, data_path, variable, start_year, end_year)
    print(f"Target grid: {target_grid.shape[1]}x{target_grid.shape[0]} ({target_grid.crs})")
    state_mask = build_state_mask(shapefile_path, target_grid) if shapefile_path else None
    if state_mask is not None:
        print(f"State mask built: {state_mask.sum()} cells inside California")
    else:
        print("State mask: None (no shapefile or mask build failed)")

    if process_data_flag:
        process_variable_data(
            data_path,
            netcdf_path,
            variable,
            target_grid,
            start_year,
            end_year,
            state_mask=state_mask,
        )

    if process_2d:
        create_2d_files(
            netcdf_path,
            output_path,
            variable,
            target_grid,
            start_year,
            end_year,
        )

    if process_daily:
        create_daily_climatology(
            netcdf_path,
            output_path,
            variable,
            target_grid,
        )

    if process_monthly:
        create_monthly_climatology(
            netcdf_path,
            output_path,
            variable,
            target_grid,
        )


if __name__ == "__main__":
    main()

