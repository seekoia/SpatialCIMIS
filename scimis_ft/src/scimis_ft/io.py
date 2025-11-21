from __future__ import annotations

import os
from glob import glob
from typing import Iterable, Optional, Sequence

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401


def _filter_years(file_paths: Sequence[str], start_year: Optional[int], end_year: Optional[int]) -> list[str]:
    if start_year is None and end_year is None:
        return list(file_paths)

    selected: list[str] = []
    for path in file_paths:
        basename = os.path.basename(path)
        digits = "".join(ch for ch in basename if ch.isdigit())
        year = None
        for window in range(len(digits) - 3):
            candidate = digits[window : window + 4]
            if candidate.isdigit():
                year = int(candidate)
                break
        if year is None:
            continue
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        selected.append(path)
    return selected


def load_spatial_cimis_variable(
    base_path: str,
    file_identifier: str,
    netcdf_variable_name: str,
    *,
    crs_epsg: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    preprocess: Optional[callable] = None,
    engine: str = "netcdf4",
    chunks: Optional[dict[str, int]] = None,
    file_pattern: Optional[str] = None,
) -> xr.DataArray:
    """
    Load Spatial CIMIS NetCDF data for the specified variable across years.

    Parameters
    ----------
    base_path:
        Directory containing NetCDF files.
    file_identifier:
        Portion of the filename after ``spatial_cimis_`` (e.g., ``'eto'`` or ``'Rs'``).
    netcdf_variable_name:
        Name of the variable stored within the NetCDF files (case sensitive).
    crs_epsg:
        EPSG code to assign to the resulting DataArray. Defaults to 3310.
    start_year, end_year:
        Optional inclusive bounds for filtering file years.
    preprocess:
        Optional callable executed by ``open_mfdataset``.
    engine:
        NetCDF backend engine (default ``'netcdf4'``).
    chunks:
        Optional chunk sizes for dask-backed arrays.

    Returns
    -------
    xarray.DataArray
        Combined data array across all selected files with CRS metadata.
    """
    if file_pattern:
        pattern = os.path.join(base_path, file_pattern)
    else:
        pattern = os.path.join(base_path, f"spatial_cimis_{file_identifier}_20*.nc")
    files = sorted(glob(pattern))
    files = _filter_years(files, start_year, end_year)

    if not files:
        raise FileNotFoundError(f"No files matched pattern {pattern} within selected years.")

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        engine=engine,
        preprocess=preprocess,
        chunks=chunks,
    )

    if netcdf_variable_name not in ds:
        raise KeyError(f"Variable '{netcdf_variable_name}' not present in dataset. Available: {list(ds.data_vars)}")

    data_array = ds[netcdf_variable_name].copy(deep=False)

    # Normalize spatial dimension names for rioxarray compatibility
    dim_renames = {}
    if "lat" in data_array.dims and "lon" in data_array.dims:
        dim_renames.update({"lat": "y", "lon": "x"})
    elif "latitude" in data_array.dims and "longitude" in data_array.dims:
        dim_renames.update({"latitude": "y", "longitude": "x"})

    if dim_renames:
        data_array = data_array.rename(dim_renames)

    # Ensure spatial dims are ordered (y, x) for raster operations
    if ("y" in data_array.dims) and ("x" in data_array.dims):
        data_array = data_array.transpose(..., "y", "x")

    # Attach latitude/longitude coordinates (if available) using renamed dims
    if "y" in data_array.dims:
        if "lat" in ds and set(ds["lat"].dims) == {"y"}:
            data_array = data_array.assign_coords(lat=ds["lat"].values)
    if "x" in data_array.dims:
        if "lon" in ds and set(ds["lon"].dims) == {"x"}:
            data_array = data_array.assign_coords(lon=ds["lon"].values)

    # Determine CRS
    inferred_epsg = _infer_epsg(data_array, explicit_epsg=crs_epsg)

    data_array.rio.write_crs(f"EPSG:{inferred_epsg}", inplace=True)
    if {"x", "y"}.issubset(set(data_array.dims)):
        data_array.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    data_array.rio.write_nodata(np.nan, inplace=True)
    ds.close()
    return data_array


def _infer_epsg(data_array: xr.DataArray, explicit_epsg: Optional[int]) -> int:
    if explicit_epsg:
        return explicit_epsg

    if "x" in data_array.coords and "y" in data_array.coords:
        x_vals = data_array.coords["x"].values
        y_vals = data_array.coords["y"].values
        if (
            np.nanmin(x_vals) >= -180
            and np.nanmax(x_vals) <= 180
            and np.nanmin(y_vals) >= -90
            and np.nanmax(y_vals) <= 90
        ):
            return 4326
    return 3310

