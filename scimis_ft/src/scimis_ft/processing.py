from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr
import geopandas as gpd

from .climatology import calculate_weekly_climatology
from .config import PipelineConfig
from .fourier import (
    fourier_transform_climatology_components,
    reconstruct_climatology_from_fourier_components,
)
from .io import load_spatial_cimis_variable
from .plotting import (
    plot_component_histograms,
    plot_climatology_timeseries,
    plot_fourier_components,
    plot_rmse_map,
)


def run_pipeline(
    config: PipelineConfig,
    *,
    save_results: bool = True,
    compute_metrics: bool = True,
    make_plots: bool = True,
    show_plots: bool = False,
) -> Dict[str, xr.DataArray | xr.Dataset | Path | None]:
    """
    Execute the full processing workflow for a Spatial CIMIS variable.
    """
    data_array = load_spatial_cimis_variable(
        str(config.base_path),
        config.file_identifier,
        config.netcdf_variable,
        start_year=config.start_year,
        end_year=config.end_year,
        file_pattern=config.file_pattern,
        crs_epsg=config.crs_epsg,
    )
    data_shape = data_array.shape
    data_dtype = str(data_array.dtype)
    data_dims = tuple(data_array.dims)

    weekly = calculate_weekly_climatology(data_array)

    # Release daily data once the weekly aggregate is available to reduce memory footprint
    del data_array
    gc.collect()

    if config.clip_path is not None:
        weekly = _clip_to_geometry(weekly, config.clip_path)
    components = fourier_transform_climatology_components(weekly, config.components)
    reconstructed = reconstruct_climatology_from_fourier_components(
        components,
        weekly.sizes["week"],
        original_time_dim_name="week",
        original_week_coordinates=weekly["week"],
        reference_attributes=weekly.attrs,
        reference_name=weekly.name or config.netcdf_variable,
    )

    rmse = None
    if compute_metrics:
        rmse = _compute_rmse(weekly, reconstructed)

    component_plot = None
    timeseries_plot = None
    rmse_plot_path = None
    histogram_plot = None
    if make_plots:
        output_dir = config.output_dir or config.base_path
        base_path = Path(output_dir)
        base_name = config.file_identifier.lower()

        component_plot = plot_fourier_components(
            components,
            output_path=base_path / f"{base_name}_fourier_components.png",
            show=show_plots,
        )

        timeseries_plot = plot_climatology_timeseries(
            weekly,
            reconstructed,
            output_path=base_path / f"{base_name}_climatology_timeseries.png",
            show=show_plots,
        )

        if rmse is not None:
            rmse_plot_path = plot_rmse_map(
                rmse,
                output_path=base_path / f"{base_name}_rmse.png",
                show=show_plots,
            )

        histogram_plot = plot_component_histograms(
            components,
            output_path=base_path / f"{base_name}_fourier_histograms.png",
            show=show_plots,
        )

    if save_results:
        _persist_outputs(config, weekly, components, reconstructed, rmse)

    return {
        "data_shape": data_shape,
        "data_dtype": data_dtype,
        "data_dims": data_dims,
        "weekly_climatology": weekly,
        "fourier_components": components,
        "reconstructed": reconstructed,
        "rmse": rmse,
        "plot_path": component_plot,
        "component_plot": component_plot,
        "timeseries_plot": timeseries_plot,
        "rmse_plot": rmse_plot_path,
        "histogram_plot": histogram_plot,
    }


def _compute_rmse(original: xr.DataArray, reconstructed: xr.DataArray) -> xr.DataArray:
    error = original - reconstructed
    mse = (error ** 2).mean(dim="week")
    rmse = np.sqrt(mse)
    rmse.attrs.update(original.attrs)
    rmse.attrs["long_name"] = "Root Mean Squared Error between original and reconstructed climatology"
    return rmse


def _persist_outputs(
    config: PipelineConfig,
    weekly: xr.DataArray,
    components: xr.Dataset,
    reconstructed: xr.DataArray,
    rmse: Optional[xr.DataArray],
) -> None:
    output_dir = config.output_dir or config.base_path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = config.file_identifier.lower()
    weekly.to_netcdf(output_dir / f"{base}_weekly_climatology.nc")
    components.to_netcdf(output_dir / f"{base}_weekly_fourier_components.nc")
    reconstructed.to_netcdf(output_dir / f"{base}_reconstructed_weekly_climatology.nc")
    if rmse is not None:
        rmse.to_netcdf(output_dir / f"{base}_reconstruction_rmse.nc")


def _clip_to_geometry(data: xr.DataArray, clip_path: Path) -> xr.DataArray:
    geometry = gpd.read_file(clip_path)
    if geometry.empty:
        raise ValueError(f"Clip shapefile '{clip_path}' contains no geometries.")

    if data.rio.crs is None:
        raise ValueError("Input data must have a CRS before clipping.")

    if geometry.crs is None:
        raise ValueError(f"Clip shapefile '{clip_path}' lacks CRS information.")

    if geometry.crs != data.rio.crs:
        data = data.rio.reproject(geometry.crs)

    clipped = data.rio.clip(geometry.geometry, drop=True, all_touched=False)
    return clipped

