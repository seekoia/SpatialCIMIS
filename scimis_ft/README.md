scimis-ft
=========

`scimis-ft` provides reusable utilities for deriving weekly climatologies from Spatial CIMIS NetCDF collections, performing Fourier decomposition, and visualising leading frequency components.

## Features

- Discover and load yearly Spatial CIMIS NetCDF files for a chosen variable.
- Compute ISO week climatologies while preserving spatial metadata and CRS.
- Extract leading Fourier amplitudes and phases with physically scaled magnitudes.
- Reconstruct climatologies from a limited number of Fourier modes.
- Plot Fourier component amplitudes/phases, per-pixel RMSE, and pixel time series comparisons.
- Optionally clip analyses to a shapefile boundary (e.g., California).
- Support custom filename patterns (e.g., `spatial_cimis_eto_*_4km.nc`).
- Command-line interface driven by a plain-text configuration file.

## Installation

```
pip install .
```

## Usage

```
scimis-ft --config /path/to/config.txt
```

Example configuration:

```
base_path=/group/moniergrp/SpatialCIMIS/netcdf
file_identifier=Rs
netcdf_variable=Rs
start_year=2010
end_year=2020
components=3
output_dir=/group/moniergrp/SpatialCIMIS/output
clip_path=/path/to/california_boundary.shp
file_pattern=spatial_cimis_eto_*_4km.nc
crs_epsg=4326
```

The CLI will generate weekly climatology NetCDF files, Fourier component datasets, reconstructed climatology, RMSE rasters, and plotting outputs (component panels, RMSE map, time-series comparison) inside the configured output directory.

### Generated Plot Files

- `<var>_fourier_components.png` – Component 1 amplitude (spanning both rows) plus amplitudes and phases for components 2 and 3.
- `<var>_climatology_timeseries.png` – Original vs reconstructed weekly climatology for a representative pixel.
- `<var>_rmse.png` – Spatial RMSE map of the reconstruction error.
- `<var>_fourier_histograms.png` – Histograms summarising amplitudes and peak-day timing for leading Fourier components.

