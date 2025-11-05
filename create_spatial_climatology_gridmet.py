#!/usr/bin/env python3
"""
Script to create Spatial CIMIS climatology from GridMET-projected data.

This creates the missing spatial_mean_eto_gridmet.nc file needed for plotting.
"""

import numpy as np
import xarray as xr
import os
from pathlib import Path

def create_spatial_climatology_gridmet():
    """Create Spatial CIMIS climatology from GridMET-projected data."""
    
    # File paths
    data_path = "/group/moniergrp/SpatialCIMIS/netcdf/test/"
    output_path = "/home/salba/SpatialCIMIS/output/test/"
    variable = "ETo"
    
    print("Creating Spatial CIMIS climatology (GridMET grid)...")
    
    # Load all GridMET-projected Spatial CIMIS files
    pattern = data_path + f"spatial_cimis_{variable.lower()}_20*_gridmet.nc"
    print(f"Loading files: {pattern}")
    
    ds = xr.open_mfdataset(pattern, combine='nested', concat_dim="time")
    var_data = ds[variable]
    
    print(f"Loaded {len(ds.time)} time steps")
    print(f"Grid shape: {var_data.shape}")
    
    # Compute climatology (mean over time)
    clim = var_data.mean(dim='time')
    
    print(f"Climatology shape: {clim.shape}")
    print(f"Value range: {float(clim.min()):.2f} to {float(clim.max()):.2f}")
    
    # Set CRS
    clim.rio.write_crs("EPSG:4326", inplace=True)
    clim.rio.write_nodata(np.nan, inplace=True)
    
    # Convert to dataset for saving
    clim_ds = clim.to_dataset(name=clim.name if clim.name else variable)
    
    # Add CRS as attribute
    clim_ds.attrs['crs'] = 'EPSG:4326'
    clim_ds.attrs['grid_mapping'] = 'crs'
    clim_ds.attrs['title'] = f'Spatial CIMIS {variable} Climatology (GridMET Grid)'
    clim_ds.attrs['description'] = f'Mean {variable} computed from GridMET-projected Spatial CIMIS data'
    
    # Save
    output_file = os.path.join(output_path, f'spatial_mean_{variable.lower()}_gridmet.nc')
    clim_ds.to_netcdf(output_file)
    
    print(f"Climatology saved to: {output_file}")
    
    # Clean up
    ds.close()
    
    return output_file

if __name__ == "__main__":
    create_spatial_climatology_gridmet()



