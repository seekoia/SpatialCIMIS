#!/usr/bin/env python3
"""
Compare NetCDF footprints between Spatial CIMIS and GridMET.

Creates an overlay plot showing spatial coverage of both datasets for a specific day.

Usage:
    python compare_netcdf_footprints.py <variable> <year> <day_of_year>
    
Examples:
    python compare_netcdf_footprints.py Tx 2010 150
    python compare_netcdf_footprints.py Rs 2015 200
    python compare_netcdf_footprints.py ETo 2020 100
"""

import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import rioxarray
from datetime import datetime, timedelta
import cartopy.crs as ccrs


# File path templates
SPATIAL_CIMIS_PATH = "/group/moniergrp/SpatialCIMIS/netcdf/"
GRIDMET_PATH = "/group/moniergrp/gridMET/"

# Variable mapping: Spatial CIMIS -> GridMET variable name
VARIABLE_MAP = {
    'Tx': 'air_temperature',
    'Tn': 'air_temperature',
    'ETo': 'potential_evapotranspiration',
    'Rs': 'surface_downwelling_shortwave_flux_in_air',
    'Tdew': 'mean_vapor_pressure_deficit',
    'U2': 'wind_speed'
}


def get_spatial_cimis_file(variable, year):
    """Get Spatial CIMIS NetCDF file path."""
    filename = f"spatial_cimis_{variable.lower()}_{year}.nc"
    filepath = Path(SPATIAL_CIMIS_PATH) / filename
    return filepath


def get_gridmet_file(variable, year):
    """Get GridMET NetCDF file path."""
    # GridMET file names use short codes
    gridmet_file_map = {
        'Tx': 'tmmx',
        'Tn': 'tmmn',
        'ETo': 'eto',
        'Rs': 'srad',
        'Tdew': 'vpd',
        'U2': 'vs'
    }
    
    gridmet_file = gridmet_file_map.get(variable)
    if gridmet_file is None:
        raise ValueError(f"Unknown variable: {variable}. Valid: {list(gridmet_file_map.keys())}")
    
    filename = f"{gridmet_file}_{year}.nc"
    filepath = Path(GRIDMET_PATH) / filename
    return filepath


def load_spatial_cimis_day(filepath, variable, day_of_year):
    """Load Spatial CIMIS data for a specific day."""
    print(f"\nLoading Spatial CIMIS data...")
    print(f"  File: {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Spatial CIMIS file not found: {filepath}")
    
    ds = xr.open_dataset(filepath)
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.dims)}")
    
    # Find the variable (handle case variations)
    var_name = None
    for v in ds.data_vars:
        if v.lower() == variable.lower():
            var_name = v
            break
    
    if var_name is None:
        raise KeyError(f"Variable '{variable}' not found. Available: {list(ds.data_vars)}")
    
    # Select the specific day
    if 'time' in ds.dims:
        data = ds[var_name].isel(time=day_of_year - 1)  # day_of_year is 1-based
    else:
        raise ValueError("No 'time' dimension found in Spatial CIMIS file")
    
    print(f"  Selected day {day_of_year}: shape {data.shape}")
    print(f"  Value range: {float(data.min()):.4f} to {float(data.max()):.4f}")
    
    # Set CRS - Spatial CIMIS is in EPSG:3310 (California Albers)
    print(f"  Setting CRS: EPSG:3310 (California Albers)")
    data.rio.write_crs(3310, inplace=True)
    if 'x' in data.dims and 'y' in data.dims:
        data.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    
    ds.close()
    return data


def load_gridmet_day(filepath, variable, day_of_year):
    """Load GridMET data for a specific day."""
    print(f"\nLoading GridMET data...")
    print(f"  File: {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"GridMET file not found: {filepath}")
    
    ds = xr.open_dataset(filepath)
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.dims)}")
    
    gridmet_var = VARIABLE_MAP.get(variable)
    if gridmet_var not in ds.data_vars:
        raise KeyError(f"GridMET variable '{gridmet_var}' not found. Available: {list(ds.data_vars)}")
    
    # Select the specific day
    if 'day' in ds.dims:
        data = ds[gridmet_var].isel(day=day_of_year - 1)  # day_of_year is 1-based
    else:
        raise ValueError("No 'day' dimension found in GridMET file")
    
    print(f"  Selected day {day_of_year}: shape {data.shape}")
    print(f"  Value range: {float(data.min()):.4f} to {float(data.max()):.4f}")
    
    # Convert units if needed
    if variable == 'ETo':
        print("  Converting GridMET ETo from mm to mm/day (already daily)")
        # GridMET ETo is already in mm/day, no conversion needed
    elif variable == 'Tx' or variable == 'Tn':
        print("  Converting GridMET temperature from K to °C")
        data = data - 273.15
        print(f"  New range: {float(data.min()):.4f} to {float(data.max()):.4f} °C")
    elif variable == 'Rs':
        print("  GridMET Rs is in W/m², no conversion needed")
    
    # Set CRS - GridMET is in EPSG:4326 (WGS84)
    print(f"  Setting CRS: EPSG:4326 (WGS84)")
    data.rio.write_crs('EPSG:4326', inplace=True)
    if 'lon' in data.dims and 'lat' in data.dims:
        data.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    elif 'x' in data.dims and 'y' in data.dims:
        data.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    
    ds.close()
    return data


def normalize_data(data):
    """Normalize data to 0-1 range."""
    data_min = float(data.min())
    data_max = float(data.max())
    
    if data_max == data_min:
        return data * 0  # All zeros if constant
    
    normalized = (data - data_min) / (data_max - data_min)
    return normalized


def plot_comparison(spatial_data, gridmet_data, variable, year, day_of_year, output_file):
    """Create overlay comparison plot using cartopy."""
    print(f"\n{'='*70}")
    print("CREATING COMPARISON PLOT")
    print(f"{'='*70}")
    
    # Get date string
    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    date_str = date.strftime('%Y-%m-%d')
    
    # Get original data values for statistics
    spatial_values = spatial_data.values
    gridmet_values = gridmet_data.values
    
    print(f"\nOriginal data ranges:")
    print(f"  Spatial CIMIS: {np.nanmin(spatial_values):.4f} to {np.nanmax(spatial_values):.4f}")
    print(f"  GridMET: {np.nanmin(gridmet_values):.4f} to {np.nanmax(gridmet_values):.4f}")
    
    # Check for valid data
    spatial_valid = np.sum(~np.isnan(spatial_values))
    gridmet_valid = np.sum(~np.isnan(gridmet_values))
    
    print(f"\nValid pixels:")
    print(f"  Spatial CIMIS: {spatial_valid:,} ({100*spatial_valid/spatial_values.size:.1f}%)")
    print(f"  GridMET: {gridmet_valid:,} ({100*gridmet_valid/gridmet_values.size:.1f}%)")
    
    # Normalize data
    print(f"\nNormalizing each dataset to 0-1...")
    spatial_norm = normalize_data(spatial_data)
    gridmet_norm = normalize_data(gridmet_data)
    
    print(f"  Spatial CIMIS normalized: {float(spatial_norm.min()):.4f} to {float(spatial_norm.max()):.4f}")
    print(f"  GridMET normalized: {float(gridmet_norm.min()):.4f} to {float(gridmet_norm.max()):.4f}")
    
    # Define projections
    # Spatial CIMIS uses California Albers (EPSG:3310)
    # Standard parameters for NAD83 / California Albers
    crs_spatial = ccrs.AlbersEqualArea(
        central_longitude=-120,
        central_latitude=0,
        standard_parallels=(34, 40.5),
        false_easting=0,
        false_northing=-4000000
    )
    
    # GridMET uses WGS84 (EPSG:4326)
    crs_gridmet = ccrs.PlateCarree()
    
    # Map projection for plotting (use PlateCarree for simplicity)
    map_proj = ccrs.PlateCarree()
    
    # Create figure with cartopy projection
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=map_proj)
    
    # Plot GridMET first (Blues) with its native projection
    print("\nPlotting GridMET data...")
    im_gridmet = gridmet_norm.plot.pcolormesh(
        ax=ax,
        x='lon',
        y='lat',
        transform=crs_gridmet,
        cmap='Blues',
        alpha=0.5,
        vmin=0,
        vmax=1,
        add_colorbar=False,
        add_labels=False
    )
    
    # Plot Spatial CIMIS (Reds) with its native projection
    print("Plotting Spatial CIMIS data...")
    im_spatial = spatial_norm.plot.pcolormesh(
        ax=ax,
        x='x',
        y='y',
        transform=crs_spatial,
        cmap='Reds',
        alpha=0.5,
        vmin=0,
        vmax=1,
        add_colorbar=False,
        add_labels=False
    )
    
    # Add map features
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    
    # Set extent to California (in lon/lat)
    ax.set_extent([-125, -114, 32, 42.5], crs=ccrs.PlateCarree())
    
    # Title
    title = f'{variable} Comparison: {date_str} (Day {day_of_year})\n'
    title += 'Blue=GridMET, Red=Spatial CIMIS (Both Normalized 0-1)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im_spatial, ax=ax, label='Normalized Value (0-1)', 
                       shrink=0.7, orientation='horizontal', pad=0.05)
    
    # Legend
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, label=f'GridMET ({gridmet_valid:,} pixels)'),
        Patch(facecolor='red', alpha=0.5, label=f'Spatial CIMIS ({spatial_valid:,} pixels)'),
        Patch(facecolor='purple', alpha=0.7, label='Overlap')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             framealpha=0.9)
    
    # Add text box with info
    info_text = f'Variable: {variable}\n'
    info_text += f'Year: {year}\n'
    info_text += f'Day: {day_of_year}\n'
    info_text += f'Date: {date_str}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"\nSpatial CIMIS:")
    print(f"  Total pixels: {spatial_values.size:,}")
    print(f"  Valid pixels: {spatial_valid:,} ({100*spatial_valid/spatial_values.size:.1f}%)")
    print(f"  NaN pixels: {np.sum(np.isnan(spatial_values)):,}")
    print(f"  Value range: {np.nanmin(spatial_values):.4f} to {np.nanmax(spatial_values):.4f}")
    print(f"  Mean: {np.nanmean(spatial_values):.4f}")
    
    print(f"\nGridMET:")
    print(f"  Total pixels: {gridmet_values.size:,}")
    print(f"  Valid pixels: {gridmet_valid:,} ({100*gridmet_valid/gridmet_values.size:.1f}%)")
    print(f"  NaN pixels: {np.sum(np.isnan(gridmet_values)):,}")
    print(f"  Value range: {np.nanmin(gridmet_values):.4f} to {np.nanmax(gridmet_values):.4f}")
    print(f"  Mean: {np.nanmean(gridmet_values):.4f}")


def main():
    """Main function."""
    if len(sys.argv) != 4:
        print("Usage: python compare_netcdf_footprints.py <variable> <year> <day_of_year>")
        print("\nExamples:")
        print("  python compare_netcdf_footprints.py Tx 2010 150")
        print("  python compare_netcdf_footprints.py Rs 2015 200")
        print("  python compare_netcdf_footprints.py ETo 2020 100")
        print("\nValid variables:", list(VARIABLE_MAP.keys()))
        sys.exit(1)
    
    variable = sys.argv[1]
    year = int(sys.argv[2])
    day_of_year = int(sys.argv[3])
    
    # Validate inputs
    if variable not in VARIABLE_MAP:
        print(f"Error: Invalid variable '{variable}'")
        print(f"Valid variables: {list(VARIABLE_MAP.keys())}")
        sys.exit(1)
    
    if day_of_year < 1 or day_of_year > 366:
        print(f"Error: day_of_year must be between 1 and 366")
        sys.exit(1)
    
    print("="*70)
    print("NETCDF FOOTPRINT COMPARISON")
    print("="*70)
    print(f"\nVariable: {variable}")
    print(f"Year: {year}")
    print(f"Day of year: {day_of_year}")
    
    # Get date string for output filename
    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    date_str = date.strftime('%Y%m%d')
    
    try:
        # Get file paths
        spatial_file = get_spatial_cimis_file(variable, year)
        gridmet_file = get_gridmet_file(variable, year)
        
        # Load data
        spatial_data = load_spatial_cimis_day(spatial_file, variable, day_of_year)
        gridmet_data = load_gridmet_day(gridmet_file, variable, day_of_year)
        
        # Create output filename
        output_file = f'netcdf_comparison_{variable.lower()}_{date_str}.png'
        
        # Create plot
        plot_comparison(spatial_data, gridmet_data, variable, year, day_of_year, output_file)
        
        print("\n" + "="*70)
        print("COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

