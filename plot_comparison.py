#!/usr/bin/env python3
"""
Combined plotting script for Spatial CIMIS analysis.

Creates three types of plots:
1. Spatial comparison: 3-panel maps (Spatial CIMIS, GridMET, Bias)
2. Station time series: Multi-panel plots (monthly climatology + yearly time series)
3. Station validation: 2x3 panel maps showing correlation, bias, and MAE for Spatial CIMIS and GridMET

Configuration file controls which plots to create.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import geopandas as gpd
import seaborn as sns
import rioxarray
import gc
import sys
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_absolute_error
import warnings
from shapely.errors import ShapelyDeprecationWarning

try:
    import skill_metrics as sm
    from skill_metrics import centered_rms_dev, error_check_stats
    HAS_SKILL_METRICS = True
except ImportError:
    HAS_SKILL_METRICS = False
    print("Warning: skill_metrics not available. Taylor diagrams will be skipped.")

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# Set plotting style
plt.style.use('default')
sns.set_context("paper")

# Define default colors for time series (Spatial CIMIS, GridMET, Station)
DEFAULT_COLORS = ['#006ba4', '#ff800e', '#ababab']  # blue, orange, gray


def load_config(config_file):
    """Load configuration from text file."""
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert boolean strings
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    
                    config[key] = value
    return config


# ============================================================================
# SPATIAL COMPARISON PLOT FUNCTIONS
# ============================================================================

def resolve_with_suffix(base_path: Path, suffixes=("_2km", "_4km"), config_suffix=None) -> Path:
    """Return existing file, trying optional suffixes if needed."""
    # First try with config suffix if provided
    if config_suffix:
        stem = base_path.stem
        suffix = base_path.suffix
        # Remove any existing suffix from stem before adding config suffix
        for existing_suffix in suffixes:
            if stem.endswith(existing_suffix):
                stem = stem[:-len(existing_suffix)]
                break
        candidate_with_config = base_path.with_name(f"{stem}{config_suffix}{suffix}")
        if candidate_with_config.exists():
            return candidate_with_config
    
    # Then try the base path
    if base_path.exists():
        return base_path

    # Finally try other suffixes
    stem = base_path.stem
    suffix = base_path.suffix
    # Remove any existing suffix from stem
    for existing_suffix in suffixes:
        if stem.endswith(existing_suffix):
            stem = stem[:-len(existing_suffix)]
            break

    for extra in suffixes:
        candidate = base_path.with_name(f"{stem}{extra}{suffix}")
        if candidate.exists():
            print(f"  Falling back to file: {candidate}")
            return candidate

    raise FileNotFoundError(f"Could not locate file for {base_path} with suffixes {suffixes}")


def create_spatial_comparison_plot(config):
    """Create 3-panel spatial comparison plot (Spatial CIMIS, GridMET, Bias)."""
    print("\n" + "="*70)
    print("SPATIAL COMPARISON PLOT")
    print("="*70)
    
    variable = config['variable']
    output_path = config['output_path']
    shapefile_path = config['shapefile_path']
    
    # Construct file paths
    suffix = config.get('spatial_file_suffix', '')
    spatial_file = Path(output_path) / config['spatial_mean_output'].replace('{variable}', variable.lower()).replace('{suffix}', suffix)
    gridmet_file = Path(output_path) / config['gridmet_mean_output'].replace('{variable}', variable.lower()).replace('{suffix}', suffix if suffix else '_4km')

    spatial_file = resolve_with_suffix(spatial_file, config_suffix=suffix)
    gridmet_suffix = suffix if suffix else '_4km'
    gridmet_file = resolve_with_suffix(gridmet_file, config_suffix=gridmet_suffix)
    
    print(f"\nVariable: {variable}")
    print(f"Spatial CIMIS file: {spatial_file}")
    print(f"GridMET file: {gridmet_file}")
    print(f"Shapefile: {shapefile_path}")
    
    # Load spatial data
    print("\n1. Loading Spatial CIMIS climatology...")
    spatial_ds = xr.open_dataset(spatial_file)
    
    # Find the variable name (handle case variations)
    spatial_var_name = None
    for var in spatial_ds.data_vars:
        if var.lower() == variable.lower():
            spatial_var_name = var
            break
    
    if spatial_var_name is None:
        raise KeyError(f"Variable '{variable}' not found in Spatial CIMIS dataset. Available: {list(spatial_ds.data_vars)}")
    
    # Check if data already has day_of_year dimension or is already a climatological mean
    if 'day_of_year' in spatial_ds[spatial_var_name].dims:
        print(f"  Computing mean over day_of_year dimension...")
        spatial_data = spatial_ds[spatial_var_name].mean(dim='day_of_year')
    else:
        print(f"  Using pre-computed climatological mean (no day_of_year dimension)")
        spatial_data = spatial_ds[spatial_var_name]
    
    print(f"  Shape: {spatial_data.shape}")
    print(f"  Value range: {float(spatial_data.min()):.2f} to {float(spatial_data.max()):.2f}")
    
    # Note: Rs unit conversion (MJ/m²/day to W/m²) is now handled during extraction in spatial_cimis_unified.py
    
    # Load GridMET data
    print("\n2. Loading GridMET climatology...")
    gridmet_ds = xr.open_dataset(gridmet_file)
    
    # Find the variable name (handle GridMET naming conventions)
    gridmet_var_name = None
    
    # First try exact match
    for var in gridmet_ds.data_vars:
        if var.lower() == variable.lower():
            gridmet_var_name = var
            break
    
    # If not found, try common GridMET variable names
    if gridmet_var_name is None and variable == 'Rs':
        if 'surface_downwelling_shortwave_flux_in_air' in gridmet_ds.data_vars:
            gridmet_var_name = 'surface_downwelling_shortwave_flux_in_air'
        elif 'srad' in gridmet_ds.data_vars:
            gridmet_var_name = 'srad'
    elif gridmet_var_name is None and variable == 'ETo':
        if 'potential_evapotranspiration' in gridmet_ds.data_vars:
            gridmet_var_name = 'potential_evapotranspiration'
        elif 'pet' in gridmet_ds.data_vars:
            gridmet_var_name = 'pet'
    elif gridmet_var_name is None and variable in ['Tx', 'Tn']:
        if 'air_temperature' in gridmet_ds.data_vars:
            gridmet_var_name = 'air_temperature'
        elif 'temperature' in gridmet_ds.data_vars:
            gridmet_var_name = 'temperature'
    
    if gridmet_var_name is None:
        raise KeyError(f"Variable '{variable}' not found in GridMET dataset. Available: {list(gridmet_ds.data_vars)}")
    
    print(f"  Using GridMET variable: '{gridmet_var_name}'")
    
    # Check if this is already a climatological mean (no day_of_year dimension)
    if 'day_of_year' in gridmet_ds[gridmet_var_name].dims:
        gridmet_data = gridmet_ds[gridmet_var_name].mean(dim='day_of_year')
        print(f"  Computing climatological mean from daily data...")
    else:
        gridmet_data = gridmet_ds[gridmet_var_name]
        print(f"  Using pre-computed climatological mean (no day_of_year dimension)")
    
    print(f"  Shape: {gridmet_data.shape}")
    print(f"  Value range: {float(gridmet_data.min()):.2f} to {float(gridmet_data.max()):.2f}")
    
    # Convert GridMET temperature from Kelvin to Celsius if needed
    if variable in ['Tx', 'Tn'] and float(gridmet_data.min()) > 200:
        print(f"  Converting GridMET temperature from Kelvin to Celsius (-273.15)...")
        gridmet_data = gridmet_data - 273.15
        print(f"  New value range: {float(gridmet_data.min()):.2f} to {float(gridmet_data.max()):.2f} °C")
    
    # Load California shapefile
    print("\n3. Loading California shapefile...")
    ca_shape = gpd.read_file(shapefile_path)
    if ca_shape.crs is None:
        ca_shape.crs = 'EPSG:4326'
    print(f"  CRS: {ca_shape.crs}")
    
    # Set CRS for spatial data
    print("\n4. Setting CRS for spatial data...")
    if 'spatial_ref' in spatial_ds or 'crs' in spatial_ds:
        # Try to read CRS from dataset
        try:
            spatial_data.rio.write_crs('EPSG:4326', inplace=True)
        except:
            pass
    
    # Set spatial dimensions
    if 'x' in spatial_data.dims and 'y' in spatial_data.dims:
        spatial_data.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    
    # Spatial CIMIS data is in EPSG:3310 (California Albers)
    spatial_data.rio.write_crs('EPSG:3310', inplace=True)
    
    # Set CRS for GridMET data
    print("\n5. Setting CRS for GridMET data...")
    if 'lon' in gridmet_data.dims and 'lat' in gridmet_data.dims:
        gridmet_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    elif 'x' in gridmet_data.dims and 'y' in gridmet_data.dims:
        gridmet_data.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    
    gridmet_data.rio.write_crs('EPSG:4326', inplace=True)
    
    # Reproject Spatial CIMIS to match shapefile CRS before clipping
    print("\n6. Reprojecting Spatial CIMIS to match shapefile CRS...")
    spatial_data_reproj = spatial_data.rio.reproject('EPSG:4326')
    print(f"    Reprojected shape: {spatial_data_reproj.shape}")
    
    # Clip both datasets to California
    print("\n7. Clipping to California boundary...")
    print("  Clipping Spatial CIMIS...")
    # Extract individual geometries from the shapefile
    ca_geometries = ca_shape.geometry.values
    spatial_clipped = spatial_data_reproj.rio.clip(ca_geometries, ca_shape.crs, drop=True)
    print(f"    Clipped shape: {spatial_clipped.shape}")
    
    print("  Clipping GridMET...")
    gridmet_clipped = gridmet_data.rio.clip(ca_geometries, ca_shape.crs, drop=True)
    print(f"    Clipped shape: {gridmet_clipped.shape}")
    
    # Reproject GridMET to match Spatial CIMIS grid
    print("\n  Reprojecting GridMET to Spatial CIMIS grid...")
    gridmet_reprojected = gridmet_clipped.rio.reproject_match(spatial_clipped)
    print(f"    Reprojected shape: {gridmet_reprojected.shape}")
    
    # Convert to NumPy arrays for memory efficiency
    print("\n  Converting to NumPy arrays...")
    spatial_values = spatial_clipped.values.copy()
    gridmet_values = gridmet_reprojected.values.copy()

    # Determine coordinate arrays
    if 'x' in spatial_clipped.coords:
        x_vals = spatial_clipped['x'].values
    elif 'lon' in spatial_clipped.coords:
        x_vals = spatial_clipped['lon'].values
    else:
        x_vals = np.arange(spatial_values.shape[1])

    if 'y' in spatial_clipped.coords:
        y_vals = spatial_clipped['y'].values
    elif 'lat' in spatial_clipped.coords:
        y_vals = spatial_clipped['lat'].values
    else:
        y_vals = np.arange(spatial_values.shape[0])

    # Ensure coordinates are ascending for plotting
    if x_vals[0] > x_vals[-1]:
        x_vals = x_vals[::-1].copy()
        spatial_values = spatial_values[:, ::-1]
        gridmet_values = gridmet_values[:, ::-1]

    if y_vals[0] > y_vals[-1]:
        y_vals = y_vals[::-1].copy()
        spatial_values = spatial_values[::-1, :]
        gridmet_values = gridmet_values[::-1, :]

    spatial_extent = [np.nanmin(x_vals), np.nanmax(x_vals), np.nanmin(y_vals), np.nanmax(y_vals)]
    spatial_origin = 'lower'
    
    # Clean up xarray objects
    del spatial_clipped, gridmet_clipped, gridmet_reprojected, spatial_data, gridmet_data
    gc.collect()
    
    # Compute bias
    print("\n  Computing bias (GridMET - Spatial CIMIS)...")
    bias_values = gridmet_values - spatial_values
    print(f"    Bias range: {np.nanmin(bias_values):.2f} to {np.nanmax(bias_values):.2f}")
    print(f"    Mean bias: {np.nanmean(bias_values):.2f}")
    
    # Create the 3-panel figure
    print("\n7. Creating comparison plot...")
    
    fig, axes = plt.subplots(ncols=3, figsize=(16, 5), constrained_layout=True)
    
    # Variable-specific parameters
    var_params = {
        'ETo': {'vmin': 0, 'vmax': 7, 'cmap': 'viridis', 'label': 'ETo (mm/day)'},
        'Tx': {'vmin': 0, 'vmax': 40, 'cmap': 'afmhot', 'label': 'Tx (°C)'},
        'Tn': {'vmin': -10, 'vmax': 25, 'cmap': 'afmhot', 'label': 'Tn (°C)'},
        'Rs': {'vmin': 150, 'vmax': 280, 'cmap': 'afmhot', 'label': 'Rs (W/m²)'}
    }
    
    params = var_params.get(variable, {'vmin': None, 'vmax': None, 'cmap': 'viridis', 'label': variable})
    
    # Panel 1: Spatial CIMIS
    if spatial_values is not None:
        im0 = axes[0].imshow(
            spatial_values,
            cmap=params['cmap'],
            vmin=params['vmin'],
            vmax=params['vmax'],
            origin=spatial_origin,
            aspect='auto',
            interpolation='nearest',
            extent=spatial_extent
        )
        plt.colorbar(im0, ax=axes[0], label=params['label'], shrink=0.8)
        axes[0].set_title('Spatial CIMIS', fontsize=12, fontweight='bold')
    
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    
    # Panel 2: GridMET
    if gridmet_values is not None:
        im1 = axes[1].imshow(
            gridmet_values,
            cmap=params['cmap'],
            vmin=params['vmin'],
            vmax=params['vmax'],
            origin=spatial_origin,
            aspect='auto',
            interpolation='nearest',
            extent=spatial_extent
        )
        plt.colorbar(im1, ax=axes[1], label=params['label'], shrink=0.8)
        axes[1].set_title('GridMET', fontsize=12, fontweight='bold')
    
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    
    # Panel 3: Bias
    if bias_values is not None:
        # Center colormap at zero
        # Use fixed range for ETo, dynamic for other variables
        if variable == 'ETo':
            vmax_bias = 2.25
        else:
            vmax_bias = max(abs(np.nanmin(bias_values)), abs(np.nanmax(bias_values)))
        im2 = axes[2].imshow(
            bias_values,
            cmap='RdBu_r',
            vmin=-vmax_bias,
            vmax=vmax_bias,
            origin=spatial_origin,
            aspect='auto',
            interpolation='nearest',
            extent=spatial_extent
        )
        plt.colorbar(im2, ax=axes[2], label=f'Bias ({params["label"]})', shrink=0.8)
        axes[2].set_title('GridMET - Spatial CIMIS', fontsize=12, fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'Bias\nNot Available', 
                    transform=axes[2].transAxes, ha='center', va='center',
                    fontsize=14, fontweight='bold')
        axes[2].set_title('Bias', fontsize=12, fontweight='bold')
    
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].get_xaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)
    
    # Save figure
    output_file = Path(output_path) / f"{variable.lower()}_gridmet_spatial_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved to: {output_file}")
    
    plt.close()
    
    # Clean up
    spatial_ds.close()
    gridmet_ds.close()
    del spatial_values, gridmet_values, bias_values
    gc.collect()
    
    print("\nSpatial comparison plot complete!")


# ============================================================================
# STATION TIME SERIES PLOT FUNCTIONS
# ============================================================================

def load_timeseries_data(config):
    """Load station, spatial, and GridMET time series data."""
    print("\nLoading time series data...")
    
    output_path = config['output_path']
    variable = config['variable']
    suffix = config.get('spatial_file_suffix', '')
    
    # Construct file paths based on variable with suffix from config
    station_file = resolve_with_suffix(
        Path(output_path) / f"station_{variable.lower()}_data.csv",
        config_suffix=suffix
    )
    spatial_file = resolve_with_suffix(
        Path(output_path) / f"spatial_cimis_station_{variable.lower()}.csv",
        config_suffix=suffix
    )
    # GridMET uses suffix or defaults to _4km
    gridmet_suffix = suffix if suffix else '_4km'
    gridmet_file = resolve_with_suffix(
        Path(output_path) / f"gridmet_station_{variable.lower()}.csv",
        config_suffix=gridmet_suffix
    )
    
    # Load station observations
    print("  Loading station observations...")
    station_data = pd.read_csv(station_file, index_col='date', parse_dates=['date'])
    station_data.index = pd.to_datetime(station_data.index, errors='coerce')
    # Filter out invalid dates (NaT)
    station_data = station_data[station_data.index.notna()]
    station_data.columns = station_data.columns.astype(str)
    print(f"  Station data: {station_data.shape}")
    if len(station_data) > 0:
        print(f"    Date range: {station_data.index.min()} to {station_data.index.max()}")
    
    # Load Spatial CIMIS predictions
    print("  Loading Spatial CIMIS data...")
    # First read without parsing dates to check format
    spatial_data_raw = pd.read_csv(spatial_file, index_col='date')
    
    # Check if index is numeric (indicating days since epoch or similar)
    if len(spatial_data_raw) > 0:
        first_val = spatial_data_raw.index[0]
        if isinstance(first_val, (int, float)) or (isinstance(first_val, str) and first_val.lstrip('-').isdigit()):
            # Dates are numeric - need to reconstruct
            print(f"    Warning: Dates are numeric format (first value: {first_val})")
            print(f"    Reconstructing date index from row count and station data range...")
            
            # Use station data date range as reference if available
            if len(station_data) > 0:
                station_start = station_data.index.min()
                station_end = station_data.index.max()
                
                # If spatial has more rows than station data, extend backwards
                if len(spatial_data_raw) > len(station_data):
                    n_extra = len(spatial_data_raw) - len(station_data)
                    date_range = pd.date_range(
                        start=station_start - pd.Timedelta(days=n_extra),
                        periods=len(spatial_data_raw),
                        freq='D'
                    )
                else:
                    # Use station date range, trim if needed
                    date_range = pd.date_range(
                        start=station_start,
                        end=station_end,
                        freq='D'
                    )[:len(spatial_data_raw)]
                
                spatial_data = spatial_data_raw.copy()
                spatial_data.index = date_range[:len(spatial_data_raw)]
                print(f"    Reconstructed date range: {spatial_data.index.min()} to {spatial_data.index.max()}")
            else:
                # Fallback: use dates from config file
                config_start = config.get('start_date', '2004-01-01')
                config_end = config.get('end_date', '2024-01-01')
                print(f"    Using config date range: {config_start} to {config_end}")
                date_range = pd.date_range(
                    start=config_start,
                    end=config_end,
                    freq='D'
                )[:len(spatial_data_raw)]
                spatial_data = spatial_data_raw.copy()
                spatial_data.index = date_range
                print(f"    Reconstructed date range: {spatial_data.index.min()} to {spatial_data.index.max()}")
        else:
            # Try to parse as dates normally
            spatial_data = spatial_data_raw.copy()
            spatial_data.index = pd.to_datetime(spatial_data.index, errors='coerce')
            
            # Check if dates are in wrong range (1969-1970 indicates parsing error)
            if len(spatial_data) > 0:
                min_date = spatial_data.index.min()
                max_date = spatial_data.index.max()
                if pd.notna(min_date) and pd.notna(max_date) and (min_date.year < 2000 or max_date.year < 2000):
                    print(f"    Warning: Dates appear misparsed ({min_date} to {max_date})")
                    print(f"    Reconstructing date index from row count...")
                    # Use station data date range as reference
                    if len(station_data) > 0:
                        station_start = station_data.index.min()
                        station_end = station_data.index.max()
                        if len(spatial_data) > len(station_data):
                            n_extra = len(spatial_data) - len(station_data)
                            date_range = pd.date_range(
                                start=station_start - pd.Timedelta(days=n_extra),
                                periods=len(spatial_data),
                                freq='D'
                            )
                        else:
                            date_range = pd.date_range(
                                start=station_start,
                                end=station_end,
                                freq='D'
                            )[:len(spatial_data)]
                        spatial_data.index = date_range[:len(spatial_data)]
                        print(f"    Reconstructed date range: {spatial_data.index.min()} to {spatial_data.index.max()}")
                    else:
                        # Fallback: use dates from config file
                        config_start = config.get('start_date', '2004-01-01')
                        config_end = config.get('end_date', '2024-01-01')
                        print(f"    Using config date range: {config_start} to {config_end}")
                        date_range = pd.date_range(
                            start=config_start,
                            end=config_end,
                            freq='D'
                        )[:len(spatial_data)]
                        spatial_data.index = date_range
                        print(f"    Reconstructed date range: {spatial_data.index.min()} to {spatial_data.index.max()}")
    else:
        spatial_data = spatial_data_raw
    
    # Filter out invalid dates (NaT)
    spatial_data = spatial_data[spatial_data.index.notna()]
    spatial_data.columns = spatial_data.columns.astype(str)
    print(f"  Spatial CIMIS: {spatial_data.shape}")
    if len(spatial_data) > 0:
        print(f"    Date range: {spatial_data.index.min()} to {spatial_data.index.max()}")
    
    # Load GridMET predictions
    print("  Loading GridMET data...")
    gridmet_data = pd.read_csv(gridmet_file, index_col='date', parse_dates=['date'])
    gridmet_data.index = pd.to_datetime(gridmet_data.index, errors='coerce')
    # Filter out invalid dates (NaT)
    gridmet_data = gridmet_data[gridmet_data.index.notna()]
    gridmet_data.columns = gridmet_data.columns.astype(str)
    print(f"  GridMET: {gridmet_data.shape}")
    if len(gridmet_data) > 0:
        print(f"    Date range: {gridmet_data.index.min()} to {gridmet_data.index.max()}")

    if station_data.shape[1] == 0:
        print("  Warning: Station data has no stations after loading; skipping station plots.")
        return None, spatial_data, gridmet_data
    
    # Note: Rs unit conversion (MJ/m²/day to W/m²) is now handled during extraction in spatial_cimis_unified.py
    # Display value ranges for verification
    if variable == 'Rs':
        print(f"    Spatial CIMIS range: {spatial_data.min().min():.2f} to {spatial_data.max().max():.2f} W/m²")
        print(f"    Station range: {station_data.min().min():.2f} to {station_data.max().max():.2f} W/m²")
        print(f"    GridMET range: {gridmet_data.min().min():.2f} to {gridmet_data.max().max():.2f} W/m²")
    
    return station_data, spatial_data, gridmet_data


def load_station_metadata(station_list_file):
    """Load station names for plot titles."""
    print("Loading station metadata...")
    station_list = pd.read_csv(station_list_file)
    
    # Clean station names
    station_list['Name'] = station_list['Name'].str.replace(" ", "")
    station_list['Name'] = station_list['Name'].str.replace("/", "")
    station_list['Name'] = station_list['Name'].str.replace(".", "")
    station_list['Name'] = station_list['Name'].str.replace("-", "")
    
    return station_list


def compute_monthly_climatology(station_data, spatial_data, gridmet_data):
    """Compute monthly climatology (mean across all years)."""
    print("Computing monthly climatology...")
    
    station_monthly = station_data.groupby(station_data.index.month).mean(numeric_only=True)
    spatial_monthly = spatial_data.groupby(spatial_data.index.month).mean(numeric_only=True)
    gridmet_monthly = gridmet_data.groupby(gridmet_data.index.month).mean(numeric_only=True)
    
    station_monthly_std = station_data.groupby(station_data.index.month).std(numeric_only=True)
    spatial_monthly_std = spatial_data.groupby(spatial_data.index.month).std(numeric_only=True)
    gridmet_monthly_std = gridmet_data.groupby(gridmet_data.index.month).std(numeric_only=True)
    
    print(f"  Monthly climatology computed: {len(station_monthly.columns)} stations")
    
    return (station_monthly, spatial_monthly, gridmet_monthly,
            station_monthly_std, spatial_monthly_std, gridmet_monthly_std)


def compute_yearly_timeseries(station_data, spatial_data, gridmet_data, 
                              min_valid_days=60):
    """
    Compute yearly time series.
    Set years with too many missing days to NaN.
    Align all datasets to common years.
    """
    print("Computing yearly time series...")
    
    # Calculate yearly means
    yearly_means = station_data.groupby(station_data.index.year).mean(numeric_only=True)
    station_yearly = yearly_means.copy()
    
    # Mask years with too many NaN values
    if not yearly_means.empty:
        nan_counts = station_data.isnull().groupby(station_data.index.year).sum()
        relevant_nan_counts = nan_counts.reindex(columns=yearly_means.columns, fill_value=0)
        mask = relevant_nan_counts > min_valid_days
        station_yearly = station_yearly.mask(mask)
    
    spatial_yearly = spatial_data.groupby(spatial_data.index.year).mean(numeric_only=True)
    gridmet_yearly = gridmet_data.groupby(gridmet_data.index.year).mean(numeric_only=True)
    
    # Align to common years (intersection of all three)
    common_years = station_yearly.index.intersection(spatial_yearly.index).intersection(gridmet_yearly.index)
    
    if len(common_years) == 0:
        print(f"  Warning: No common years found between datasets!")
        print(f"    Station years: {sorted(station_yearly.index.tolist())}")
        print(f"    Spatial years: {sorted(spatial_yearly.index.tolist())}")
        print(f"    GridMET years: {sorted(gridmet_yearly.index.tolist())}")
        # Return empty DataFrames with proper structure
        return (pd.DataFrame(index=common_years, columns=station_yearly.columns),
                pd.DataFrame(index=common_years, columns=spatial_yearly.columns),
                pd.DataFrame(index=common_years, columns=gridmet_yearly.columns))
    
    station_yearly = station_yearly.loc[common_years]
    spatial_yearly = spatial_yearly.loc[common_years]
    gridmet_yearly = gridmet_yearly.loc[common_years]
    
    print(f"  Yearly data aligned to {len(common_years)} common years ({common_years.min()}-{common_years.max()})")
    
    return station_yearly, spatial_yearly, gridmet_yearly


def calculate_ylimits(station_list, station_monthly, spatial_monthly, gridmet_monthly,
                     station_yearly, spatial_yearly, gridmet_yearly):
    """Calculate consistent y-axis limits across all stations."""
    print("Calculating y-axis limits...")
    
    # For left plots (monthly): global min/max
    global_y_min_left = np.inf
    global_y_max_left = -np.inf
    
    # For right plots (yearly): max observed range
    max_observed_range_right = 0
    
    for sn in station_list:
        sn_str = str(sn)
        
        # Left plot data (monthly)
        if sn_str in station_monthly.columns and sn_str in spatial_monthly.columns:
            current_min = np.nanmin([
                np.nanmin(spatial_monthly[sn_str].values),
                np.nanmin(gridmet_monthly[sn_str].values),
                np.nanmin(station_monthly[sn_str].values)
            ])
            current_max = np.nanmax([
                np.nanmax(spatial_monthly[sn_str].values),
                np.nanmax(gridmet_monthly[sn_str].values),
                np.nanmax(station_monthly[sn_str].values)
            ])
            global_y_min_left = min(global_y_min_left, current_min)
            global_y_max_left = max(global_y_max_left, current_max)
        
        # Right plot data (yearly)
        if sn_str in station_yearly.columns and sn_str in spatial_yearly.columns:
            # Check if arrays are not empty
            spatial_vals = spatial_yearly[sn_str].values
            gridmet_vals = gridmet_yearly[sn_str].values
            station_vals = station_yearly[sn_str].values
            
            if len(spatial_vals) > 0 and len(gridmet_vals) > 0 and len(station_vals) > 0:
                current_min = np.nanmin([
                    np.nanmin(spatial_vals),
                    np.nanmin(gridmet_vals),
                    np.nanmin(station_vals)
                ])
                current_max = np.nanmax([
                    np.nanmax(spatial_vals),
                    np.nanmax(gridmet_vals),
                    np.nanmax(station_vals)
                ])
                if np.isfinite(current_min) and np.isfinite(current_max):
                    current_range = current_max - current_min
                    max_observed_range_right = max(max_observed_range_right, current_range)
    
    # Add padding
    padding_left = (global_y_max_left - global_y_min_left) * 0.05
    if padding_left == 0:
        padding_left = 0.5
    final_y_min_left = global_y_min_left - padding_left
    final_y_max_left = global_y_max_left + padding_left
    
    if max_observed_range_right == 0:
        final_target_range_right = 1.0
    else:
        final_target_range_right = max_observed_range_right * 1.10
    
    return final_y_min_left, final_y_max_left, final_target_range_right


def plot_station_comparison(station_list, station_names, variable,
                           station_monthly, spatial_monthly, gridmet_monthly,
                           station_yearly, spatial_yearly, gridmet_yearly,
                           output_file=None):
    """Create the multi-panel time series plot."""
    print("\nCreating plot...")
    
    num_rows = len(station_list)
    
    # Choose colors based on variable (Spatial CIMIS, GridMET, Station)
    if variable == 'Rs':
        COLORS = ['blue', 'red', '#ababab']  # Rs uses blue/red/gray
    elif variable == 'ETo':
        COLORS = ['#006ba4', '#ff800e', '#ababab']  # ETo uses blue/orange/gray
    else:
        COLORS = DEFAULT_COLORS  # Default: blue/orange/gray
    
    # Calculate y-axis limits
    final_y_min_left, final_y_max_left, final_target_range_right = calculate_ylimits(
        station_list, station_monthly, spatial_monthly, gridmet_monthly,
        station_yearly, spatial_yearly, gridmet_yearly
    )
    
    # Create figure
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(9, 9)
    fig.set_constrained_layout_pads(h_pad=1/100, hspace=0)
    
    raw_subfigs = fig.subfigures(nrows=num_rows, ncols=1)
    
    # Handle single vs multiple subfigures
    if num_rows == 1 and not isinstance(raw_subfigs, (list, np.ndarray)):
        subfigs_iterable = [raw_subfigs]
    else:
        subfigs_iterable = raw_subfigs
    
    # Variable-specific y-axis labels
    ylabel_map = {
        'ETo': 'Mean Daily ETo (mm/day)',
        'Tx': 'Mean Daily Tx (°C)',
        'Tn': 'Mean Daily Tn (°C)',
        'Rs': 'Mean Daily Rs (W/m²)',
        'Tdew': 'Mean Daily Tdew (°C)',
        'U2': 'Mean Daily U2 (m/s)'
    }
    ylabel = ylabel_map.get(variable, f'Mean Daily {variable}')
    
    # Plot each station
    for i, sn in enumerate(station_list):
        sn_str = str(sn)
        sn_int = int(sn)
        current_subfig = subfigs_iterable[i]
        
        ax = current_subfig.subplots(nrows=1, ncols=2)
        
        # Add title
        plot_letter = chr(ord('a') + i)
        title_content = f"{station_names.get(sn_int, f'Station {sn}')} ({sn})"
        full_title = f"{plot_letter}) {title_content}"
        ax[0].text(0, 1.02, full_title, transform=ax[0].transAxes,
                  ha='left', va='bottom', fontsize=12, fontweight='bold')
        
        # Check if station exists in data
        if sn_str not in station_monthly.columns:
            print(f"  Warning: Station {sn} not found in station data")
        if sn_str not in spatial_monthly.columns:
            print(f"  Warning: Station {sn} not found in spatial data")
        
        # --- Left plot: Monthly climatology ---
        if sn_str in station_monthly.columns and sn_str in spatial_monthly.columns:
            ax[0].plot(gridmet_monthly.index, gridmet_monthly[sn_str].values,
                      color=COLORS[1], label='GridMET', linestyle='-', 
                      marker='o', zorder=2)
            ax[0].plot(station_monthly.index, station_monthly[sn_str].values,
                      color=COLORS[2], label='Station', linestyle='-',
                      marker='o', zorder=3)
            ax[0].plot(spatial_monthly.index, spatial_monthly[sn_str].values,
                      color=COLORS[0], label='Spatial CIMIS', linestyle='-',
                      marker='o', zorder=4)
            ax[0].set_ylabel(ylabel, fontsize=10)
            ax[0].legend(fontsize=8)
            ax[0].set_ylim(final_y_min_left, final_y_max_left)
        
        # --- Right plot: Yearly time series ---
        if sn_str in station_yearly.columns and sn_str in spatial_yearly.columns:
            # Check if we have data to plot
            if len(station_yearly) > 0 and len(spatial_yearly) > 0 and len(gridmet_yearly) > 0:
                ax[1].plot(station_yearly.index.values, spatial_yearly[sn_str].values,
                          '-o', color=COLORS[0], label='Spatial CIMIS')
                ax[1].plot(station_yearly.index.values, gridmet_yearly[sn_str].values,
                          '-o', color=COLORS[1], label='GridMET')
                ax[1].plot(station_yearly.index.values, station_yearly[sn_str].values,
                          '-o', color=COLORS[2], label='Station')
                ax[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                
                # Apply consistent range
                spatial_vals = spatial_yearly[sn_str].values
                gridmet_vals = gridmet_yearly[sn_str].values
                station_vals = station_yearly[sn_str].values
                
                if len(spatial_vals) > 0 and len(gridmet_vals) > 0 and len(station_vals) > 0:
                    local_min = np.nanmin([
                        np.nanmin(spatial_vals),
                        np.nanmin(gridmet_vals),
                        np.nanmin(station_vals)
                    ])
                    local_max = np.nanmax([
                        np.nanmax(spatial_vals),
                        np.nanmax(gridmet_vals),
                        np.nanmax(station_vals)
                    ])
                    
                    if np.isfinite(local_min) and np.isfinite(local_max):
                        local_midpoint = (local_min + local_max) / 2
                        plot_y_min = local_midpoint - (final_target_range_right / 2)
                        plot_y_max = local_midpoint + (final_target_range_right / 2)
                        ax[1].set_ylim(plot_y_min, plot_y_max)
                    else:
                        ax[1].set_ylim(0, final_target_range_right)
            else:
                ax[1].text(0.5, 0.5, 'No yearly data\navailable', 
                          transform=ax[1].transAxes, ha='center', va='center',
                          fontsize=12)
                ax[1].set_ylim(0, 1)
        
        # --- X-axis labels ---
        if i == num_rows - 1:  # Last row
            # Custom month labels for left plot
            if len(spatial_monthly.index) == 12:
                tick_positions = [spatial_monthly.index[idx] for idx in [1, 3, 5, 7, 9, 11]]
                tick_labels = ['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec']
                ax[0].set_xticks(tick_positions)
                ax[0].set_xticklabels(tick_labels)
        else:
            ax[0].set_xlabel(' ')
            ax[1].set_xlabel(' ')
            
            # Phantom ticks for alignment
            if len(spatial_monthly.index) == 12:
                phantom_ticks = [spatial_monthly.index[idx] for idx in [1, 3, 5, 7, 9, 11]]
                ax[0].set_xticks(phantom_ticks)
                ax[0].set_xticklabels([''] * len(phantom_ticks))
            else:
                ax[0].set_xticklabels([])
            
            ax[1].set_xticklabels([])
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def create_station_timeseries_plot(config):
    """Create station time series comparison plot."""
    print("\n" + "="*70)
    print("STATION TIME SERIES PLOT")
    print("="*70)
    
    variable = config['variable']
    output_path = config['output_path']
    station_list_file = config.get('station_list_file', 'CIMIS_Stations.csv')
    
    # Get station list from config or use default
    station_list_str = config.get('plot_stations', '158,84,117,15,136')
    station_list = [int(s.strip()) for s in station_list_str.split(',')]
    
    print(f"\nVariable: {variable}")
    print(f"Plotting stations: {station_list}")
    
    # Load data
    station_data, spatial_data, gridmet_data = load_timeseries_data(config)

    if station_data is None:
        print("  Skipping station time series plot due to missing station data.")
        return
    
    # Load station names
    station_info = load_station_metadata(station_list_file)
    station_names = dict(zip(station_info['StationNbr'], station_info['Name']))
    
    # Compute climatologies
    (station_monthly, spatial_monthly, gridmet_monthly,
     station_monthly_std, spatial_monthly_std, gridmet_monthly_std) = compute_monthly_climatology(
        station_data, spatial_data, gridmet_data
    )
    
    # Compute yearly time series
    station_yearly, spatial_yearly, gridmet_yearly = compute_yearly_timeseries(
        station_data, spatial_data, gridmet_data
    )
    
    # Create plot
    output_file = Path(output_path) / f"{variable.lower()}_station_timeseries.png"
    plot_station_comparison(
        station_list, station_names, variable,
        station_monthly, spatial_monthly, gridmet_monthly,
        station_yearly, spatial_yearly, gridmet_yearly,
        output_file
    )
    
    print("\nStation time series plot complete!")


# ============================================================================
# STATION VALIDATION PLOT FUNCTIONS
# ============================================================================

def calculate_station_statistics(station_data, spatial_data, gridmet_data):
    """Calculate correlation, bias, and MAE for all stations."""
    print("\nCalculating station statistics...")
    
    # Convert station columns to str for consistent indexing
    station_data.columns = station_data.columns.astype(str)
    spatial_data.columns = spatial_data.columns.astype(str)
    gridmet_data.columns = gridmet_data.columns.astype(str)
    
    # Get common stations
    common_stations = list(set(station_data.columns) & 
                          set(spatial_data.columns) & 
                          set(gridmet_data.columns))
    
    print(f"  Found {len(common_stations)} common stations")
    
    # Initialize result series
    spatial_cor = pd.Series(index=common_stations, dtype=float)
    gridmet_cor = pd.Series(index=common_stations, dtype=float)
    spatial_bias = pd.Series(index=common_stations, dtype=float)
    gridmet_bias = pd.Series(index=common_stations, dtype=float)
    spatial_mae = pd.Series(index=common_stations, dtype=float)
    gridmet_mae = pd.Series(index=common_stations, dtype=float)
    
    # Calculate for each station (vectorized where possible)
    total_stations = len(common_stations)
    print(f"  Processing {total_stations} stations...")
    
    for idx, station in enumerate(common_stations, 1):
        if idx % 20 == 0 or idx == total_stations:
            print(f"    Station {idx}/{total_stations}...", end='\r', flush=True)
        
        # Spatial CIMIS statistics
        stat_vals = station_data[station]
        spat_vals = spatial_data[station]
        valid_mask = stat_vals.notna() & spat_vals.notna()
        
        if valid_mask.sum() > 2:  # Need at least 3 points for correlation
            stat_valid = stat_vals[valid_mask]
            spat_valid = spat_vals[valid_mask]
            
            # Correlation
            spatial_cor[station] = stat_valid.corr(spat_valid, method='pearson')
            # Bias (model - observed)
            spatial_bias[station] = (spat_valid - stat_valid).mean()
            # MAE
            spatial_mae[station] = (spat_valid - stat_valid).abs().mean()
        
        # GridMET statistics
        grid_vals = gridmet_data[station]
        valid_mask = stat_vals.notna() & grid_vals.notna()
        
        if valid_mask.sum() > 2:  # Need at least 3 points for correlation
            stat_valid = stat_vals[valid_mask]
            grid_valid = grid_vals[valid_mask]
            
            # Correlation
            gridmet_cor[station] = stat_valid.corr(grid_valid, method='pearson')
            # Bias (model - observed)
            gridmet_bias[station] = (grid_valid - stat_valid).mean()
            # MAE
            gridmet_mae[station] = (grid_valid - stat_valid).abs().mean()
    
    print(f"    Completed {total_stations}/{total_stations} stations.")
    
    stats = {
        'spatCor': spatial_cor,
        'gridCor': gridmet_cor,
        'spatBias': spatial_bias,
        'gridBias': gridmet_bias,
        'spatMae': spatial_mae,
        'gridMae': gridmet_mae
    }
    
    return stats, common_stations


def create_station_geodataframe(station_list_file, stats, common_stations):
    """Create GeoDataFrame with station locations and statistics."""
    print("\nCreating station GeoDataFrame...")
    
    # Load station metadata
    station_list = pd.read_csv(station_list_file)
    
    # Parse lat/lon if needed
    if 'HmsLatitude' in station_list.columns:
        if station_list['HmsLatitude'].dtype == 'object':
            station_list['HmsLatitude'] = station_list['HmsLatitude'].str.split('/ ').str[-1]
            station_list['HmsLongitude'] = station_list['HmsLongitude'].str.split('/ ').str[-1]
        
        station_list['HmsLatitude'] = pd.to_numeric(station_list['HmsLatitude'], errors='coerce')
        station_list['HmsLongitude'] = pd.to_numeric(station_list['HmsLongitude'], errors='coerce')
        
        station_list = station_list.rename(columns={
            'HmsLatitude': 'Latitude',
            'HmsLongitude': 'Longitude'
        })
    
    # Filter to common stations (convert station numbers to strings for matching)
    station_list['StationNbr'] = station_list['StationNbr'].astype(str)
    stations_subset = station_list[station_list['StationNbr'].isin(common_stations)].copy()
    
    # Create GeoDataFrame
    stations_gdf = gpd.GeoDataFrame(
        stations_subset,
        geometry=gpd.points_from_xy(stations_subset.Longitude, stations_subset.Latitude),
        crs="EPSG:4326"
    )
    
    # Add statistics
    for stat_name, stat_series in stats.items():
        stations_gdf[stat_name] = stations_gdf['StationNbr'].map(stat_series)
    
    print(f"  Created GeoDataFrame with {len(stations_gdf)} stations")
    
    return stations_gdf


def create_station_validation_plot(config):
    """Create station validation plot showing correlation, bias, and MAE maps."""
    print("\n" + "="*70)
    print("STATION VALIDATION PLOT")
    print("="*70)
    
    variable = config['variable']
    output_path = config['output_path']
    shapefile_path = config['shapefile_path']
    station_list_file = config.get('station_list_file', 'CIMIS_Stations.csv')
    
    print(f"\nVariable: {variable}")
    
    # Load time series data (reuse existing function)
    station_data, spatial_data, gridmet_data = load_timeseries_data(config)
    
    if station_data is None or station_data.empty:
        print("  Skipping station validation plot due to missing station data.")
        return
    
    # Calculate statistics
    stats, common_stations = calculate_station_statistics(
        station_data, spatial_data, gridmet_data
    )
    
    if len(common_stations) == 0:
        print("  No common stations found. Skipping plot.")
        return
    
    # Create GeoDataFrame
    stations_gdf = create_station_geodataframe(station_list_file, stats, common_stations)
    
    # Calculate dynamic colormap ranges based on data
    # Bias: symmetric range around 0 based on maximum absolute bias
    max_abs_bias_spat = stations_gdf['spatBias'].abs().max() if 'spatBias' in stations_gdf.columns else 0
    max_abs_bias_grid = stations_gdf['gridBias'].abs().max() if 'gridBias' in stations_gdf.columns else 0
    max_abs_bias = max(max_abs_bias_spat, max_abs_bias_grid)
    bias_vmax = max_abs_bias if max_abs_bias > 0 else 1.0  # Avoid zero range
    bias_vmin = -bias_vmax
    
    # MAE: range from 0 to maximum MAE
    max_mae_spat = stations_gdf['spatMae'].max() if 'spatMae' in stations_gdf.columns else 0
    max_mae_grid = stations_gdf['gridMae'].max() if 'gridMae' in stations_gdf.columns else 0
    max_mae = max(max_mae_spat, max_mae_grid)
    mae_vmax = max_mae if max_mae > 0 else 1.0  # Avoid zero range
    mae_vmin = 0.0
    
    print(f"\n  Colormap ranges:")
    print(f"    Bias: {bias_vmin:.2f} to {bias_vmax:.2f}")
    print(f"    MAE: {mae_vmin:.2f} to {mae_vmax:.2f}")
    
    # Load California boundary
    print("\nLoading California shapefile...")
    california = gpd.read_file(shapefile_path)
    if california.crs is None:
        california.crs = 'EPSG:4326'
    california = california.to_crs("EPSG:4326")
    
    # Create figure
    print("\nCreating plot...")
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(15, 10)
    
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle('Spatial CIMIS', fontsize=16)
    
    # Spatial CIMIS plots
    (ax1, ax2, ax3) = subfigs[0].subplots(nrows=1, ncols=3)
    
    # Plot California boundaries
    california.plot(ax=ax1, color="white", edgecolor="black", linewidth=1)
    california.plot(ax=ax2, color="white", edgecolor="black", linewidth=1)
    california.plot(ax=ax3, color="white", edgecolor="black", linewidth=1)
    
    # Correlation
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax1, cmap='cividis', legend=True, cax=cax, 
                     column='spatCor', marker='s', markersize=70, 
                     vmin=0.7, vmax=0.98)
    ax1.set_title('Correlation', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    
    # Bias
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax2, cmap='RdBu_r', column='spatBias', cax=cax, 
                     legend=True, marker='s', markersize=70, 
                     vmin=bias_vmin, vmax=bias_vmax)
    ax2.set_title('Bias', fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    # MAE
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax3, cmap='binary', column='spatMae', cax=cax, 
                     legend=True, marker='s', markersize=70, 
                     vmin=mae_vmin, vmax=mae_vmax)
    ax3.set_title('MAE', fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    # GridMET plots
    subfigs[1].suptitle('GridMET', fontsize=16)
    (ax1, ax2, ax3) = subfigs[1].subplots(nrows=1, ncols=3)
    
    # Plot California boundaries
    california.plot(ax=ax1, color="white", edgecolor="black", linewidth=1)
    california.plot(ax=ax2, color="white", edgecolor="black", linewidth=1)
    california.plot(ax=ax3, color="white", edgecolor="black", linewidth=1)
    
    # Correlation
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax1, cmap='cividis', legend=True, cax=cax, 
                     column='gridCor', marker='s', markersize=70, 
                     vmin=0.7, vmax=0.98)
    ax1.set_title('Correlation', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    
    # Bias
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax2, cmap='RdBu_r', column='gridBias', cax=cax, 
                     legend=True, marker='s', markersize=70, 
                     vmin=bias_vmin, vmax=bias_vmax)
    ax2.set_title('Bias', fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    # MAE
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax3, cmap='binary', column='gridMae', cax=cax, 
                     legend=True, marker='s', markersize=70, 
                     vmin=mae_vmin, vmax=mae_vmax)
    ax3.set_title('MAE', fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    # Save figure
    output_file = Path(output_path) / f"{variable.lower()}_station_validation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved to: {output_file}")
    
    plt.close()
    
    print("\nStation validation plot complete!")


# ============================================================================
# TAYLOR DIAGRAM FUNCTIONS
# ============================================================================

def taylor_statistics(predicted, reference, field='', norm=False):
    """
    Calculate Taylor diagram statistics.
    
    Parameters:
    -----------
    predicted : array-like
        Predicted/model values
    reference : array-like
        Reference/observed values
    field : str, optional
        Field name for error checking
    norm : bool, optional
        Whether to normalize statistics (default: False)
    
    Returns:
    --------
    dict : Dictionary with keys 'ccoef', 'crmsd', 'sdev', 'type'
    """
    p, r = error_check_stats(predicted, reference, field)
    
    # Calculate correlation coefficient
    ccoef_matrix = np.corrcoef(p, r)
    if ccoef_matrix.shape == (2, 2):
        ccoef = ccoef_matrix[0, 1]
    else:
        ccoef = float(ccoef_matrix[0]) if len(ccoef_matrix) > 0 else 0.0
    
    # Calculate centered root-mean-square (RMS) difference
    crmsd = [0.0, centered_rms_dev(p, r)]
    
    # Calculate standard deviations
    sdevp = np.std(p)
    sdevr = np.std(r)
    sdev = [sdevr, sdevp]
    
    # Normalize if requested
    if norm:
        if sdevr > 0:
            sdev = [sdevr/sdevr, sdevp/sdevr]
            crmsd = [0.0, centered_rms_dev(p, r)/sdevr]
        else:
            # Handle edge case where reference std is zero
            sdev = [1.0, 0.0]
            crmsd = [0.0, 0.0]
    
    stats = {'ccoef': ccoef, 'crmsd': crmsd, 'sdev': sdev}
    stats['type'] = 'normalized' if norm else 'unnormalized'
    
    return stats


def create_taylor_stations_plot(config):
    """Create Taylor diagram comparing all stations (Spatial CIMIS vs GridMET)."""
    if not HAS_SKILL_METRICS:
        print("  Skipping Taylor diagram: skill_metrics not available")
        return
    
    print("\n" + "="*70)
    print("TAYLOR DIAGRAM - STATIONS")
    print("="*70)
    
    variable = config['variable']
    output_path = config['output_path']
    
    # Load time series data
    station_data, spatial_data, gridmet_data = load_timeseries_data(config)
    
    if station_data is None or station_data.empty:
        print("  Skipping Taylor diagram due to missing station data.")
        return
    
    # Calculate statistics for each station
    print("\nCalculating Taylor statistics for all stations...")
    tstats = []
    gstats = []
    
    common_stations = list(set(station_data.columns) & 
                          set(spatial_data.columns) & 
                          set(gridmet_data.columns))
    
    for station in common_stations:
        # Filter valid data
        valid_mask = (station_data[station].notna() & 
                     spatial_data[station].notna() & 
                     gridmet_data[station].notna())
        
        if valid_mask.sum() < 10:  # Need minimum data points
            continue
        
        stat_vals = station_data[station][valid_mask].values
        spat_vals = spatial_data[station][valid_mask].values
        grid_vals = gridmet_data[station][valid_mask].values
        
        # Calculate statistics
        try:
            tstat = taylor_statistics(spat_vals, stat_vals, norm=True)
            gstat = taylor_statistics(grid_vals, stat_vals, norm=True)
            
            tstats.append({
                'sdev': tstat['sdev'][1],
                'crmsd': tstat['crmsd'][1],
                'ccoef': tstat['ccoef']
            })
            gstats.append({
                'sdev': gstat['sdev'][1],
                'crmsd': gstat['crmsd'][1],
                'ccoef': gstat['ccoef']
            })
        except Exception as e:
            print(f"    Warning: Could not calculate stats for station {station}: {e}")
            continue
    
    if len(tstats) == 0:
        print("  No valid statistics calculated. Skipping plot.")
        return
    
    # Extract arrays for plotting
    tsdev = np.array([s['sdev'] for s in tstats])
    tcrmsd = np.array([s['crmsd'] for s in tstats])
    tcoef = np.array([s['ccoef'] for s in tstats])
    
    gsdev = np.array([s['sdev'] for s in gstats])
    gcrmsd = np.array([s['crmsd'] for s in gstats])
    gcoef = np.array([s['ccoef'] for s in gstats])
    
    # Create figure
    print("\nCreating Taylor diagram...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Colors
    colors = ['#006ba4', '#ff800e']  # Blue for Spatial CIMIS, Orange for GridMET
    
    # Plot reference (observation) point
    sm.taylor_diagram(ax, [1.0], [0.0], [1.0],
                     styleOBS='-',
                     colOBS='k',
                     markerobs='.',
                     titleOBS='observation',
                     numberPanels=1,
                     markersymbol='.')
    
    # Plot GridMET stations
    sm.taylor_diagram(ax, gsdev, gcrmsd, gcoef,
                     numberPanels=1,
                     markerLabelColor='k',
                     markercolor=colors[1],
                     markerSize=15,
                     overlay='on')
    
    # Plot Spatial CIMIS stations
    sm.taylor_diagram(ax, tsdev, tcrmsd, tcoef,
                     numberPanels=1,
                     markerLabelColor='k',
                     markercolor=colors[0],
                     markerSize=15,
                     overlay='on')
    
    ax.set_title(f'Taylor Diagram - {variable} (All Stations)', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], 
               markersize=10, label='Spatial CIMIS'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], 
               markersize=10, label='GridMET'),
        Line2D([0], [0], marker='.', color='k', markersize=10, label='Observation')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save figure
    output_file = Path(output_path) / f"{variable.lower()}_taylor_stations.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Saved to: {output_file}")
    
    plt.close()
    print("\nTaylor diagram (stations) complete!")


def create_taylor_seasonal_plot(config):
    """Create Taylor diagram comparing seasonal data."""
    if not HAS_SKILL_METRICS:
        print("  Skipping Taylor diagram: skill_metrics not available")
        return
    
    print("\n" + "="*70)
    print("TAYLOR DIAGRAM - SEASONAL")
    print("="*70)
    
    variable = config['variable']
    output_path = config['output_path']
    
    # Load time series data
    station_data, spatial_data, gridmet_data = load_timeseries_data(config)
    
    if station_data is None or station_data.empty:
        print("  Skipping Taylor diagram due to missing station data.")
        return
    
    # Calculate seasonal means
    print("\nCalculating seasonal statistics...")
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    
    tstats_seas = {}
    gstats_seas = {}
    
    # Get common stations
    common_stations = list(set(station_data.columns) & 
                          set(spatial_data.columns) & 
                          set(gridmet_data.columns))
    
    for season_name, months in seasons.items():
        # Filter to season
        season_mask = station_data.index.month.isin(months)
        stat_seasonal = station_data[season_mask].mean()
        spat_seasonal = spatial_data[season_mask].mean()
        grid_seasonal = gridmet_data[season_mask].mean()
        
        # Calculate statistics across all stations
        valid_mask = (stat_seasonal.notna() & 
                     spat_seasonal.notna() & 
                     grid_seasonal.notna())
        
        if valid_mask.sum() < 5:
            continue
        
        stat_vals = stat_seasonal[valid_mask].values
        spat_vals = spat_seasonal[valid_mask].values
        grid_vals = grid_seasonal[valid_mask].values
        
        try:
            tstat = taylor_statistics(spat_vals, stat_vals, norm=True)
            gstat = taylor_statistics(grid_vals, stat_vals, norm=True)
            
            tstats_seas[season_name] = {
                'sdev': tstat['sdev'][1],
                'crmsd': tstat['crmsd'][1],
                'ccoef': tstat['ccoef']
            }
            gstats_seas[season_name] = {
                'sdev': gstat['sdev'][1],
                'crmsd': gstat['crmsd'][1],
                'ccoef': gstat['ccoef']
            }
        except Exception as e:
            print(f"    Warning: Could not calculate stats for {season_name}: {e}")
            continue
    
    if len(tstats_seas) == 0:
        print("  No valid seasonal statistics calculated. Skipping plot.")
        return
    
    # Extract arrays for plotting
    season_order = ['DJF', 'MAM', 'JJA', 'SON']
    tsdev_seas = np.array([tstats_seas[s]['sdev'] for s in season_order if s in tstats_seas])
    tcrmsd_seas = np.array([tstats_seas[s]['crmsd'] for s in season_order if s in tstats_seas])
    tcoef_seas = np.array([tstats_seas[s]['ccoef'] for s in season_order if s in tstats_seas])
    
    gsdev_seas = np.array([gstats_seas[s]['sdev'] for s in season_order if s in gstats_seas])
    gcrmsd_seas = np.array([gstats_seas[s]['crmsd'] for s in season_order if s in gstats_seas])
    gcoef_seas = np.array([gstats_seas[s]['ccoef'] for s in season_order if s in gstats_seas])
    
    # Create figure
    print("\nCreating Taylor diagram...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#006ba4', '#ff800e']
    markers = ['*', '^', 'o', 'P']
    
    # Plot reference
    sm.taylor_diagram(ax, [1.0], [0.0], [1.0],
                     styleOBS='-',
                     colOBS='k',
                     markerobs='.',
                     titleOBS='observation',
                     numberPanels=1,
                     markersymbol='.')
    
    # Plot GridMET seasons
    for i, (sdev, crmsd, coef, marker) in enumerate(zip(gsdev_seas, gcrmsd_seas, gcoef_seas, markers[:len(gsdev_seas)])):
        sm.taylor_diagram(ax, np.asarray([sdev]), np.asarray([crmsd]), np.asarray([coef]),
                         numberPanels=1,
                         markersymbol=marker,
                         markercolor=colors[1],
                         markerSize=15,
                         overlay='on')
    
    # Plot Spatial CIMIS seasons
    for i, (sdev, crmsd, coef, marker) in enumerate(zip(tsdev_seas, tcrmsd_seas, tcoef_seas, markers[:len(tsdev_seas)])):
        sm.taylor_diagram(ax, np.asarray([sdev]), np.asarray([crmsd]), np.asarray([coef]),
                         numberPanels=1,
                         markersymbol=marker,
                         markercolor=colors[0],
                         markerSize=15,
                         overlay='on')
    
    ax.set_title(f'Taylor Diagram - {variable} (Seasonal)', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=colors[0], 
               markersize=10, label='Spatial CIMIS DJF'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=colors[0], 
               markersize=10, label='Spatial CIMIS MAM'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], 
               markersize=10, label='Spatial CIMIS JJA'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor=colors[0], 
               markersize=10, label='Spatial CIMIS SON'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=colors[1], 
               markersize=10, label='GridMET DJF'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=colors[1], 
               markersize=10, label='GridMET MAM'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], 
               markersize=10, label='GridMET JJA'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor=colors[1], 
               markersize=10, label='GridMET SON'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Save figure
    output_file = Path(output_path) / f"{variable.lower()}_taylor_seasonal.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Saved to: {output_file}")
    
    plt.close()
    print("\nTaylor diagram (seasonal) complete!")


def create_taylor_yearly_plot(config):
    """Create Taylor diagram comparing yearly data."""
    if not HAS_SKILL_METRICS:
        print("  Skipping Taylor diagram: skill_metrics not available")
        return
    
    print("\n" + "="*70)
    print("TAYLOR DIAGRAM - YEARLY")
    print("="*70)
    
    variable = config['variable']
    output_path = config['output_path']
    
    # Load time series data
    station_data, spatial_data, gridmet_data = load_timeseries_data(config)
    
    if station_data is None or station_data.empty:
        print("  Skipping Taylor diagram due to missing station data.")
        return
    
    # Calculate yearly means
    print("\nCalculating yearly statistics...")
    station_yearly = station_data.groupby(station_data.index.year).mean()
    spatial_yearly = spatial_data.groupby(spatial_data.index.year).mean()
    gridmet_yearly = gridmet_data.groupby(gridmet_data.index.year).mean()
    
    # Align to common years
    common_years = (station_yearly.index.intersection(spatial_yearly.index)
                   .intersection(gridmet_yearly.index))
    
    if len(common_years) == 0:
        print("  No common years found. Skipping plot.")
        return
    
    station_yearly = station_yearly.loc[common_years]
    spatial_yearly = spatial_yearly.loc[common_years]
    gridmet_yearly = gridmet_yearly.loc[common_years]
    
    # Calculate statistics for each year
    tstats_year = []
    gstats_year = []
    
    common_stations = list(set(station_yearly.columns) & 
                          set(spatial_yearly.columns) & 
                          set(gridmet_yearly.columns))
    
    for year in common_years:
        stat_vals = station_yearly.loc[year, common_stations].values
        spat_vals = spatial_yearly.loc[year, common_stations].values
        grid_vals = gridmet_yearly.loc[year, common_stations].values
        
        # Filter valid data
        valid_mask = (pd.notna(stat_vals) & pd.notna(spat_vals) & pd.notna(grid_vals))
        
        if valid_mask.sum() < 5:
            continue
        
        stat_valid = stat_vals[valid_mask]
        spat_valid = spat_vals[valid_mask]
        grid_valid = grid_vals[valid_mask]
        
        try:
            tstat = taylor_statistics(spat_valid, stat_valid, norm=True)
            gstat = taylor_statistics(grid_valid, stat_valid, norm=True)
            
            tstats_year.append({
                'sdev': tstat['sdev'][1],
                'crmsd': tstat['crmsd'][1],
                'ccoef': tstat['ccoef'],
                'year': year
            })
            gstats_year.append({
                'sdev': gstat['sdev'][1],
                'crmsd': gstat['crmsd'][1],
                'ccoef': gstat['ccoef'],
                'year': year
            })
        except Exception as e:
            print(f"    Warning: Could not calculate stats for year {year}: {e}")
            continue
    
    if len(tstats_year) == 0:
        print("  No valid yearly statistics calculated. Skipping plot.")
        return
    
    # Extract arrays for plotting
    tsdev_year = np.array([s['sdev'] for s in tstats_year])
    tcrmsd_year = np.array([s['crmsd'] for s in tstats_year])
    tcoef_year = np.array([s['ccoef'] for s in tstats_year])
    
    gsdev_year = np.array([s['sdev'] for s in gstats_year])
    gcrmsd_year = np.array([s['crmsd'] for s in gstats_year])
    gcoef_year = np.array([s['ccoef'] for s in gstats_year])
    
    # Create figure
    print("\nCreating Taylor diagram...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#006ba4', '#ff800e']
    
    # Plot reference
    sm.taylor_diagram(ax, [1.0], [0.0], [1.0],
                     styleOBS='-',
                     colOBS='k',
                     markerobs='.',
                     titleOBS='observation',
                     numberPanels=1,
                     markersymbol='.')
    
    # Plot GridMET years (skip first year as reference)
    if len(gsdev_year) > 1:
        sm.taylor_diagram(ax, gsdev_year[1:], gcrmsd_year[1:], gcoef_year[1:],
                         numberPanels=1,
                         markerLabelColor='k',
                         markercolor=colors[1],
                         markerSize=15,
                         overlay='on')
    
    # Plot Spatial CIMIS years
    sm.taylor_diagram(ax, tsdev_year, tcrmsd_year, tcoef_year,
                     numberPanels=1,
                     markerLabelColor='k',
                     markercolor=colors[0],
                     markerSize=15,
                     overlay='on')
    
    ax.set_title(f'Taylor Diagram - {variable} (Yearly)', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], 
               markersize=10, label='Spatial CIMIS'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], 
               markersize=10, label='GridMET'),
        Line2D([0], [0], marker='.', color='k', markersize=10, label='Observation')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save figure
    output_file = Path(output_path) / f"{variable.lower()}_taylor_yearly.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved to: {output_file}")
    
    plt.close()
    print("\nTaylor diagram (yearly) complete!")


def create_taylor_regional_plot(config):
    """Create Taylor diagram comparing regional data."""
    if not HAS_SKILL_METRICS:
        print("  Skipping Taylor diagram: skill_metrics not available")
        return
    
    print("\n" + "="*70)
    print("TAYLOR DIAGRAM - REGIONAL")
    print("="*70)
    
    variable = config['variable']
    output_path = config['output_path']
    station_list_file = config.get('station_list_file', 'CIMIS_Stations.csv')
    
    # Load station metadata for regions
    try:
        station_list = pd.read_csv(station_list_file)
        if 'Region' not in station_list.columns:
            print("  Station list does not have 'Region' column. Skipping regional Taylor diagram.")
            return
    except Exception as e:
        print(f"  Could not load station list: {e}. Skipping regional Taylor diagram.")
        return
    
    # Load time series data
    station_data, spatial_data, gridmet_data = load_timeseries_data(config)
    
    if station_data is None or station_data.empty:
        print("  Skipping Taylor diagram due to missing station data.")
        return
    
    # Get region mapping
    station_list['StationNbr'] = station_list['StationNbr'].astype(str)
    region_map = dict(zip(station_list['StationNbr'], station_list['Region']))
    
    # Calculate regional means
    print("\nCalculating regional statistics...")
    regions = station_list['Region'].dropna().unique()
    
    tstats_reg = {}
    gstats_reg = {}
    
    for region in regions:
        # Get stations in this region
        region_stations = [s for s in station_data.columns 
                          if str(s) in region_map and region_map[str(s)] == region]
        
        if len(region_stations) == 0:
            continue
        
        # Calculate regional means
        stat_regional = station_data[region_stations].mean(axis=1)
        spat_regional = spatial_data[region_stations].mean(axis=1)
        grid_regional = gridmet_data[region_stations].mean(axis=1)
        
        # Filter valid data
        valid_mask = (stat_regional.notna() & 
                     spat_regional.notna() & 
                     grid_regional.notna())
        
        if valid_mask.sum() < 30:  # Need minimum data points
            continue
        
        stat_vals = stat_regional[valid_mask].values
        spat_vals = spat_regional[valid_mask].values
        grid_vals = grid_regional[valid_mask].values
        
        try:
            tstat = taylor_statistics(spat_vals, stat_vals, norm=True)
            gstat = taylor_statistics(grid_vals, stat_vals, norm=True)
            
            tstats_reg[region] = {
                'sdev': tstat['sdev'][1],
                'crmsd': tstat['crmsd'][1],
                'ccoef': tstat['ccoef']
            }
            gstats_reg[region] = {
                'sdev': gstat['sdev'][1],
                'crmsd': gstat['crmsd'][1],
                'ccoef': gstat['ccoef']
            }
        except Exception as e:
            print(f"    Warning: Could not calculate stats for region {region}: {e}")
            continue
    
    if len(tstats_reg) == 0:
        print("  No valid regional statistics calculated. Skipping plot.")
        return
    
    # Extract arrays for plotting
    regions_list = sorted(tstats_reg.keys())
    tsdev_reg = np.array([tstats_reg[r]['sdev'] for r in regions_list])
    tcrmsd_reg = np.array([tstats_reg[r]['crmsd'] for r in regions_list])
    tcoef_reg = np.array([tstats_reg[r]['ccoef'] for r in regions_list])
    
    gsdev_reg = np.array([gstats_reg[r]['sdev'] for r in regions_list])
    gcrmsd_reg = np.array([gstats_reg[r]['crmsd'] for r in regions_list])
    gcoef_reg = np.array([gstats_reg[r]['ccoef'] for r in regions_list])
    
    # Create figure
    print("\nCreating Taylor diagram...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#006ba4', '#ff800e']
    
    # Plot reference
    sm.taylor_diagram(ax, [1.0], [0.0], [1.0],
                     styleOBS='-',
                     colOBS='k',
                     markerobs='.',
                     titleOBS='observation',
                     numberPanels=1,
                     markersymbol='.')
    
    # Plot GridMET regions
    sm.taylor_diagram(ax, gsdev_reg, gcrmsd_reg, gcoef_reg,
                     numberPanels=1,
                     markerLabelColor='k',
                     markercolor=colors[1],
                     markerSize=15,
                     overlay='on')
    
    # Plot Spatial CIMIS regions
    sm.taylor_diagram(ax, tsdev_reg, tcrmsd_reg, tcoef_reg,
                     numberPanels=1,
                     markerLabelColor='k',
                     markercolor=colors[0],
                     markerSize=15,
                     overlay='on')
    
    ax.set_title(f'Taylor Diagram - {variable} (Regional)', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], 
               markersize=10, label='Spatial CIMIS'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], 
               markersize=10, label='GridMET'),
        Line2D([0], [0], marker='.', color='k', markersize=10, label='Observation')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save figure
    output_file = Path(output_path) / f"{variable.lower()}_taylor_regional.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Saved to: {output_file}")
    
    plt.close()
    print("\nTaylor diagram (regional) complete!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    if len(sys.argv) < 2:
        config_file = 'spatial_cimis_config.txt'
    else:
        config_file = sys.argv[1]
    
    print("="*70)
    print("SPATIAL CIMIS COMPARISON PLOTTING")
    print("="*70)
    print(f"\nConfig file: {config_file}")
    
    # Load configuration
    config = load_config(config_file)
    
    # Check which plots to create
    create_spatial_plot = config.get('plot_spatial_comparison', True)
    create_timeseries_plot = config.get('plot_station_timeseries', True)
    create_validation_plot = config.get('plot_station_validation', True)
    create_taylor_stations = config.get('plot_taylor_stations', False)
    create_taylor_seasonal = config.get('plot_taylor_seasonal', False)
    create_taylor_yearly = config.get('plot_taylor_yearly', False)
    create_taylor_regional = config.get('plot_taylor_regional', False)
    
    print(f"\nPlots to create:")
    print(f"  Spatial comparison: {create_spatial_plot}")
    print(f"  Station time series: {create_timeseries_plot}")
    print(f"  Station validation: {create_validation_plot}")
    print(f"  Taylor diagram (stations): {create_taylor_stations}")
    print(f"  Taylor diagram (seasonal): {create_taylor_seasonal}")
    print(f"  Taylor diagram (yearly): {create_taylor_yearly}")
    print(f"  Taylor diagram (regional): {create_taylor_regional}")
    
    # Create plots
    if create_spatial_plot:
        try:
            create_spatial_comparison_plot(config)
        except Exception as e:
            print(f"\nError creating spatial comparison plot: {e}")
            import traceback
            traceback.print_exc()
    
    if create_timeseries_plot:
        try:
            create_station_timeseries_plot(config)
        except Exception as e:
            print(f"\nError creating station time series plot: {e}")
            import traceback
            traceback.print_exc()
    
    if create_validation_plot:
        try:
            create_station_validation_plot(config)
        except Exception as e:
            print(f"\nError creating station validation plot: {e}")
            import traceback
            traceback.print_exc()
    
    # Create Taylor diagrams
    if create_taylor_stations:
        try:
            create_taylor_stations_plot(config)
        except Exception as e:
            print(f"\nError creating Taylor diagram (stations): {e}")
            import traceback
            traceback.print_exc()
    
    if create_taylor_seasonal:
        try:
            create_taylor_seasonal_plot(config)
        except Exception as e:
            print(f"\nError creating Taylor diagram (seasonal): {e}")
            import traceback
            traceback.print_exc()
    
    if create_taylor_yearly:
        try:
            create_taylor_yearly_plot(config)
        except Exception as e:
            print(f"\nError creating Taylor diagram (yearly): {e}")
            import traceback
            traceback.print_exc()
    
    if create_taylor_regional:
        try:
            create_taylor_regional_plot(config)
        except Exception as e:
            print(f"\nError creating Taylor diagram (regional): {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

