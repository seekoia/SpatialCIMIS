#!/usr/bin/env python3
"""
Combined plotting script for Spatial CIMIS analysis.

Creates two types of plots:
1. Spatial comparison: 3-panel maps (Spatial CIMIS, GridMET, Bias)
2. Station time series: Multi-panel plots (monthly climatology + yearly time series)

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

def create_spatial_comparison_plot(config):
    """Create 3-panel spatial comparison plot (Spatial CIMIS, GridMET, Bias)."""
    print("\n" + "="*70)
    print("SPATIAL COMPARISON PLOT")
    print("="*70)
    
    variable = config['variable']
    output_path = config['output_path']
    shapefile_path = config['shapefile_path']
    
    # Construct file paths
    spatial_file = Path(output_path) / config['spatial_mean_output'].replace('{variable}', variable.lower())
    gridmet_file = Path(output_path) / config['gridmet_mean_output'].replace('{variable}', variable.lower())
    
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
    
    # Convert Rs from MJ/m²/day to W/m² to match GridMET units
    if variable == 'Rs':
        print(f"  Converting Rs from MJ/m²/day to W/m² (× 11.57)...")
        spatial_data = spatial_data * 11.57
        print(f"  New value range: {float(spatial_data.min()):.2f} to {float(spatial_data.max()):.2f}")
    
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
        im0 = axes[0].imshow(spatial_values, cmap=params['cmap'],
                           vmin=params['vmin'], vmax=params['vmax'],
                           origin='upper', aspect='auto', interpolation='nearest')
        plt.colorbar(im0, ax=axes[0], label=params['label'], shrink=0.8)
        axes[0].set_title('Spatial CIMIS', fontsize=12, fontweight='bold')
    
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    
    # Panel 2: GridMET
    if gridmet_values is not None:
        im1 = axes[1].imshow(gridmet_values, cmap=params['cmap'],
                           vmin=params['vmin'], vmax=params['vmax'],
                           origin='upper', aspect='auto', interpolation='nearest')
        plt.colorbar(im1, ax=axes[1], label=params['label'], shrink=0.8)
        axes[1].set_title('GridMET', fontsize=12, fontweight='bold')
    
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    
    # Panel 3: Bias
    if bias_values is not None:
        # Center colormap at zero
        vmax_bias = max(abs(np.nanmin(bias_values)), abs(np.nanmax(bias_values)))
        im2 = axes[2].imshow(bias_values, cmap='RdBu_r',
                           vmin=-vmax_bias, vmax=vmax_bias,
                           origin='upper', aspect='auto', interpolation='nearest')
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
    
    # Construct file paths based on variable
    station_file = Path(output_path) / f"station_{variable.lower()}_data.csv"
    spatial_file = Path(output_path) / f"spatial_cimis_station_{variable.lower()}.csv"
    gridmet_file = Path(output_path) / f"gridmet_station_{variable.lower()}.csv"
    
    # Load station observations
    station_data = pd.read_csv(station_file, index_col='date', parse_dates=True)
    station_data.columns = station_data.columns.astype(str)
    print(f"  Station data: {station_data.shape}")
    
    # Load Spatial CIMIS predictions
    spatial_data = pd.read_csv(spatial_file, index_col='date', parse_dates=True)
    spatial_data.columns = spatial_data.columns.astype(str)
    print(f"  Spatial CIMIS: {spatial_data.shape}")
    
    # Load GridMET predictions
    gridmet_data = pd.read_csv(gridmet_file, index_col='date', parse_dates=True)
    gridmet_data.columns = gridmet_data.columns.astype(str)
    print(f"  GridMET: {gridmet_data.shape}")
    
    # Convert Rs from MJ/m²/day to W/m² to match GridMET and Station units
    if variable == 'Rs':
        print("  Converting Spatial CIMIS Rs from MJ/m²/day to W/m² (× 11.57)...")
        spatial_data = spatial_data * 11.57
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
            current_min = np.nanmin([
                np.nanmin(spatial_yearly[sn_str].values),
                np.nanmin(gridmet_yearly[sn_str].values),
                np.nanmin(station_yearly[sn_str].values)
            ])
            current_max = np.nanmax([
                np.nanmax(spatial_yearly[sn_str].values),
                np.nanmax(gridmet_yearly[sn_str].values),
                np.nanmax(station_yearly[sn_str].values)
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
            ax[1].plot(station_yearly.index.values, spatial_yearly[sn_str].values,
                      '-o', color=COLORS[0], label='Spatial CIMIS')
            ax[1].plot(station_yearly.index.values, gridmet_yearly[sn_str].values,
                      '-o', color=COLORS[1], label='GridMET')
            ax[1].plot(station_yearly.index.values, station_yearly[sn_str].values,
                      '-o', color=COLORS[2], label='Station')
            ax[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            # Apply consistent range
            local_min = np.nanmin([
                np.nanmin(spatial_yearly[sn_str].values),
                np.nanmin(gridmet_yearly[sn_str].values),
                np.nanmin(station_yearly[sn_str].values)
            ])
            local_max = np.nanmax([
                np.nanmax(spatial_yearly[sn_str].values),
                np.nanmax(gridmet_yearly[sn_str].values),
                np.nanmax(station_yearly[sn_str].values)
            ])
            
            if np.isfinite(local_min) and np.isfinite(local_max):
                local_midpoint = (local_min + local_max) / 2
                plot_y_min = local_midpoint - (final_target_range_right / 2)
                plot_y_max = local_midpoint + (final_target_range_right / 2)
                ax[1].set_ylim(plot_y_min, plot_y_max)
            else:
                ax[1].set_ylim(0, final_target_range_right)
        
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
# MAIN
# ============================================================================

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python plot_comparison.py <config_file>")
        sys.exit(1)
    
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
    
    print(f"\nPlots to create:")
    print(f"  Spatial comparison: {create_spatial_plot}")
    print(f"  Station time series: {create_timeseries_plot}")
    
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
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

