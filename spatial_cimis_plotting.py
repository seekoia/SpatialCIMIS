#!/usr/bin/env python3
"""
Spatial CIMIS Plotting Script

This script handles visualization of:
1. Station locations and data
2. Spatial CIMIS maps
3. GridMET comparisons
4. Bias and error analysis
5. Time series plots

Configuration is read from a text file (analysis_config.txt by default)
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import rioxarray as rio
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False


def load_config(config_file='analysis_config.txt'):
    """Load configuration from a text file."""
    config = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if value.lower() in ['true', 'false']:
                        config[key] = value.lower() == 'true'
                    elif value.isdigit():
                        config[key] = int(value)
                    else:
                        config[key] = value
        print(f"✓ Configuration loaded from: {config_file}")
    except FileNotFoundError:
        print(f"Warning: Config file {config_file} not found.")
        config = {}
    
    return config


def plot_spatial_climatology(config):
    """
    Plot spatial climatology map.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    """
    print("\n1. Plotting spatial climatology...")
    
    variable = config.get('variable', 'ETo')
    output_path = config.get('output_path', 'output/')
    
    # Load climatology file
    clim_file = output_path + f'spatial_mean_{variable.lower()}.nc'
    
    if not os.path.exists(clim_file):
        print(f"  Warning: Climatology file not found: {clim_file}")
        return
    
    ds = xr.open_dataset(clim_file)
    
    # Get the variable (might be stored as DataArray or in a Dataset)
    if variable in ds.data_vars:
        data = ds[variable].values
    else:
        # Try to get the first data variable
        data_vars = [v for v in ds.data_vars if v not in ['lat', 'lon', 'x', 'y']]
        if data_vars:
            data = ds[data_vars[0]].values
        else:
            data = ds.to_array().values[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot using imshow
    im = ax.imshow(data, cmap='viridis', origin='lower', aspect='auto',
                   interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Set variable-specific labels
    var_labels = {
        'ETo': 'Reference ET (mm/day)',
        'Tx': 'Maximum Temperature (°C)',
        'Tn': 'Minimum Temperature (°C)',
        'Rs': 'Solar Radiation (MJ/m²/day)'
    }
    cbar.set_label(var_labels.get(variable, variable), fontsize=12)
    
    # Add title
    ax.set_title(f'Spatial CIMIS {variable} Climatology\n2004-2023 Mean',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Grid X', fontsize=11)
    ax.set_ylabel('Grid Y', fontsize=11)
    
    # Save figure
    plot_file = output_path + f'spatial_{variable.lower()}_climatology.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved to: {plot_file}")
    
    ds.close()


def plot_sample_days(config, n_days=6):
    """
    Plot sample days from spatial data.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    n_days : int
        Number of sample days to plot
    """
    print(f"\n2. Plotting {n_days} sample days...")
    
    variable = config.get('variable', 'ETo')
    netcdf_path = config.get('spatial_netcdf_path', '/group/moniergrp/SpatialCIMIS/netcdf/')
    output_path = config.get('output_path', 'output/')
    
    # Load one year of data (e.g., 2023)
    year = 2023
    data_file = netcdf_path + f'spatial_cimis_{variable.lower()}_{year}.nc'
    
    if not os.path.exists(data_file):
        print(f"  Warning: Data file not found: {data_file}")
        # Try to find any available year
        import glob
        files = glob.glob(netcdf_path + f'spatial_cimis_{variable.lower()}_20*.nc')
        if files:
            data_file = files[-1]
            year = data_file.split('_')[-1].replace('.nc', '')
            print(f"  Using available file: {data_file}")
        else:
            print(f"  No data files found!")
            return
    
    ds = xr.open_dataset(data_file)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Sample days throughout the year
    sample_indices = np.linspace(0, len(ds.time)-1, n_days, dtype=int)
    
    # Variable-specific plot parameters
    var_params = {
        'ETo': {'cmap': 'YlGnBu', 'vmin': 0, 'vmax': 10, 'label': 'ET (mm/day)'},
        'Tx': {'cmap': 'RdYlBu_r', 'vmin': -10, 'vmax': 45, 'label': 'Temp (°C)'},
        'Tn': {'cmap': 'RdYlBu_r', 'vmin': -15, 'vmax': 30, 'label': 'Temp (°C)'},
        'Rs': {'cmap': 'plasma', 'vmin': 0, 'vmax': 35, 'label': 'Radiation (MJ/m²/day)'}
    }
    
    params = var_params.get(variable, {'cmap': 'viridis', 'vmin': None, 'vmax': None, 'label': variable})
    
    for idx, day_idx in enumerate(sample_indices):
        ax = axes[idx]
        
        # Get data for this day
        data_day = ds[variable].isel(time=day_idx).values
        time_val = pd.Timestamp(ds['time'].isel(time=day_idx).values)
        
        # Plot
        im = ax.imshow(data_day, cmap=params['cmap'], 
                      vmin=params['vmin'], vmax=params['vmax'],
                      origin='lower', aspect='auto', interpolation='nearest')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label(params['label'], fontsize=9)
        
        # Statistics
        valid_count = np.sum(~np.isnan(data_day))
        if valid_count > 0:
            mean_val = np.nanmean(data_day)
            min_val = np.nanmin(data_day)
            max_val = np.nanmax(data_day)
            ax.set_title(f'{time_val.strftime("%b %d, %Y")}\n'
                        f'{valid_count:,} cells | μ={mean_val:.1f} | [{min_val:.1f}, {max_val:.1f}]',
                        fontsize=10, fontweight='bold')
        else:
            ax.set_title(f'{time_val.strftime("%b %d, %Y")}\nNo data',
                        fontsize=10)
        
        ax.set_xlabel('Grid X', fontsize=8)
        ax.set_ylabel('Grid Y', fontsize=8)
        ax.tick_params(labelsize=7)
    
    plt.suptitle(f'Spatial CIMIS {variable} - Sample Days from {year}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plot_file = output_path + f'spatial_{variable.lower()}_sample_days.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved to: {plot_file}")
    
    ds.close()


def plot_station_map(config):
    """
    Plot map of station locations.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    """
    print("\n3. Plotting station locations...")
    
    station_list_file = config.get('station_list_file', 'CIMIS_Stations.csv')
    output_path = config.get('output_path', 'output/')
    
    if not os.path.exists(station_list_file):
        print(f"  Warning: Station list file not found: {station_list_file}")
        return
    
    # Load station list
    station_list = pd.read_csv(station_list_file)
    
    # Clean up columns if needed
    if 'HmsLatitude' in station_list.columns:
        station_list['HmsLatitude'] = station_list['HmsLatitude'].str.split('/ ').str[-1]
        station_list['HmsLongitude'] = station_list['HmsLongitude'].str.split('/ ').str[-1]
        station_list.columns = ['Latitude' if x=='HmsLatitude' else x for x in station_list.columns]
        station_list.columns = ['Longitude' if x=='HmsLongitude' else x for x in station_list.columns]
    
    station_list['DisconnectDate'] = pd.to_datetime(station_list['DisconnectDate'])
    station_list['ConnectDate'] = pd.to_datetime(station_list['ConnectDate'])
    
    # Filter active stations
    start_date = config.get('start_date', '2004-01-01')
    end_date = config.get('end_date', '2024-01-01')
    
    active_stations = station_list.loc[
        (station_list['ConnectDate'] <= start_date) &
        (station_list['DisconnectDate'] > end_date)
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all stations in gray
    ax.scatter(station_list['Longitude'], station_list['Latitude'],
              c='lightgray', s=20, alpha=0.5, label='All Stations')
    
    # Plot active stations in color
    ax.scatter(active_stations['Longitude'], active_stations['Latitude'],
              c='red', s=50, alpha=0.8, label='Active Stations', edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'CIMIS Station Locations\n{len(active_stations)} active stations ({start_date} to {end_date})',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save
    plot_file = output_path + 'station_locations.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved to: {plot_file}")


def plot_time_series(config, n_stations=5):
    """
    Plot time series for a few sample stations.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    n_stations : int
        Number of stations to plot
    """
    print(f"\n4. Plotting time series for {n_stations} stations...")
    
    variable = config.get('variable', 'ETo')
    output_path = config.get('output_path', 'output/')
    
    # Load station data
    station_file = output_path + f'station_{variable.lower()}_data.csv'
    
    if not os.path.exists(station_file):
        print(f"  Warning: Station data file not found: {station_file}")
        return
    
    station_data = pd.read_csv(station_file, index_col=0, parse_dates=True)
    
    # Select a few stations with good coverage
    station_coverage = station_data.count()
    top_stations = station_coverage.nlargest(n_stations).index
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot each station
    for station in top_stations:
        station_data[station].plot(ax=ax, alpha=0.7, linewidth=0.5, label=f'Station {station}')
    
    ax.set_xlabel('Date', fontsize=12)
    
    var_labels = {
        'ETo': 'Reference ET (mm/day)',
        'Tx': 'Maximum Temperature (°C)',
        'Tn': 'Minimum Temperature (°C)',
        'Rs': 'Solar Radiation (MJ/m²/day)'
    }
    ax.set_ylabel(var_labels.get(variable, variable), fontsize=12)
    ax.set_title(f'CIMIS Station {variable} Time Series\n{n_stations} stations with best coverage',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Save
    plot_file = output_path + f'station_{variable.lower()}_timeseries.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved to: {plot_file}")


def main():
    """Main plotting workflow."""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'analysis_config.txt'
    
    print("="*70)
    print("Spatial CIMIS Plotting")
    print("="*70)
    
    # Load configuration
    config = load_config(config_file)
    
    # Create output directory
    output_path = config.get('output_path', '/home/salba/SpatialCIMIS/output/')
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\nVariable: {config.get('variable', 'ETo')}")
    print(f"Output path: {output_path}")
    
    try:
        # Create various plots
        plot_spatial_climatology(config)
        plot_sample_days(config, n_days=6)
        plot_station_map(config)
        plot_time_series(config, n_stations=5)
        
        print(f"\n{'='*70}")
        print("✓ All plots created successfully!")
        print(f"Output directory: {output_path}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n✗ Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()






