#!/usr/bin/env python3
"""
Script to check and visualize Spatial CIMIS processing results
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import os

# Paths
netcdf_path = '/group/moniergrp/SpatialCIMIS/netcdf/'
output_path = '/home/salba/SpatialCIMIS/results_check/'

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Variable to check
variable = 'Tx'

print("="*60)
print("Spatial CIMIS Results Check")
print("="*60)

# 1. Check the mask
print("\n1. Checking mask file...")
mask_file = netcdf_path + f'spatial_cimis_{variable.lower()}_mask.nc'

try:
    ds_mask = xr.open_dataset(mask_file)
    mask = ds_mask['mask'].values
    lat = ds_mask['lat'].values
    lon = ds_mask['lon'].values
    
    valid_cells = np.sum(mask == 1)
    invalid_cells = np.sum(mask == 0)
    
    print(f"   Mask file: {mask_file}")
    print(f"   Grid shape: {mask.shape}")
    print(f"   Valid cells (mask=1): {valid_cells:,}")
    print(f"   Invalid cells (mask=0): {invalid_cells:,}")
    print(f"   Coverage: {valid_cells/(valid_cells+invalid_cells)*100:.1f}%")
    
    # Plot the mask
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use pcolormesh with lat/lon
    pcm = ax.pcolormesh(lon, lat, mask, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
    
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Mask (1=data, 0=no data)', fontsize=12)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Spatial CIMIS {variable} Mask\n{valid_cells:,} cells with valid data', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    mask_plot_file = output_path + f'{variable.lower()}_mask.png'
    plt.savefig(mask_plot_file, dpi=150, bbox_inches='tight')
    print(f"   Mask plot saved: {mask_plot_file}")
    plt.close()
    
except Exception as e:
    print(f"   Error reading mask: {e}")
    mask = None

print("\n2. Checking data files...")

# Check available years
import glob
data_files = sorted(glob.glob(netcdf_path + f'spatial_cimis_{variable.lower()}_20*.nc'))
print(f"   Found {len(data_files)} data files")

if data_files:
    # Check first few files
    for data_file in data_files[:3]:
        year = data_file.split('_')[-1].replace('.nc', '')
        print(f"\n   Checking year {year}...")
        
        try:
            ds_data = xr.open_dataset(data_file)
            
            # Print dataset info
            # Get actual variable name from file (case-sensitive)
            var_name = variable if variable in ds_data.data_vars else variable.upper()
            
            print(f"      Dataset shape: {ds_data[var_name].shape}")
            print(f"      Variables: {list(ds_data.data_vars)}")
            
            # Check a few days
            days_to_check = [0, 100, 200]
            
            for day_idx in days_to_check:
                if day_idx < len(ds_data['time']):
                    data_day = ds_data[var_name].isel(time=day_idx).values
                    time_val = ds_data['time'].isel(time=day_idx).values
                    
                    valid_count = np.sum(~np.isnan(data_day))
                    
                    if valid_count > 0:
                        min_val = np.nanmin(data_day)
                        max_val = np.nanmax(data_day)
                        mean_val = np.nanmean(data_day)
                        print(f"      Day {day_idx} ({time_val}): {valid_count:,} valid values, "
                              f"range [{min_val:.1f}, {max_val:.1f}], mean {mean_val:.1f}")
                    else:
                        print(f"      Day {day_idx} ({time_val}): ALL NaN values!")
            
            # Plot a few sample days
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            
            sample_days = [0, 100, 200, 300]
            
            for idx, day_idx in enumerate(sample_days):
                if day_idx < len(ds_data['time']):
                    ax = axes[idx]
                    
                    data_day = ds_data[var_name].isel(time=day_idx).values
                    time_val = pd.Timestamp(ds_data['time'].isel(time=day_idx).values)
                    
                    # Plot
                    if np.sum(~np.isnan(data_day)) > 0:
                        pcm = ax.pcolormesh(lon, lat, data_day, cmap='RdYlBu_r', 
                                           shading='auto', vmin=-10, vmax=45)
                        plt.colorbar(pcm, ax=ax, label='Temperature (°C)')
                        
                        valid = np.sum(~np.isnan(data_day))
                        ax.set_title(f'{time_val.strftime("%Y-%m-%d")}\n'
                                    f'{valid:,} valid values, '
                                    f'mean: {np.nanmean(data_day):.1f}°C',
                                    fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'NO DATA', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=20)
                        ax.set_title(f'{time_val.strftime("%Y-%m-%d")}\nNo valid data',
                                    fontsize=10)
                    
                    ax.set_xlabel('Longitude', fontsize=10)
                    ax.set_ylabel('Latitude', fontsize=10)
                    ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'Spatial CIMIS {variable} - Sample Days from {year}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            data_plot_file = output_path + f'{variable.lower()}_{year}_samples.png'
            plt.savefig(data_plot_file, dpi=150, bbox_inches='tight')
            print(f"      Sample days plot saved: {data_plot_file}")
            plt.close()
            
            ds_data.close()
            
        except Exception as e:
            print(f"      Error reading {year}: {e}")

else:
    print("   No data files found!")

print("\n" + "="*60)
print("Results check complete!")
print(f"Output saved to: {output_path}")
print("="*60)

# Import pandas for timestamp formatting
import pandas as pd

