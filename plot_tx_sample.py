#!/usr/bin/env python3
"""
Quick plot of Tx data to verify it's working
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# Open the 2023 file from local netcdf directory
ds = xr.open_dataset('/home/salba/SpatialCIMIS/netcdf/spatial_cimis_tx_2023.nc')

print("="*60)
print("Spatial CIMIS Tx Data Visualization")
print("="*60)

print(f"\nVariables in file: {list(ds.data_vars)}")
print(f"Dataset dimensions: {dict(ds.dims)}")

# Get lat/lon for plotting
# Use x/y coordinates instead of lat/lon to avoid NaN issues
y_coords = ds['y'].values
x_coords = ds['x'].values
# Create meshgrid
x_grid, y_grid = np.meshgrid(x_coords, y_coords)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

# Sample days: Jan 1, Apr 1, Jul 1, Oct 1
sample_days = [0, 90, 180, 270]
month_names = ['January 1', 'April 1', 'July 1', 'October 1']

for idx, (day_idx, month_name) in enumerate(zip(sample_days, month_names)):
    ax = axes[idx]
    
    # Get data for this day
    data_day = ds['Tx'].isel(time=day_idx).values
    
    # Plot
    pcm = ax.pcolormesh(x_grid, y_grid, data_day, cmap='RdYlBu_r', 
                       shading='auto', vmin=-5, vmax=40)
    
    # Add colorbar
    cbar = plt.colorbar(pcm, ax=ax, label='Temperature (°C)')
    
    # Stats
    valid_count = np.sum(~np.isnan(data_day))
    if valid_count > 0:
        mean_temp = np.nanmean(data_day)
        ax.set_title(f'{month_name}, 2023\n{valid_count:,} cells, mean: {mean_temp:.1f}°C', 
                    fontsize=11, fontweight='bold')
    else:
        ax.set_title(f'{month_name}, 2023\nNo data', fontsize=11)
    
    ax.set_xlabel('X Coordinate (m)', fontsize=10)
    ax.set_ylabel('Y Coordinate (m)', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.suptitle('Spatial CIMIS Maximum Temperature (Tx) - Sample Days from 2023', 
            fontsize=16, fontweight='bold')
plt.tight_layout()

# Save figure
output_file = '/home/salba/SpatialCIMIS/tx_sample_days.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

# Also create a mask visualization
fig2, ax2 = plt.subplots(figsize=(12, 10))

mask_ds = xr.open_dataset('/home/salba/SpatialCIMIS/netcdf/spatial_cimis_tx_mask.nc')
mask = mask_ds['mask'].values

# Plot mask
im = ax2.pcolormesh(x_grid, y_grid, mask, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
cbar2 = plt.colorbar(im, ax=ax2, label='Mask (1=data, 0=no data)')

valid_cells = np.sum(mask == 1)
ax2.set_title(f'Spatial CIMIS Tx Mask\n{valid_cells:,} cells with valid data (37.2% coverage)', 
             fontsize=14, fontweight='bold')
ax2.set_xlabel('X Coordinate (m)', fontsize=12)
ax2.set_ylabel('Y Coordinate (m)', fontsize=12)
ax2.grid(True, alpha=0.3)

mask_file = '/home/salba/SpatialCIMIS/tx_mask_visualization.png'
plt.savefig(mask_file, dpi=150, bbox_inches='tight')
print(f"✓ Mask plot saved to: {mask_file}")

print("\n" + "="*60)
print("Visualization complete!")
print("Download the PNG files to view them on your local machine.")
print("="*60)

ds.close()
mask_ds.close()

