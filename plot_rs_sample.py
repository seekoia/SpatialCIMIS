#!/usr/bin/env python3
"""
Quick plot of Rs data to verify it's working
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# Open the 2023 file
ds = xr.open_dataset('/group/moniergrp/SpatialCIMIS/netcdf/test/spatial_cimis_rs_2019.nc')

print("="*60)
print("Spatial CIMIS Rs (Solar Radiation) Data Visualization")
print("="*60)

print(f"\nVariables in file: {list(ds.data_vars)}")
print(f"Dataset dimensions: {dict(ds.dims)}")

# Get coordinates
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
    data_day = ds['Rs'].isel(time=day_idx).values
    
    # Plot
    im = ax.imshow(data_day, cmap='plasma', vmin=0, vmax=35,
                   origin='lower', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Solar Radiation (MJ/m²/day)', fontsize=9)
    
    # Stats
    valid_count = np.sum(~np.isnan(data_day))
    if valid_count > 0:
        mean_val = np.nanmean(data_day)
        min_val = np.nanmin(data_day)
        max_val = np.nanmax(data_day)
        ax.set_title(f'{month_name}, 2023\n{valid_count:,} cells | mean: {mean_val:.1f} MJ/m²/day | [{min_val:.1f}, {max_val:.1f}]', 
                    fontsize=10, fontweight='bold')
    else:
        ax.set_title(f'{month_name}, 2023\nNo data', fontsize=10)
    
    ax.set_xlabel('Grid X', fontsize=9)
    ax.set_ylabel('Grid Y', fontsize=9)
    ax.tick_params(labelsize=8)

plt.suptitle('Spatial CIMIS Solar Radiation (Rs) - Sample Days from 2023', 
            fontsize=16, fontweight='bold')
plt.tight_layout()

# Save figure
output_file = '/home/salba/SpatialCIMIS/rs_sample_days.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

# Also create a mask visualization
fig2, ax2 = plt.subplots(figsize=(12, 10))

mask_ds = xr.open_dataset('/group/moniergrp/SpatialCIMIS/netcdf/spatial_cimis_rs_mask.nc')
mask = mask_ds['mask'].values

# Plot mask
im = ax2.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1,
               origin='lower', aspect='auto', interpolation='nearest')
cbar2 = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar2.set_label('Mask (1=data, 0=no data)', fontsize=11)

valid_cells = np.sum(mask == 1)
coverage = valid_cells / mask.size * 100
ax2.set_title(f'Spatial CIMIS Rs Mask\n{valid_cells:,} cells with valid data ({coverage:.1f}% coverage)', 
             fontsize=14, fontweight='bold')
ax2.set_xlabel('Grid X', fontsize=11)
ax2.set_ylabel('Grid Y', fontsize=11)
ax2.tick_params(labelsize=9)

mask_file = '/home/salba/SpatialCIMIS/rs_mask_visualization.png'
plt.savefig(mask_file, dpi=150, bbox_inches='tight')
print(f"✓ Mask plot saved to: {mask_file}")

# Print statistics
print("\n" + "="*60)
print("2023 Rs Data Summary:")
print("="*60)

all_data = ds['Rs'].values
valid_all = all_data[~np.isnan(all_data)]

if len(valid_all) > 0:
    print(f"\nOverall Statistics:")
    print(f"  Total valid values: {len(valid_all):,}")
    print(f"  Solar radiation range: {valid_all.min():.2f} to {valid_all.max():.2f} MJ/m²/day")
    print(f"  Mean: {valid_all.mean():.2f} MJ/m²/day")
    print(f"  Median: {np.median(valid_all):.2f} MJ/m²/day")
    print(f"  Std dev: {valid_all.std():.2f} MJ/m²/day")
    
    # Per-day coverage
    valid_per_day = np.sum(~np.isnan(all_data), axis=(1,2))
    print(f"\nDaily Coverage:")
    print(f"  Min cells/day: {valid_per_day.min():,}")
    print(f"  Max cells/day: {valid_per_day.max():,}")
    print(f"  Mean cells/day: {valid_per_day.mean():.0f}")
    print(f"  Days with full coverage: {np.sum(valid_per_day == valid_per_day.max())}/{len(valid_per_day)}")
    
    # Seasonal averages
    print(f"\nSeasonal Averages:")
    winter_days = list(range(0, 60)) + list(range(335, 365))  # Dec-Feb
    spring_days = list(range(60, 152))   # Mar-May
    summer_days = list(range(152, 244))  # Jun-Aug
    fall_days = list(range(244, 335))    # Sep-Nov
    
    winter_mean = np.nanmean(all_data[winter_days])
    spring_mean = np.nanmean(all_data[spring_days])
    summer_mean = np.nanmean(all_data[summer_days])
    fall_mean = np.nanmean(all_data[fall_days])
    
    print(f"  Winter (Dec-Feb): {winter_mean:.1f} MJ/m²/day")
    print(f"  Spring (Mar-May): {spring_mean:.1f} MJ/m²/day")
    print(f"  Summer (Jun-Aug): {summer_mean:.1f} MJ/m²/day")
    print(f"  Fall (Sep-Nov): {fall_mean:.1f} MJ/m²/day")
else:
    print("\n✗ ERROR: No valid data in file!")

ds.close()
mask_ds.close()

print("\n" + "="*60)
print("Visualization complete!")
print("Download the PNG files to view them on your local machine:")
print(f"  - {output_file}")
print(f"  - {mask_file}")
print("="*60)





