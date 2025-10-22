#!/usr/bin/env python3
"""
Plot a few days from 2004 Tx data
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

print("Loading 2004 Tx data...")

# Open the 2004 file
ds = xr.open_dataset('/group/moniergrp/SpatialCIMIS/netcdf/spatial_cimis_tx_2004.nc')

print(f"Variables: {list(ds.data_vars)}")
print(f"Dimensions: {dict(ds.dims)}")
print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

# Get coordinates
y = ds['y'].values
x = ds['x'].values

# Create figure with 6 subplots (2 rows x 3 cols)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Sample days throughout the year
sample_days = [
    (0, "Jan 1, 2004"),
    (31, "Feb 1, 2004"),
    (120, "May 1, 2004"),
    (180, "Jul 1, 2004"),
    (270, "Oct 1, 2004"),
    (330, "Dec 1, 2004")
]

for idx, (day_idx, title) in enumerate(sample_days):
    ax = axes[idx]
    
    # Get data for this day
    data_day = ds['Tx'].isel(time=day_idx).values
    
    # Create the plot using imshow for simplicity
    im = ax.imshow(data_day, cmap='RdYlBu_r', vmin=-5, vmax=40, 
                   origin='lower', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Temperature (°C)', fontsize=9)
    
    # Add statistics
    valid_count = np.sum(~np.isnan(data_day))
    if valid_count > 0:
        min_temp = np.nanmin(data_day)
        max_temp = np.nanmax(data_day)
        mean_temp = np.nanmean(data_day)
        ax.set_title(f'{title}\n{valid_count:,} cells | Min: {min_temp:.1f}°C | Max: {max_temp:.1f}°C | Mean: {mean_temp:.1f}°C', 
                    fontsize=10, fontweight='bold')
    else:
        ax.set_title(f'{title}\nNo data', fontsize=10)
    
    ax.set_xlabel('X Index', fontsize=9)
    ax.set_ylabel('Y Index', fontsize=9)

plt.suptitle('Spatial CIMIS Maximum Temperature (Tx) - 2004 Sample Days', 
            fontsize=16, fontweight='bold')
plt.tight_layout()

# Save figure
output_file = '/home/salba/SpatialCIMIS/tx_2004_six_days.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

# Print summary statistics for the entire year
print(f"\n{'='*60}")
print("2004 Annual Statistics:")
print(f"{'='*60}")

all_data = ds['Tx'].values
valid_data = all_data[~np.isnan(all_data)]

if len(valid_data) > 0:
    print(f"Total valid values: {len(valid_data):,}")
    print(f"Temperature range: {valid_data.min():.2f}°C to {valid_data.max():.2f}°C")
    print(f"Mean temperature: {valid_data.mean():.2f}°C")
    print(f"Std deviation: {valid_data.std():.2f}°C")
    
    # Check coverage per day
    valid_per_day = np.sum(~np.isnan(all_data), axis=(1,2))
    print(f"\nCoverage statistics:")
    print(f"  Min cells per day: {valid_per_day.min():,}")
    print(f"  Max cells per day: {valid_per_day.max():,}")
    print(f"  Mean cells per day: {valid_per_day.mean():.0f}")
    print(f"  Days with full coverage (102,572 cells): {np.sum(valid_per_day == 102572)}")
else:
    print("ERROR: No valid data found!")

ds.close()

print(f"\n{'='*60}")
print("Done! Download the PNG file to view it.")
print(f"{'='*60}")







