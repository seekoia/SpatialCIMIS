#!/usr/bin/env python3
"""
Plot a few days from /home/salba/SpatialCIMIS/netcdf/spatial_cimis_tx_2023.nc
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

print("="*70)
print("Plotting Sample Days from 2023 Tx Data")
print("="*70)

# Open the specific file
data_file = '/home/salba/SpatialCIMIS/netcdf/spatial_cimis_tx_2023.nc'
print(f"\nOpening: {data_file}")

ds = xr.open_dataset(data_file)

print(f"Variables: {list(ds.data_vars)}")
print(f"Dimensions: {dict(ds.dims)}")
print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

# Create figure with 6 subplots (2 rows x 3 cols)
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
axes = axes.flatten()

# Select 6 days throughout 2023
sample_days = [
    (14, "Jan 15"),      # Day 15 - mid winter
    (75, "Mar 17"),      # Day 76 - early spring  
    (135, "May 16"),     # Day 136 - late spring
    (195, "Jul 15"),     # Day 196 - mid summer (hottest)
    (255, "Sep 13"),     # Day 256 - early fall
    (320, "Nov 17")      # Day 321 - late fall
]

print(f"\nPlotting {len(sample_days)} sample days...")

for idx, (day_idx, day_label) in enumerate(sample_days):
    ax = axes[idx]
    
    # Get data for this day
    data_day = ds['Tx'].isel(time=day_idx).values
    time_val = ds['time'].isel(time=day_idx).values
    
    # Calculate statistics
    valid_count = np.sum(~np.isnan(data_day))
    
    if valid_count > 0:
        min_temp = np.nanmin(data_day)
        max_temp = np.nanmax(data_day)
        mean_temp = np.nanmean(data_day)
        
        # Plot using imshow
        im = ax.imshow(data_day, cmap='RdYlBu_r', vmin=-10, vmax=45,
                      origin='lower', aspect='auto', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('°C', fontsize=9)
        
        # Title with stats
        ax.set_title(f'{day_label}, 2023 (Day {day_idx+1})\n' + 
                   f'{valid_count:,} cells | Range: [{min_temp:.1f}, {max_temp:.1f}]°C | Mean: {mean_temp:.1f}°C',
                   fontsize=10, fontweight='bold')
        
        print(f"  {day_label}: {valid_count:,} cells | {min_temp:.1f} to {max_temp:.1f}°C | mean={mean_temp:.1f}°C")
    else:
        ax.text(0.5, 0.5, 'NO DATA', transform=ax.transAxes,
               ha='center', va='center', fontsize=18, color='red', fontweight='bold')
        ax.set_title(f'{day_label}, 2023 (Day {day_idx+1})\nNo valid data', fontsize=10)
        print(f"  {day_label}: NO DATA")
    
    ax.set_xlabel('Grid X', fontsize=8)
    ax.set_ylabel('Grid Y', fontsize=8)
    ax.tick_params(labelsize=7)

# Overall title
plt.suptitle('Spatial CIMIS Maximum Temperature (Tx)\nSample Days from 2023',
            fontsize=18, fontweight='bold')
plt.tight_layout()

# Save figure
output_file = '/home/salba/SpatialCIMIS/tx_2023_sample_days.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

# Print full year summary
print(f"\n{'='*70}")
print("2023 Full Year Summary:")
print(f"{'='*70}")

all_data = ds['Tx'].values
valid_all = all_data[~np.isnan(all_data)]

if len(valid_all) > 0:
    print(f"\nOverall Statistics:")
    print(f"  Total valid values: {len(valid_all):,}")
    print(f"  Temperature range: {valid_all.min():.2f}°C to {valid_all.max():.2f}°C")
    print(f"  Mean: {valid_all.mean():.2f}°C")
    print(f"  Median: {np.median(valid_all):.2f}°C")
    print(f"  Std dev: {valid_all.std():.2f}°C")
    
    # Per-day coverage
    valid_per_day = np.sum(~np.isnan(all_data), axis=(1,2))
    print(f"\nDaily Coverage:")
    print(f"  Min cells/day: {valid_per_day.min():,}")
    print(f"  Max cells/day: {valid_per_day.max():,}")
    print(f"  Mean cells/day: {valid_per_day.mean():.0f}")
    print(f"  Days with data: {np.sum(valid_per_day > 0)}/{len(valid_per_day)}")
    
    # Seasonal statistics
    print(f"\nSeasonal Averages:")
    winter_days = list(range(0, 60)) + list(range(335, 365))  # Dec-Feb
    spring_days = list(range(60, 152))   # Mar-May
    summer_days = list(range(152, 244))  # Jun-Aug
    fall_days = list(range(244, 335))    # Sep-Nov
    
    winter_mean = np.nanmean(all_data[winter_days])
    spring_mean = np.nanmean(all_data[spring_days])
    summer_mean = np.nanmean(all_data[summer_days])
    fall_mean = np.nanmean(all_data[fall_days])
    
    print(f"  Winter (Dec-Feb): {winter_mean:.1f}°C")
    print(f"  Spring (Mar-May): {spring_mean:.1f}°C")
    print(f"  Summer (Jun-Aug): {summer_mean:.1f}°C")
    print(f"  Fall (Sep-Nov): {fall_mean:.1f}°C")
else:
    print("\n✗ ERROR: No valid data in file!")

ds.close()

print(f"\n{'='*70}")
print("✓ Complete! Download the PNG file to view it.")
print(f"File: {output_file}")
print(f"{'='*70}")







