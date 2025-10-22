#!/usr/bin/env python3
"""
Plot a few sample days from Tx NetCDF data
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys

# Get year from command line or use default
if len(sys.argv) > 1:
    year = int(sys.argv[1])
else:
    year = 2023

# Paths
netcdf_path = '/home/salba/SpatialCIMIS/netcdf/'
output_path = '/home/salba/SpatialCIMIS/'

print("="*70)
print(f"Plotting Sample Days from {year} Tx Data")
print("="*70)

# Open the data file
data_file = netcdf_path + f'spatial_cimis_tx_{year}.nc'
print(f"\nOpening: {data_file}")

try:
    ds = xr.open_dataset(data_file)
    
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Time steps: {len(ds.time)}")
    
    # Get coordinates
    y = ds['y'].values
    x = ds['x'].values
    
    # Create figure with 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Select sample days throughout the year
    # Day of year: 1, 60, 120, 180, 240, 300
    sample_days = [
        (0, "Jan 1"),
        (59, "Mar 1"),
        (119, "May 1"), 
        (179, "Jul 1"),
        (239, "Sep 1"),
        (299, "Nov 1")
    ]
    
    print(f"\nPlotting {len(sample_days)} sample days...")
    
    for idx, (day_idx, day_label) in enumerate(sample_days):
        ax = axes[idx]
        
        # Get data for this day
        if day_idx < len(ds.time):
            data_day = ds['Tx'].isel(time=day_idx).values
            time_val = ds['time'].isel(time=day_idx).values
            
            # Calculate statistics
            valid_count = np.sum(~np.isnan(data_day))
            
            if valid_count > 0:
                min_temp = np.nanmin(data_day)
                max_temp = np.nanmax(data_day)
                mean_temp = np.nanmean(data_day)
                
                # Plot using imshow (simpler than pcolormesh)
                im = ax.imshow(data_day, cmap='RdYlBu_r', vmin=-10, vmax=45,
                              origin='lower', aspect='auto', interpolation='nearest')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Temp (°C)', fontsize=9)
                
                # Title with stats
                ax.set_title(f'{day_label}, {year}\n' + 
                           f'{valid_count:,} cells | ' +
                           f'{min_temp:.1f}°C to {max_temp:.1f}°C | ' +
                           f'μ={mean_temp:.1f}°C',
                           fontsize=10, fontweight='bold')
                
                print(f"  {day_label}: {valid_count:,} cells, {mean_temp:.1f}°C mean")
            else:
                ax.text(0.5, 0.5, 'NO DATA', transform=ax.transAxes,
                       ha='center', va='center', fontsize=16, color='red')
                ax.set_title(f'{day_label}, {year}\nNo data', fontsize=10)
                print(f"  {day_label}: NO DATA")
            
            ax.set_xlabel('X Index', fontsize=8)
            ax.set_ylabel('Y Index', fontsize=8)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Overall title
    plt.suptitle(f'Spatial CIMIS Maximum Temperature (Tx) - Sample Days from {year}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = output_path + f'tx_{year}_sample_days.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    
    # Print annual summary
    print(f"\n{'='*70}")
    print(f"{year} Annual Summary:")
    print(f"{'='*70}")
    
    all_data = ds['Tx'].values
    valid_all = all_data[~np.isnan(all_data)]
    
    if len(valid_all) > 0:
        print(f"Total valid values: {len(valid_all):,}")
        print(f"Temperature range: {valid_all.min():.2f}°C to {valid_all.max():.2f}°C")
        print(f"Mean temperature: {valid_all.mean():.2f}°C")
        print(f"Median temperature: {np.median(valid_all):.2f}°C")
        
        # Per-day coverage
        valid_per_day = np.sum(~np.isnan(all_data), axis=(1,2))
        print(f"\nDaily coverage:")
        print(f"  Min: {valid_per_day.min():,} cells")
        print(f"  Max: {valid_per_day.max():,} cells")
        print(f"  Mean: {valid_per_day.mean():.0f} cells")
        print(f"  Days with data: {np.sum(valid_per_day > 0)}/{len(valid_per_day)}")
    
    ds.close()
    
    print(f"\n{'='*70}")
    print("✓ Visualization complete!")
    print(f"Download {output_file} to view it")
    print(f"{'='*70}")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)







