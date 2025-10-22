#!/usr/bin/env python3
"""
Script to plot one day from the Tx NetCDF files to check performance
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

# Open one of the NetCDF files
netcdf_path = '/group/moniergrp/SpatialCIMIS/netcdf/'
nc_file = netcdf_path + 'spatial_cimis_tx_2023.nc'

print(f"Opening file: {nc_file}")
ds = xr.open_dataset(nc_file)

print("\nDataset information:")
print(ds)

# Select one day (e.g., July 15, 2023 - typically a hot day in California)
day_index = 195  # Approximately July 15
tx_data = ds['TX'].isel(time=day_index)

# Get the date
time_val = ds['time'].isel(time=day_index).values
print(f"\nPlotting data for time index {day_index}: {time_val}")

# Get coordinates
lat = ds['lat'].values
lon = ds['lon'].values
tx_values = tx_data.values

print(f"\nData statistics:")
print(f"  Min Tx: {np.nanmin(tx_values):.2f}°C")
print(f"  Max Tx: {np.nanmax(tx_values):.2f}°C")
print(f"  Mean Tx: {np.nanmean(tx_values):.2f}°C")
print(f"  Number of valid values: {np.sum(~np.isnan(tx_values))}")
print(f"  Number of NaN values: {np.sum(np.isnan(tx_values))}")

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10))

# Create a filled contour plot
levels = np.arange(0, 50, 2)  # Temperature levels from 0 to 50°C
cf = ax.contourf(lon, lat, tx_values, levels=levels, cmap='RdYlBu_r', extend='both')

# Add colorbar
cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
cbar.set_label('Maximum Temperature (°C)', fontsize=12)

# Add coastlines or state boundaries if possible
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title(f'Spatial CIMIS Maximum Temperature (Tx)\nDay {day_index} of 2023', fontsize=14, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3)

# Save the figure
output_file = '/home/salba/SpatialCIMIS/tx_check_plot.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Also create a simpler plot with pcolormesh
fig2, ax2 = plt.subplots(figsize=(12, 10))

# Use pcolormesh for faster rendering
pcm = ax2.pcolormesh(lon, lat, tx_values, cmap='RdYlBu_r', shading='auto', vmin=0, vmax=45)

# Add colorbar
cbar2 = plt.colorbar(pcm, ax=ax2, orientation='vertical', pad=0.02, shrink=0.8)
cbar2.set_label('Maximum Temperature (°C)', fontsize=12)

ax2.set_xlabel('Longitude', fontsize=12)
ax2.set_ylabel('Latitude', fontsize=12)
ax2.set_title(f'Spatial CIMIS Maximum Temperature (Tx)\nDay {day_index} of 2023 (pcolormesh)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Save the second figure
output_file2 = '/home/salba/SpatialCIMIS/tx_check_plot_pcolormesh.png'
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"Alternative plot saved to: {output_file2}")

# Close the dataset
ds.close()

print("\nDone! You can download the PNG files to view them locally.")

