#!/usr/bin/env python3
"""
Plot Spatial CIMIS ETo Data Reprojected to GridMET Grid

This script plots a few sample days from the GridMET reprojected Spatial CIMIS ETo data
to visualize the reprojection results and data quality.

Author: UC Davis Global Environmental Change Lab
Date: 2024
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import os

def plot_eto_days(nc_file, output_dir='./plots', days_to_plot=[0, 30, 60, 90, 120, 150]):
    """
    Plot ETo data for selected days from the GridMET reprojected file.
    
    Parameters:
    -----------
    nc_file : str
        Path to the NetCDF file
    output_dir : str
        Directory to save plots
    days_to_plot : list
        List of day indices to plot (0-based)
    """
    
    print(f"Loading data from: {nc_file}")
    
    # Load the dataset
    ds = xr.open_dataset(nc_file)
    
    print(f"Dataset dimensions: {ds.dims}")
    print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
    print(f"Latitude range: {ds.lat.min().values:.3f} to {ds.lat.max().values:.3f}")
    print(f"Longitude range: {ds.lon.min().values:.3f} to {ds.lon.max().values:.3f}")
    
    # Check latitude orientation
    if ds.lat[0] > ds.lat[-1]:
        print("⚠️  Latitude appears to be inverted (descending order)")
    else:
        print("✓ Latitude is in ascending order")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert time to dates for better labeling
    # Use xarray's built-in time conversion which handles CF conventions properly
    try:
        # Convert time coordinate to datetime
        ds_time = ds.time.to_pandas()
        print(f"Time conversion successful. First date: {ds_time.iloc[0]}")
    except Exception as e:
        print(f"Time conversion failed: {e}")
        # Fallback: create simple day numbers
        ds_time = [f"Day {i+1}" for i in range(len(ds.time))]
    
    # Create figure with subplots
    n_days = len(days_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                           subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    for i, day_idx in enumerate(days_to_plot):
        if day_idx >= len(ds.time):
            print(f"Warning: Day index {day_idx} exceeds available days ({len(ds.time)})")
            continue
            
        # Get the data for this day
        eto_data = ds.ETo.isel(time=day_idx)
        
        # Calculate the actual date
        if hasattr(ds_time, 'iloc'):
            # If ds_time is a pandas Series
            plot_date = ds_time.iloc[day_idx]
            if hasattr(plot_date, 'strftime'):
                date_str = plot_date.strftime("%Y-%m-%d")
            else:
                date_str = str(plot_date)
        else:
            # If ds_time is a list of strings
            date_str = ds_time[day_idx]
        
        # Create the plot
        ax = axes[i]
        
        # Plot the data
        im = ax.pcolormesh(ds.lon, ds.lat, eto_data, 
                          transform=ccrs.PlateCarree(),
                          cmap='viridis', 
                          vmin=0, vmax=8,
                          shading='auto')
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        
        # Set extent to California
        ax.set_extent([-125, -114, 32, 42], crs=ccrs.PlateCarree())
        
        # Add title
        ax.set_title(f'ETo - {date_str} (Day {day_idx+1})', 
                    fontsize=12, fontweight='bold')
        
        # Add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    
    # Remove empty subplots
    for i in range(n_days, len(axes)):
        fig.delaxes(axes[i])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[:n_days], orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('Reference Evapotranspiration (mm/day)', fontsize=12)
    
    # Add main title
    fig.suptitle('Spatial CIMIS ETo Data - GridMET Grid Reprojection\nSample Days 2023', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the plot
    output_file = os.path.join(output_dir, 'spatial_cimis_eto_gridmet_sample_days.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()
    
    # Create a summary statistics plot
    plot_summary_stats(ds, output_dir)
    
    return ds

def plot_summary_stats(ds, output_dir):
    """
    Create summary statistics plots for the dataset.
    """
    
    print("\nCreating summary statistics plots...")
    
    # Calculate statistics
    eto_mean = ds.ETo.mean(dim='time')
    eto_max = ds.ETo.max(dim='time')
    eto_min = ds.ETo.min(dim='time')
    eto_std = ds.ETo.std(dim='time')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    stats_data = [eto_mean, eto_max, eto_min, eto_std]
    stats_titles = ['Annual Mean ETo', 'Annual Maximum ETo', 
                   'Annual Minimum ETo', 'Annual Std Dev ETo']
    stats_labels = ['Mean ETo (mm/day)', 'Max ETo (mm/day)', 
                   'Min ETo (mm/day)', 'Std Dev ETo (mm/day)']
    
    for i, (data, title, label) in enumerate(zip(stats_data, stats_titles, stats_labels)):
        ax = axes[i]
        
        # Plot the data
        im = ax.pcolormesh(ds.lon, ds.lat, data,
                          transform=ccrs.PlateCarree(),
                          cmap='viridis',
                          shading='auto')
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        
        # Set extent to California
        ax.set_extent([-125, -114, 32, 42], crs=ccrs.PlateCarree())
        
        # Add title
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        cbar.set_label(label, fontsize=10)
    
    # Add main title
    fig.suptitle('Spatial CIMIS ETo Statistics - GridMET Grid (2023)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the plot
    output_file = os.path.join(output_dir, 'spatial_cimis_eto_gridmet_statistics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Statistics plot saved to: {output_file}")
    
    plt.show()
    
    # Print some basic statistics
    print(f"\nDataset Statistics:")
    print(f"  Mean ETo: {eto_mean.mean().values:.2f} mm/day")
    print(f"  Max ETo: {eto_max.max().values:.2f} mm/day")
    print(f"  Min ETo: {eto_min.min().values:.2f} mm/day")
    print(f"  Std Dev: {eto_std.mean().values:.2f} mm/day")

def main():
    """Main function to run the plotting script."""
    
    # Path to the GridMET reprojected file
    nc_file = "/group/moniergrp/SpatialCIMIS/netcdf/test/spatial_cimis_eto_2023_gridmet.nc"
    
    # Check if file exists
    if not os.path.exists(nc_file):
        print(f"Error: File not found: {nc_file}")
        print("Please check the file path and ensure the GridMET processing completed successfully.")
        return
    
    print("="*60)
    print("Spatial CIMIS ETo GridMET Grid Visualization")
    print("="*60)
    
    # Plot sample days
    ds = plot_eto_days(nc_file, 
                      output_dir='./plots',
                      days_to_plot=[0, 30, 60, 90, 120, 150])  # Every ~30 days
    
    print("\n" + "="*60)
    print("Plotting completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
