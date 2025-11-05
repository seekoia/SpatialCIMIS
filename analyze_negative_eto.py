#!/usr/bin/env python3
"""
Analyze Negative ETo Days in Spatial CIMIS Data

This script identifies days when Spatial CIMIS ETo data has negative values
and plots those days from the GridMET reprojected data to investigate
the spatial patterns and potential causes.

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
import pandas as pd

def find_negative_eto_days(nc_file, threshold=-0.1):
    """
    Find days when Spatial CIMIS ETo data has negative values.
    
    Parameters:
    -----------
    nc_file : str
        Path to the GridMET reprojected NetCDF file
    threshold : float
        Threshold for negative values (default: -0.1 mm/day)
        
    Returns:
    --------
    list : List of day indices with negative ETo values
    """
    
    print(f"Loading data from: {nc_file}")
    ds = xr.open_dataset(nc_file)
    
    print(f"Dataset dimensions: {ds.dims}")
    print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
    print(f"Latitude range: {ds.lat.min().values:.3f} to {ds.lat.max().values:.3f}")
    print(f"Longitude range: {ds.lon.min().values:.3f} to {ds.lon.max().values:.3f}")
    
    # Find days with negative ETo values
    print(f"\nSearching for days with ETo < {threshold} mm/day...")
    
    # Calculate daily minimum ETo across all grid cells
    daily_min_eto = ds.ETo.min(dim=['lat', 'lon'])
    
    # Find days with negative values
    negative_days = np.where(daily_min_eto.values < threshold)[0]
    
    print(f"Found {len(negative_days)} days with ETo < {threshold} mm/day")
    
    if len(negative_days) > 0:
        print(f"Day indices: {negative_days[:10]}{'...' if len(negative_days) > 10 else ''}")
        print(f"Minimum ETo values: {daily_min_eto.values[negative_days[:10]]}")
    
    return negative_days, daily_min_eto

def plot_negative_eto_days(nc_file, negative_days, daily_min_eto, output_dir='./plots', max_days=6):
    """
    Plot days with negative ETo values.
    
    Parameters:
    -----------
    nc_file : str
        Path to the NetCDF file
    negative_days : array
        Array of day indices with negative ETo
    daily_min_eto : xarray.DataArray
        Daily minimum ETo values
    output_dir : str
        Output directory for plots
    max_days : int
        Maximum number of days to plot
    """
    
    print(f"\nPlotting up to {max_days} days with negative ETo values...")
    
    # Load the dataset
    ds = xr.open_dataset(nc_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert time to dates for better labeling
    try:
        # Convert time coordinate to datetime
        ds_time = ds.time.to_pandas()
        print(f"Time conversion successful. First date: {ds_time.iloc[0]}")
    except Exception as e:
        print(f"Time conversion failed: {e}")
        # Fallback: create simple day numbers
        ds_time = [f"Day {i+1}" for i in range(len(ds.time))]
    
    # Select days to plot (up to max_days)
    days_to_plot = negative_days[:max_days]
    
    # Create figure with subplots
    n_days = len(days_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                           subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    for i, day_idx in enumerate(days_to_plot):
        if i >= len(axes):
            break
            
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
        
        # Plot the data with a symmetric color scale around zero
        vmin = min(-5, eto_data.min().values)
        vmax = max(5, eto_data.max().values)
        
        im = ax.pcolormesh(ds.lon, ds.lat, eto_data, 
                          transform=ccrs.PlateCarree(),
                          cmap='RdBu_r',  # Red-Blue reversed for better negative values
                          vmin=vmin, vmax=vmax,
                          shading='auto')
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        
        # Set extent to California
        ax.set_extent([-125, -114, 32, 42], crs=ccrs.PlateCarree())
        
        # Add title with min ETo value
        min_eto = daily_min_eto.isel(time=day_idx).values
        ax.set_title(f'{date_str} (Day {day_idx+1})\nMin ETo: {min_eto:.2f} mm/day', 
                    fontsize=11, fontweight='bold')
        
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
    cbar.set_label('ETo (mm/day)', fontsize=12)
    
    # Add main title
    fig.suptitle('Spatial CIMIS ETo Days with Negative Values (2023)\nGridMET Reprojected Data', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the plot
    output_file = os.path.join(output_dir, 'spatial_cimis_negative_eto_days.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Negative ETo days plot saved to: {output_file}")
    
    plt.show()

def analyze_negative_patterns(nc_file, negative_days, daily_min_eto):
    """
    Analyze patterns in negative ETo days.
    
    Parameters:
    -----------
    nc_file : str
        Path to the NetCDF file
    negative_days : array
        Array of day indices with negative ETo
    daily_min_eto : xarray.DataArray
        Daily minimum ETo values
    """
    
    print(f"\nAnalyzing patterns in negative ETo days...")
    
    # Load the dataset
    ds = xr.open_dataset(nc_file)
    
    # Convert time to dates
    try:
        ds_time = ds.time.to_pandas()
        dates = ds_time.iloc[negative_days]
    except:
        dates = [f"Day {i+1}" for i in negative_days]
    
    # Analyze temporal patterns
    print(f"\nTemporal Analysis:")
    print(f"  Total days with negative ETo: {len(negative_days)}")
    print(f"  Percentage of year: {len(negative_days)/365*100:.1f}%")
    
    # Analyze spatial patterns
    print(f"\nSpatial Analysis:")
    
    # Calculate statistics for negative days
    negative_data = ds.ETo.isel(time=negative_days)
    
    # Find grid cells that frequently have negative values
    negative_cells = (negative_data < 0).sum(dim='time')
    max_negative_days = negative_cells.max().values
    
    print(f"  Maximum negative days per grid cell: {max_negative_days}")
    print(f"  Grid cells with >10 negative days: {(negative_cells > 10).sum().values}")
    print(f"  Grid cells with >20 negative days: {(negative_cells > 20).sum().values}")
    
    # Analyze seasonal patterns
    if hasattr(dates, 'month'):
        monthly_counts = []
        for month in range(1, 13):
            month_days = dates[dates.month == month]
            monthly_counts.append(len(month_days))
        
        print(f"\nSeasonal Analysis:")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, count in zip(months, monthly_counts):
            if count > 0:
                print(f"  {month}: {count} days")
    
    # Show some specific negative values
    print(f"\nMost Negative ETo Values:")
    sorted_indices = np.argsort(daily_min_eto.values[negative_days])
    for i in range(min(5, len(negative_days))):
        idx = negative_days[sorted_indices[i]]
        min_val = daily_min_eto.values[idx]
        if hasattr(dates, 'iloc'):
            date_str = dates.iloc[sorted_indices[i]].strftime("%Y-%m-%d")
        else:
            date_str = f"Day {idx+1}"
        print(f"  {date_str}: {min_val:.2f} mm/day")

def create_negative_summary_plot(nc_file, negative_days, output_dir='./plots'):
    """
    Create a summary plot showing the frequency of negative ETo values.
    
    Parameters:
    -----------
    nc_file : str
        Path to the NetCDF file
    negative_days : array
        Array of day indices with negative ETo
    output_dir : str
        Output directory for plots
    """
    
    print(f"\nCreating negative ETo summary plot...")
    
    # Load the dataset
    ds = xr.open_dataset(nc_file)
    
    # Calculate frequency of negative values per grid cell
    negative_data = ds.ETo.isel(time=negative_days)
    negative_frequency = (negative_data < 0).sum(dim='time')
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8),
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot the frequency
    im = ax.pcolormesh(ds.lon, ds.lat, negative_frequency,
                      transform=ccrs.PlateCarree(),
                      cmap='Reds',
                      shading='auto')
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    
    # Set extent to California
    ax.set_extent([-125, -114, 32, 42], crs=ccrs.PlateCarree())
    
    # Add title
    ax.set_title('Frequency of Negative ETo Values (2023)\nNumber of days with ETo < 0 mm/day', 
                fontsize=14, fontweight='bold')
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8)
    cbar.set_label('Number of days with negative ETo', fontsize=12)
    
    # Save the plot
    output_file = os.path.join(output_dir, 'spatial_cimis_negative_eto_frequency.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Negative ETo frequency plot saved to: {output_file}")
    
    plt.show()

def main():
    """Main function to analyze negative ETo days."""
    
    # Path to the GridMET reprojected file
    nc_file = "/group/moniergrp/SpatialCIMIS/netcdf/test/spatial_cimis_eto_2023_gridmet.nc"
    output_dir = "./plots"
    
    # Check if file exists
    if not os.path.exists(nc_file):
        print(f"Error: File not found: {nc_file}")
        print("Please check the file path and ensure the GridMET processing completed successfully.")
        return
    
    print("="*60)
    print("Spatial CIMIS Negative ETo Analysis")
    print("="*60)
    
    # Find days with negative ETo values
    negative_days, daily_min_eto = find_negative_eto_days(nc_file, threshold=-0.1)
    
    if len(negative_days) == 0:
        print("No days with negative ETo values found!")
        return
    
    # Plot the negative ETo days
    plot_negative_eto_days(nc_file, negative_days, daily_min_eto, output_dir, max_days=6)
    
    # Analyze patterns
    analyze_negative_patterns(nc_file, negative_days, daily_min_eto)
    
    # Create summary plot
    create_negative_summary_plot(nc_file, negative_days, output_dir)
    
    print("\n" + "="*60)
    print("Negative ETo analysis completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()




