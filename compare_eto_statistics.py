#!/usr/bin/env python3
"""
Compare Spatial CIMIS ETo Statistics: Original vs GridMET Reprojected

This script compares statistics between the original Spatial CIMIS ETo data
and the GridMET reprojected version to assess the reprojection quality.

Author: UC Davis Global Environmental Change Lab
Date: 2024
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from datetime import datetime
import pyproj
from pyproj import Transformer

def load_and_prepare_data(original_file, gridmet_file, gridmet_pet_file):
    """
    Load and prepare all three datasets for comparison.
    
    Parameters:
    -----------
    original_file : str
        Path to original Spatial CIMIS NetCDF file
    gridmet_file : str
        Path to GridMET reprojected NetCDF file
    gridmet_pet_file : str
        Path to original GridMET PET NetCDF file
        
    Returns:
    --------
    tuple : (original_ds, gridmet_ds, gridmet_pet_ds)
    """
    
    print("Loading datasets...")
    
    # Load original data
    print(f"Loading original: {original_file}")
    original_ds = xr.open_dataset(original_file)
    
    # Load GridMET reprojected data
    print(f"Loading GridMET reprojected: {gridmet_file}")
    gridmet_ds = xr.open_dataset(gridmet_file)
    
    # Load original GridMET PET data
    print(f"Loading GridMET PET: {gridmet_pet_file}")
    gridmet_pet_ds = xr.open_dataset(gridmet_pet_file)
    
    # Print basic info
    print(f"\nOriginal dataset:")
    print(f"  Dimensions: {original_ds.dims}")
    if 'lat' in original_ds.coords and 'lon' in original_ds.coords:
        print(f"  Lat range: {original_ds.lat.min().values:.3f} to {original_ds.lat.max().values:.3f}")
        print(f"  Lon range: {original_ds.lon.min().values:.3f} to {original_ds.lon.max().values:.3f}")
    else:
        print(f"  X range: {original_ds.x.min().values:.0f} to {original_ds.x.max().values:.0f}")
        print(f"  Y range: {original_ds.y.min().values:.0f} to {original_ds.y.max().values:.0f}")
        print("  Note: Using projected coordinates (x, y) instead of lat/lon")
    
    print(f"\nGridMET reprojected dataset:")
    print(f"  Dimensions: {gridmet_ds.dims}")
    print(f"  Lat range: {gridmet_ds.lat.min().values:.3f} to {gridmet_ds.lat.max().values:.3f}")
    print(f"  Lon range: {gridmet_ds.lon.min().values:.3f} to {gridmet_ds.lon.max().values:.3f}")
    
    print(f"\nGridMET PET dataset:")
    print(f"  Dimensions: {gridmet_pet_ds.dims}")
    print(f"  Lat range: {gridmet_pet_ds.lat.min().values:.3f} to {gridmet_pet_ds.lat.max().values:.3f}")
    print(f"  Lon range: {gridmet_pet_ds.lon.min().values:.3f} to {gridmet_pet_ds.lon.max().values:.3f}")
    
    return original_ds, gridmet_ds, gridmet_pet_ds

def calculate_statistics(ds, dataset_name):
    """
    Calculate annual statistics for a dataset.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset to analyze
    dataset_name : str
        Name for labeling
        
    Returns:
    --------
    dict : Dictionary of statistics
    """
    
    print(f"\nCalculating statistics for {dataset_name}...")
    
    # Determine the variable name and time dimension
    if 'ETo' in ds.data_vars:
        var_name = 'ETo'
        time_dim = 'time'
    elif 'potential_evapotranspiration' in ds.data_vars:
        var_name = 'potential_evapotranspiration'
        time_dim = 'day'
    else:
        # Try to find any variable that looks like ETo/PET
        possible_vars = [v for v in ds.data_vars if 'eto' in v.lower() or 'pet' in v.lower() or 'evapo' in v.lower()]
        if possible_vars:
            var_name = possible_vars[0]
            time_dim = 'time' if 'time' in ds.dims else 'day'
        else:
            raise ValueError(f"Could not find ETo/PET variable in {dataset_name}")
    
    print(f"  Using variable: {var_name}, time dimension: {time_dim}")
    
    # Calculate annual statistics
    eto_mean = ds[var_name].mean(dim=time_dim)
    eto_max = ds[var_name].max(dim=time_dim)
    eto_min = ds[var_name].min(dim=time_dim)
    eto_std = ds[var_name].std(dim=time_dim)
    
    # Calculate overall statistics
    overall_mean = eto_mean.mean().values
    overall_max = eto_max.max().values
    overall_min = eto_min.min().values
    overall_std = eto_std.mean().values
    
    print(f"  Overall mean ETo: {overall_mean:.2f} mm/day")
    print(f"  Overall max ETo: {overall_max:.2f} mm/day")
    print(f"  Overall min ETo: {overall_min:.2f} mm/day")
    print(f"  Overall std ETo: {overall_std:.2f} mm/day")
    
    return {
        'mean': eto_mean,
        'max': eto_max,
        'min': eto_min,
        'std': eto_std,
        'overall_mean': overall_mean,
        'overall_max': overall_max,
        'overall_min': overall_min,
        'overall_std': overall_std
    }

def plot_comparison_statistics(original_stats, gridmet_stats, gridmet_pet_stats, original_ds, gridmet_ds, gridmet_pet_ds, output_dir):
    """
    Create side-by-side comparison plots between all three datasets.
    
    Parameters:
    -----------
    original_stats : dict
        Statistics from original dataset
    gridmet_stats : dict
        Statistics from GridMET reprojected dataset
    gridmet_pet_stats : dict
        Statistics from GridMET PET dataset
    original_ds : xarray.Dataset
        Original dataset
    gridmet_ds : xarray.Dataset
        GridMET reprojected dataset
    gridmet_pet_ds : xarray.Dataset
        GridMET PET dataset
    output_dir : str
        Output directory for plots
    """
    
    print("\nCreating three-way comparison plots...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots for three-way comparison (3 rows, 4 columns)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Define statistics to plot
    stats_names = ['mean', 'max', 'min', 'std']
    stats_titles = ['Annual Mean ETo', 'Annual Maximum ETo', 
                   'Annual Minimum ETo', 'Annual Std Dev ETo']
    stats_labels = ['Mean ETo (mm/day)', 'Max ETo (mm/day)', 
                   'Min ETo (mm/day)', 'Std Dev ETo (mm/day)']
    
    # Plot original data (top row) - using proper coordinate transformation
    for i, (stat_name, title, label) in enumerate(zip(stats_names, stats_titles, stats_labels)):
        ax = axes[0, i]
        
        # Get projected coordinates
        x_coords = original_ds.x.values
        y_coords = original_ds.y.values
        
        # Create proper coordinate transformation from California Albers to WGS84
        # California Albers EPSG:3310
        transformer = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)
        
        # Transform coordinates
        lon_mesh, lat_mesh = transformer.transform(
            np.meshgrid(x_coords, y_coords)[0],
            np.meshgrid(x_coords, y_coords)[1]
        )
        
        # Plot the data
        im = ax.pcolormesh(lon_mesh, lat_mesh, original_stats[stat_name],
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
        ax.set_title(f'Original Spatial CIMIS: {title}', fontsize=12, fontweight='bold')
        
        # Add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        cbar.set_label(label, fontsize=10)
    
    # Plot GridMET reprojected data (middle row)
    for i, (stat_name, title, label) in enumerate(zip(stats_names, stats_titles, stats_labels)):
        ax = axes[1, i]
        
        # Plot the data
        im = ax.pcolormesh(gridmet_ds.lon, gridmet_ds.lat, gridmet_stats[stat_name],
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
        ax.set_title(f'GridMET Reprojected: {title}', fontsize=12, fontweight='bold')
        
        # Add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        cbar.set_label(label, fontsize=10)
    
    # Plot GridMET PET data (bottom row)
    for i, (stat_name, title, label) in enumerate(zip(stats_names, stats_titles, stats_labels)):
        ax = axes[2, i]
        
        # Plot the data
        im = ax.pcolormesh(gridmet_pet_ds.lon, gridmet_pet_ds.lat, gridmet_pet_stats[stat_name],
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
        ax.set_title(f'GridMET PET: {title}', fontsize=12, fontweight='bold')
        
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
    fig.suptitle('ETo Statistics Comparison: Spatial CIMIS vs GridMET Reprojected vs GridMET PET (2023)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the plot
    output_file = os.path.join(output_dir, 'spatial_cimis_eto_three_way_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Three-way comparison plot saved to: {output_file}")
    
    plt.show()

def plot_difference_maps(original_stats, gridmet_stats, original_ds, gridmet_ds, output_dir):
    """
    Create a focused mean comparison plot.
    
    Parameters:
    -----------
    original_stats : dict
        Statistics from original dataset
    gridmet_stats : dict
        Statistics from GridMET dataset
    original_ds : xarray.Dataset
        Original dataset
    gridmet_ds : xarray.Dataset
        GridMET dataset
    output_dir : str
        Output directory for plots
    """
    
    print("\nCreating mean comparison plot...")
    
    # Create figure for mean comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot original mean (left)
    ax1 = axes[0]
    
    # Get projected coordinates
    x_coords = original_ds.x.values
    y_coords = original_ds.y.values
    
    # Create proper coordinate transformation from California Albers to WGS84
    transformer = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)
    
    # Transform coordinates
    lon_mesh, lat_mesh = transformer.transform(
        np.meshgrid(x_coords, y_coords)[0],
        np.meshgrid(x_coords, y_coords)[1]
    )
    
    im1 = ax1.pcolormesh(lon_mesh, lat_mesh, original_stats['mean'],
                        transform=ccrs.PlateCarree(),
                        cmap='viridis',
                        shading='auto')
    
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax1.add_feature(cfeature.STATES, linewidth=0.5)
    ax1.set_extent([-125, -114, 32, 42], crs=ccrs.PlateCarree())
    ax1.set_title('Original Spatial CIMIS ETo Mean', fontsize=12, fontweight='bold')
    
    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar1.set_label('Mean ETo (mm/day)', fontsize=10)
    
    # Plot GridMET mean (right)
    ax2 = axes[1]
    im2 = ax2.pcolormesh(gridmet_ds.lon, gridmet_ds.lat, gridmet_stats['mean'],
                        transform=ccrs.PlateCarree(),
                        cmap='viridis',
                        shading='auto')
    
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax2.add_feature(cfeature.STATES, linewidth=0.5)
    ax2.set_extent([-125, -114, 32, 42], crs=ccrs.PlateCarree())
    ax2.set_title('GridMET Reprojected ETo Mean', fontsize=12, fontweight='bold')
    
    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar2.set_label('Mean ETo (mm/day)', fontsize=10)
    
    # Add main title
    fig.suptitle('Spatial CIMIS ETo Mean Comparison: Original vs GridMET Reprojected', 
                fontsize=14, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the plot
    output_file = os.path.join(output_dir, 'spatial_cimis_eto_mean_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Mean comparison plot saved to: {output_file}")
    
    plt.show()

def print_summary_comparison(original_stats, gridmet_stats):
    """
    Print a summary comparison of statistics.
    
    Parameters:
    -----------
    original_stats : dict
        Statistics from original dataset
    gridmet_stats : dict
        Statistics from GridMET dataset
    """
    
    print("\n" + "="*60)
    print("STATISTICS COMPARISON SUMMARY")
    print("="*60)
    
    print(f"{'Metric':<20} {'Original':<12} {'GridMET':<12} {'Difference':<12}")
    print("-" * 60)
    
    # Compare overall statistics
    metrics = [
        ('Mean ETo (mm/day)', original_stats['overall_mean'], gridmet_stats['overall_mean']),
        ('Max ETo (mm/day)', original_stats['overall_max'], gridmet_stats['overall_max']),
        ('Min ETo (mm/day)', original_stats['overall_min'], gridmet_stats['overall_min']),
        ('Std Dev (mm/day)', original_stats['overall_std'], gridmet_stats['overall_std'])
    ]
    
    for metric_name, orig_val, gridmet_val in metrics:
        diff = gridmet_val - orig_val
        print(f"{metric_name:<20} {orig_val:<12.2f} {gridmet_val:<12.2f} {diff:<12.2f}")
    
    print("\n" + "="*60)
    print("REPROJECTION QUALITY ASSESSMENT")
    print("="*60)
    
    # Calculate relative differences
    mean_diff_pct = ((gridmet_stats['overall_mean'] - original_stats['overall_mean']) / 
                     original_stats['overall_mean'] * 100)
    max_diff_pct = ((gridmet_stats['overall_max'] - original_stats['overall_max']) / 
                    original_stats['overall_max'] * 100)
    
    print(f"Mean ETo difference: {mean_diff_pct:.2f}%")
    print(f"Max ETo difference: {max_diff_pct:.2f}%")
    
    if abs(mean_diff_pct) < 5:
        print("✓ Mean difference < 5% - Good reprojection quality")
    else:
        print("⚠ Mean difference > 5% - Check reprojection parameters")
    
    if abs(max_diff_pct) < 10:
        print("✓ Max difference < 10% - Good reprojection quality")
    else:
        print("⚠ Max difference > 10% - Check reprojection parameters")

def main():
    """Main function to run the comparison analysis."""
    
    # File paths
    original_file = "/group/moniergrp/SpatialCIMIS/netcdf/test/spatial_cimis_eto_2023.nc"
    gridmet_file = "/group/moniergrp/SpatialCIMIS/netcdf/test/spatial_cimis_eto_2023_gridmet.nc"
    gridmet_pet_file = "/group/moniergrp/gridMET/pet_2023.nc"
    output_dir = "./plots"
    
    # Check if files exist
    if not os.path.exists(original_file):
        print(f"Error: Original file not found: {original_file}")
        return
    
    if not os.path.exists(gridmet_file):
        print(f"Error: GridMET reprojected file not found: {gridmet_file}")
        return
    
    if not os.path.exists(gridmet_pet_file):
        print(f"Error: GridMET PET file not found: {gridmet_pet_file}")
        return
    
    print("="*60)
    print("ETo Statistics Comparison")
    print("Spatial CIMIS vs GridMET Reprojected vs GridMET PET")
    print("="*60)
    
    # Load and prepare data
    original_ds, gridmet_ds, gridmet_pet_ds = load_and_prepare_data(original_file, gridmet_file, gridmet_pet_file)
    
    # Calculate statistics
    original_stats = calculate_statistics(original_ds, "Original Spatial CIMIS")
    gridmet_stats = calculate_statistics(gridmet_ds, "GridMET Reprojected")
    gridmet_pet_stats = calculate_statistics(gridmet_pet_ds, "GridMET PET")
    
    # Create comparison plots
    plot_comparison_statistics(original_stats, gridmet_stats, gridmet_pet_stats, 
                             original_ds, gridmet_ds, gridmet_pet_ds, output_dir)
    
    # Print summary comparison
    print_summary_comparison(original_stats, gridmet_stats)
    
    print("\n" + "="*60)
    print("Three-way comparison analysis completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
