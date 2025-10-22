#!/usr/bin/env python3
"""
Plot Rs data before and after Oct 23, 2018 grid change
Shows the spatial extent change with overlapping transparent layers
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import pandas as pd
from pyproj import Transformer
import os

def plot_grid_change(before_date='2018-10-22', after_date='2018-10-24', 
                     netcdf_path='/group/moniergrp/SpatialCIMIS/netcdf/test/',
                     output_file='rs_grid_change_comparison.png'):
    """
    Plot Rs data before and after grid change.
    
    Parameters:
    -----------
    before_date : str
        Date before grid change (format: YYYY-MM-DD)
    after_date : str
        Date after grid change (format: YYYY-MM-DD)
    netcdf_path : str
        Path to NetCDF files
    output_file : str
        Output filename
    """
    
    print("="*60)
    print("Rs Grid Change Visualization")
    print("="*60)
    print(f"\nBefore: {before_date}")
    print(f"After:  {after_date}")
    print(f"Data:   {netcdf_path}")
    
    # Load 2018 data
    print(f"\nLoading Rs 2018 data...")
    ds = xr.open_dataset(f'{netcdf_path}spatial_cimis_rs_2018.nc')
    
    print(f"  Grid shape: {ds.Rs.shape}")
    print(f"  Date range: {ds.time.min().values} to {ds.time.max().values}")
    
    # Extract specific days
    print(f"\nExtracting dates...")
    before_data = ds.Rs.sel(time=before_date, method='nearest').values
    after_data = ds.Rs.sel(time=after_date, method='nearest').values
    
    # Get coordinates
    x = ds.x.values
    y = ds.y.values
    
    print(f"  Before ({before_date}): {np.isfinite(before_data).sum():,} valid cells")
    print(f"  After  ({after_date}): {np.isfinite(after_data).sum():,} valid cells")
    print(f"  Lost coverage: {np.isfinite(before_data).sum() - np.isfinite(after_data).sum():,} cells")
    
    # Load station locations
    print(f"\nLoading station metadata...")
    try:
        # Try to load from unified analysis output first
        meta_file = '/home/salba/SpatialCIMIS/output/station_metadata_rs.csv'
        if not os.path.exists(meta_file):
            # Fallback to main station list
            meta_file = '/group/moniergrp/SpatialCIMIS/CIMIS/CIMIS_Stations.csv'
        
        station_meta = pd.read_csv(meta_file)
        print(f"  Loaded {len(station_meta)} stations")
        
        # Load spatial CIMIS station data to determine which stations have coverage
        spatial_data = pd.read_csv('/home/salba/SpatialCIMIS/output/spatial_cimis_station_rs.csv',
                                   index_col='date', parse_dates=True)
        
        # Check which stations have data before/after
        before_check = spatial_data[(spatial_data.index >= before_date) & 
                                    (spatial_data.index <= before_date)]
        after_check = spatial_data[(spatial_data.index >= after_date) & 
                                   (spatial_data.index <= after_date)]
        
        stations_before = set(before_check.columns[before_check.notna().any()])
        stations_after = set(after_check.columns[after_check.notna().any()])
        
        lost_stations = stations_before - stations_after
        kept_stations = stations_after
        
        print(f"  Stations with coverage before: {len(stations_before)}")
        print(f"  Stations with coverage after: {len(stations_after)}")
        print(f"  Stations that lost coverage: {len(lost_stations)}")
        
        # Transform station lat/lon to projected coordinates
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3310", always_xy=True)
        station_meta['x_proj'], station_meta['y_proj'] = transformer.transform(
            station_meta['Longitude'].values,
            station_meta['Latitude'].values
        )
        
        print(f"  Station coordinate range:")
        print(f"    X: {station_meta['x_proj'].min():.0f} to {station_meta['x_proj'].max():.0f} m")
        print(f"    Y: {station_meta['y_proj'].min():.0f} to {station_meta['y_proj'].max():.0f} m")
        print(f"  Grid extent:")
        print(f"    X: {x.min():.0f} to {x.max():.0f} m")
        print(f"    Y: {y.min():.0f} to {y.max():.0f} m")
        
        has_stations = True
        
    except Exception as e:
        print(f"  Warning: Could not load station data: {e}")
        has_stations = False
    
    # Create figure
    print(f"\nCreating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Before only (red)
    ax1 = axes[0]
    before_masked = np.where(np.isfinite(before_data), before_data, np.nan)
    
    # Use pcolormesh instead of imshow for better coordinate handling
    X, Y = np.meshgrid(x, y)
    im1 = ax1.pcolormesh(X, Y, before_masked, cmap='Reds', alpha=0.6, 
                         vmin=0, vmax=30, shading='auto')
    
    # Add stations to panel 1
    if has_stations:
        # All stations (should have coverage before)
        scatter1 = ax1.scatter(station_meta['x_proj'], station_meta['y_proj'], 
                   c='black', s=80, marker='o', edgecolors='yellow', linewidths=1.5,
                   alpha=1.0, zorder=100, label='CIMIS Stations')
        print(f"\n  Panel 1: Plotted {len(station_meta)} stations")
    
    ax1.set_title(f'Before Grid Change\n{before_date}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(y.min(), y.max())
    plt.colorbar(im1, ax=ax1, label='Rs (MJ/m²/day)', shrink=0.7)
    if has_stations:
        ax1.legend(loc='upper right', fontsize=8)
    
    # Panel 2: After only (blue)
    ax2 = axes[1]
    after_masked = np.where(np.isfinite(after_data), after_data, np.nan)
    
    im2 = ax2.pcolormesh(X, Y, after_masked, cmap='Blues', alpha=0.6,
                         vmin=0, vmax=30, shading='auto')
    
    # Add stations to panel 2 - color by coverage status
    if has_stations:
        # Stations that kept coverage (green)
        kept_meta = station_meta[station_meta['StationNbr'].astype(str).isin(kept_stations)]
        if not kept_meta.empty:
            scatter2a = ax2.scatter(kept_meta['x_proj'], kept_meta['y_proj'],
                       c='green', s=80, marker='o', edgecolors='white', linewidths=2,
                       alpha=1.0, zorder=100, label='Kept coverage')
            print(f"  Panel 2: Plotted {len(kept_meta)} stations with kept coverage")
        
        # Stations that lost coverage (orange)
        lost_meta = station_meta[station_meta['StationNbr'].astype(str).isin(lost_stations)]
        if not lost_meta.empty:
            scatter2b = ax2.scatter(lost_meta['x_proj'], lost_meta['y_proj'],
                       c='orange', s=100, marker='x', edgecolors='black', linewidths=2,
                       alpha=1.0, zorder=100, label='Lost coverage')
            print(f"  Panel 2: Plotted {len(lost_meta)} stations with lost coverage")
    
    ax2.set_title(f'After Grid Change\n{after_date}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(y.min(), y.max())
    plt.colorbar(im2, ax=ax2, label='Rs (MJ/m²/day)', shrink=0.7)
    if has_stations:
        ax2.legend(loc='upper right', fontsize=8)
    
    # Panel 3: Overlay - both with transparency
    ax3 = axes[2]
    
    # Plot before in red with transparency
    im3a = ax3.pcolormesh(X, Y, before_masked, cmap='Reds', alpha=0.6,
                          vmin=0, vmax=30, shading='auto')
    
    # Plot after in blue with transparency
    im3b = ax3.pcolormesh(X, Y, after_masked, cmap='Blues', alpha=0.6,
                          vmin=0, vmax=30, shading='auto')
    
    # Add stations to panel 3 - color by coverage status
    if has_stations:
        print(f"\n  Adding stations to overlay panel...")
        # Stations that kept coverage (green circles)
        kept_meta = station_meta[station_meta['StationNbr'].astype(str).isin(kept_stations)]
        print(f"    Kept coverage: {len(kept_meta)} stations")
        if not kept_meta.empty:
            ax3.scatter(kept_meta['x_proj'], kept_meta['y_proj'],
                       c='green', s=100, marker='o', edgecolors='white', linewidths=2,
                       alpha=1.0, zorder=100, label='Kept coverage')
        
        # Stations that lost coverage (red X)
        lost_meta = station_meta[station_meta['StationNbr'].astype(str).isin(lost_stations)]
        print(f"    Lost coverage: {len(lost_meta)} stations")
        if not lost_meta.empty:
            ax3.scatter(lost_meta['x_proj'], lost_meta['y_proj'],
                       c='red', s=120, marker='x', edgecolors='black', linewidths=2.5,
                       alpha=1.0, zorder=100, label='Lost coverage')
    
    ax3.set_title('Overlay: Red=Before, Blue=After\nPurple=Overlap', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    ax3.set_xlim(x.min(), x.max())
    ax3.set_ylim(y.min(), y.max())
    
    # Create custom legend
    red_patch = mpatches.Patch(color='red', alpha=0.6, label=f'Grid: Before only')
    blue_patch = mpatches.Patch(color='blue', alpha=0.6, label=f'Grid: After only')
    purple_patch = mpatches.Patch(color='purple', alpha=0.6, label='Grid: Overlap')
    
    # Add station markers to legend if available
    if has_stations:
        from matplotlib.lines import Line2D
        green_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                             markersize=7, label='Stations: Kept', markeredgecolor='white')
        red_marker = Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                           markersize=7, label='Stations: Lost', markeredgecolor='black')
        ax3.legend(handles=[red_patch, blue_patch, purple_patch, green_marker, red_marker], 
                  loc='upper right', fontsize=8)
    else:
        ax3.legend(handles=[red_patch, blue_patch, purple_patch], 
                  loc='upper right', fontsize=9)
    
    # Add statistics text
    before_only = np.isfinite(before_data) & ~np.isfinite(after_data)
    after_only = np.isfinite(after_data) & ~np.isfinite(before_data)
    overlap = np.isfinite(before_data) & np.isfinite(after_data)
    
    if has_stations:
        stats_text = f"""Grid Change Statistics:
Grid cells:
  Before only: {before_only.sum():,} cells
  After only:  {after_only.sum():,} cells  
  Overlap:     {overlap.sum():,} cells
  Lost:        {before_only.sum():,} cells

Stations:
  Total:       {len(station_meta)}
  Kept:        {len(kept_stations)}
  Lost:        {len(lost_stations)}"""
    else:
        stats_text = f"""Grid Change Statistics:
Before only: {before_only.sum():,} cells
After only:  {after_only.sum():,} cells  
Overlap:     {overlap.sum():,} cells
Lost:        {before_only.sum():,} cells"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Spatial CIMIS Rs - Grid Coverage Change on Oct 23, 2018',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_file}")
    
    plt.close()
    ds.close()
    
    print("\n" + "="*60)
    print("Plot complete!")
    print("="*60)
    
    return {
        'before_cells': np.isfinite(before_data).sum(),
        'after_cells': np.isfinite(after_data).sum(),
        'lost_cells': before_only.sum(),
        'gained_cells': after_only.sum(),
        'overlap_cells': overlap.sum()
    }


if __name__ == "__main__":
    import sys
    
    # Allow custom dates from command line
    before = sys.argv[1] if len(sys.argv) > 1 else '2018-10-22'
    after = sys.argv[2] if len(sys.argv) > 2 else '2018-10-24'
    output = sys.argv[3] if len(sys.argv) > 3 else 'rs_grid_change_comparison.png'
    
    stats = plot_grid_change(before_date=before, after_date=after, output_file=output)
    
    print(f"\nSummary:")
    print(f"  Valid cells before: {stats['before_cells']:,}")
    print(f"  Valid cells after:  {stats['after_cells']:,}")
    print(f"  Lost coverage:      {stats['lost_cells']:,} cells")
    print(f"  Gained coverage:    {stats['gained_cells']:,} cells")
    print(f"  Overlap:            {stats['overlap_cells']:,} cells")
    print(f"  Change:             {stats['after_cells'] - stats['before_cells']:+,} cells")

