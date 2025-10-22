#!/usr/bin/env python3
"""
Extract metadata from ASCII raster files and create a DataFrame.

This script scans ASCII files for a variable and extracts:
- Date information
- Dimensions (rows, cols)
- Extent (corners)
- Pixel size
- CRS
- File size
- NODATA value

Usage:
    python extract_ascii_metadata.py <variable> [output_csv]
    
Examples:
    python extract_ascii_metadata.py ETo
    python extract_ascii_metadata.py Tx tx_metadata.csv
    python extract_ascii_metadata.py Rs rs_metadata.csv
"""

import sys
import glob
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Patch


def parse_date_from_filename(filename, year=None):
    """
    Parse date from filename.
    
    Examples:
        ETo.2010001.asc -> 2010-01-01
        Tx001.asc (with year=2010) -> 2010-01-01
    """
    # Try to extract year and day of year from filename
    # Pattern 1: Variable.YYYYDDD.asc
    match = re.search(r'\.(\d{4})(\d{3})\.', filename)
    if match:
        year = int(match.group(1))
        doy = int(match.group(2))
        date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        return date
    
    # Pattern 2: VariableDDD.asc (need year provided)
    match = re.search(r'[A-Za-z]+(\d{3})\.', filename)
    if match and year:
        doy = int(match.group(1))
        date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        return date
    
    return None


def extract_metadata(filepath):
    """Extract metadata from a single ASCII file."""
    try:
        with rasterio.open(filepath) as src:
            # Read data to check if it's all NaN
            data = src.read(1).astype(float)
            
            # Replace nodata with NaN
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            
            # Check for valid data
            valid_pixels = np.sum(~np.isnan(data))
            total_pixels = data.size
            valid_percentage = 100 * valid_pixels / total_pixels
            
            metadata = {
                'filepath': str(filepath),
                'filename': Path(filepath).name,
                'ncols': src.width,
                'nrows': src.height,
                'xllcorner': src.bounds.left,
                'yllcorner': src.bounds.bottom,
                'xurcorner': src.bounds.right,
                'yurcorner': src.bounds.top,
                'cellsize_x': src.res[0],
                'cellsize_y': abs(src.res[1]),  # abs because can be negative
                'nodata': src.nodata,
                'crs': str(src.crs) if src.crs else None,
                'transform': str(src.transform),
                'file_size_bytes': os.path.getsize(filepath),
                'file_size_kb': os.path.getsize(filepath) / 1024,
                'file_size_mb': os.path.getsize(filepath) / (1024 * 1024),
                'valid_pixels': int(valid_pixels),
                'total_pixels': int(total_pixels),
                'valid_percentage': float(valid_percentage),
                'has_valid_data': valid_pixels > 0
            }
            
            # Calculate extent
            metadata['width_extent'] = metadata['xurcorner'] - metadata['xllcorner']
            metadata['height_extent'] = metadata['yurcorner'] - metadata['yllcorner']
            
            return metadata
    except Exception as e:
        print(f"  Error reading {Path(filepath).name}: {e}")
        return None


def scan_ascii_files(pattern, max_files=None):
    """
    Scan ASCII files and extract metadata.
    
    Parameters:
    -----------
    pattern : str
        Glob pattern for files
    max_files : int or None
        Maximum number of files to process
    """
    print(f"\nSearching for files: {pattern}")
    
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"Error: No files found matching pattern")
        return None
    
    print(f"Found {len(files)} files")
    
    if max_files:
        files = files[:max_files]
        print(f"Processing first {len(files)} files")
    
    print("\nExtracting metadata...")
    
    metadata_list = []
    
    for i, filepath in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(files)} files...")
        
        metadata = extract_metadata(filepath)
        
        if metadata:
            # Try to parse date from filename
            filename = Path(filepath).name
            
            # Try with year in filename
            date = parse_date_from_filename(filename)
            
            # If no date found, try to extract year from path
            if not date:
                year_match = re.search(r'/(\d{4})/', filepath)
                if year_match:
                    year = int(year_match.group(1))
                    date = parse_date_from_filename(filename, year)
            
            metadata['date'] = date
            metadata['year'] = date.year if date else None
            metadata['month'] = date.month if date else None
            metadata['day'] = date.day if date else None
            metadata['day_of_year'] = date.timetuple().tm_yday if date else None
            
            metadata_list.append(metadata)
    
    print(f"  Successfully extracted metadata from {len(metadata_list)} files")
    
    if not metadata_list:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Reorder columns for better readability
    col_order = [
        'filename', 'date', 'year', 'month', 'day', 'day_of_year',
        'ncols', 'nrows', 'cellsize_x', 'cellsize_y',
        'xllcorner', 'yllcorner', 'xurcorner', 'yurcorner',
        'width_extent', 'height_extent',
        'valid_pixels', 'total_pixels', 'valid_percentage', 'has_valid_data',
        'nodata', 'crs',
        'file_size_bytes', 'file_size_kb', 'file_size_mb',
        'filepath', 'transform'
    ]
    
    # Only include columns that exist
    col_order = [col for col in col_order if col in df.columns]
    df = df[col_order]
    
    # Sort by date if available
    if 'date' in df.columns and df['date'].notna().any():
        df = df.sort_values('date')
    
    return df


def plot_unique_footprints(df, variable, output_file=None):
    """
    Plot one example of each unique footprint on the same map.
    
    Parameters:
    -----------
    df : DataFrame
        Metadata dataframe
    variable : str
        Variable name
    output_file : str
        Output filename for the plot
    """
    print(f"\n{'='*70}")
    print("CREATING FOOTPRINT COMPARISON PLOT")
    print(f"{'='*70}")
    
    # Filter to only files with valid data
    df_valid = df[df['has_valid_data'] == True].copy()
    
    if len(df_valid) == 0:
        print("\nNo files with valid data found!")
        return
    
    print(f"\nFiles with valid data: {len(df_valid)} / {len(df)}")
    
    # Find unique dimensions (only from valid files)
    unique_dims = df_valid.groupby(['ncols', 'nrows']).first().reset_index()
    
    print(f"\nFound {len(unique_dims)} unique footprints (with valid data):")
    for idx, row in unique_dims.iterrows():
        print(f"  {row['ncols']} × {row['nrows']}: {row['filename']} ({row['valid_percentage']:.1f}% valid)")
    
    if len(unique_dims) == 0:
        print("No unique footprints found")
        return
    
    # Define colors for different footprints
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    # Create map projection (use PlateCarree for simplicity)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # California Albers projection for data
    crs_data = ccrs.AlbersEqualArea(
        central_longitude=-120,
        central_latitude=0,
        standard_parallels=(34, 40.5),
        false_easting=0,
        false_northing=-4000000
    )
    
    legend_elements = []
    
    # Plot each unique footprint
    for idx, row in unique_dims.iterrows():
        color = colors[idx % len(colors)]
        filepath = row['filepath']
        
        print(f"\nPlotting: {Path(filepath).name}")
        print(f"  Dimensions: {row['ncols']} × {row['nrows']}")
        
        try:
            # Read the raster data
            with rasterio.open(filepath) as src:
                data = src.read(1).astype(float)
                
                # Replace nodata with NaN
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                
                # Normalize data for better visualization
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    vmin, vmax = np.percentile(valid_data, [2, 98])
                    data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
                else:
                    data_norm = data
                
                # Get extent in the data's CRS
                extent = [src.bounds.left, src.bounds.right, 
                         src.bounds.bottom, src.bounds.top]
                
                print(f"  Extent: {extent}")
                print(f"  Valid pixels: {np.sum(~np.isnan(data)):,}")
            
            # Plot using pcolormesh for better handling with cartopy
            # Create coordinate arrays - flip y to correct orientation
            x = np.linspace(extent[0], extent[1], row['ncols'] + 1)
            y = np.linspace(extent[3], extent[2], row['nrows'] + 1)  # Flip: top to bottom
            
            # Use single color with transparency
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap([color])
            
            # Create a mask for valid data
            plot_data = np.where(~np.isnan(data), 1, np.nan)
            
            im = ax.pcolormesh(x, y, plot_data,
                             transform=crs_data,
                             cmap=cmap,
                             alpha=0.6,
                             shading='auto')
            
            # Add to legend
            label = f"{row['ncols']}×{row['nrows']} ({row['filename']})"
            legend_elements.append(Patch(facecolor=color, alpha=0.6, label=label))
            
        except Exception as e:
            print(f"  Error plotting {filepath}: {e}")
            continue
    
    # Add coastlines and features
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    
    # Set extent to California
    ax.set_extent([-125, -114, 32, 42.5], crs=ccrs.PlateCarree())
    
    # Title
    title = f'{variable} ASCII File Footprints\n'
    title += f'({len(unique_dims)} unique dimensions, 60% transparency)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
             framealpha=0.9, title='Dimensions (cols×rows)')
    
    plt.tight_layout()
    
    # Save plot
    if output_file is None:
        output_file = f'{variable.lower()}_footprints.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    plt.close()


def print_summary_statistics(df):
    """Print summary statistics of the metadata."""
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    print(f"\nTotal files: {len(df)}")
    
    # Date range
    if 'date' in df.columns and df['date'].notna().any():
        date_min = df['date'].min()
        date_max = df['date'].max()
        print(f"Date range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
        print(f"Years: {sorted(df['year'].dropna().unique())}")
    
    # Dimensions
    print(f"\nDimensions:")
    print(f"  Unique column counts: {sorted(df['ncols'].unique())}")
    print(f"  Unique row counts: {sorted(df['nrows'].unique())}")
    
    # Cell size
    print(f"\nCell size:")
    print(f"  X: {df['cellsize_x'].unique()}")
    print(f"  Y: {df['cellsize_y'].unique()}")
    
    # Extent
    print(f"\nExtent:")
    print(f"  X range: {df['xllcorner'].min():.2f} to {df['xurcorner'].max():.2f}")
    print(f"  Y range: {df['yllcorner'].min():.2f} to {df['yurcorner'].max():.2f}")
    
    # File sizes
    print(f"\nFile sizes:")
    print(f"  Total: {df['file_size_mb'].sum():.2f} MB")
    print(f"  Mean: {df['file_size_kb'].mean():.2f} KB")
    print(f"  Min: {df['file_size_kb'].min():.2f} KB")
    print(f"  Max: {df['file_size_kb'].max():.2f} KB")
    
    # Check for dimension changes
    print(f"\nDimension consistency:")
    ncols_counts = df['ncols'].value_counts()
    for ncols, count in ncols_counts.items():
        pct = 100 * count / len(df)
        print(f"  {ncols} columns: {count} files ({pct:.1f}%)")
    
    # Check for footprint changes over time
    if 'date' in df.columns and len(df['ncols'].unique()) > 1:
        print(f"\nFootprint changes over time:")
        for ncols in sorted(df['ncols'].unique()):
            subset = df[df['ncols'] == ncols]
            if not subset['date'].isna().all():
                date_range = f"{subset['date'].min().strftime('%Y-%m-%d')} to {subset['date'].max().strftime('%Y-%m-%d')}"
                print(f"  {ncols} columns: {date_range} ({len(subset)} files)")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python extract_ascii_metadata.py <variable> [output_csv] [--max-files N]")
        print("\nExamples:")
        print("  python extract_ascii_metadata.py ETo")
        print("  python extract_ascii_metadata.py Tx tx_metadata.csv")
        print("  python extract_ascii_metadata.py Rs rs_metadata.csv --max-files 1000")
        sys.exit(1)
    
    variable = sys.argv[1]
    output_csv = None
    max_files = None
    
    # Parse additional arguments
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '--max-files' and i + 1 < len(sys.argv):
            max_files = int(sys.argv[i + 1])
        elif not sys.argv[i].startswith('--') and sys.argv[i-1] != '--max-files':
            output_csv = sys.argv[i]
    
    if output_csv is None:
        output_csv = f"{variable.lower()}_metadata.csv"
    
    print("="*70)
    print("ASCII METADATA EXTRACTION")
    print("="*70)
    print(f"\nVariable: {variable}")
    print(f"Output: {output_csv}")
    if max_files:
        print(f"Max files: {max_files}")
    
    # Search pattern
    pattern = f"/group/moniergrp/SpatialCIMIS/ascii/{variable}*.asc"
    
    # Extract metadata
    df = scan_ascii_files(pattern, max_files)
    
    if df is None or len(df) == 0:
        print("\nError: No metadata extracted")
        sys.exit(1)
    
    # Print summary
    print_summary_statistics(df)
    
    # Create footprint comparison plot
    plot_file = f"{variable.lower()}_footprints.png"
    plot_unique_footprints(df, variable, plot_file)
    
    # Save to CSV
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    df.to_csv(output_csv, index=False)
    print(f"\nMetadata saved to: {output_csv}")
    print(f"Total records: {len(df)}")
    
    # Print first few rows
    print(f"\nFirst 5 records:")
    print(df[['filename', 'date', 'ncols', 'nrows', 'file_size_kb']].head())
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

