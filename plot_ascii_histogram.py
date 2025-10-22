#!/usr/bin/env python3
"""
Plot histogram of number of rows (or other attributes) in ASCII raster files.

This script reads a group of ASCII files matching a wildcard pattern and creates
a histogram of their dimensions or other properties.

Usage:
    python plot_ascii_histogram.py "/path/to/files/*.asc" --attribute nrows
    python plot_ascii_histogram.py "/path/to/files/Tx*.asc" --attribute ncols
    python plot_ascii_histogram.py "/group/moniergrp/SpatialCIMIS/Tx/2010/*.asc"
"""

import sys
import argparse
from pathlib import Path
import glob
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def read_ascii_attribute(filepath, attribute='nrows'):
    """
    Read a specific attribute from an ASCII raster file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the ASCII file
    attribute : str
        Attribute to read: 'nrows', 'ncols', 'cellsize_x', 'cellsize_y', 
                          'xllcorner', 'yllcorner', 'nodata'
    
    Returns:
    --------
    value : float or int or None
        The attribute value, or None if error
    """
    try:
        with rasterio.open(filepath) as src:
            if attribute == 'nrows':
                return src.height
            elif attribute == 'ncols':
                return src.width
            elif attribute == 'cellsize_x':
                return src.res[0]
            elif attribute == 'cellsize_y':
                return src.res[1]
            elif attribute == 'xllcorner':
                return src.bounds.left
            elif attribute == 'yllcorner':
                return src.bounds.bottom
            elif attribute == 'xurcorner':
                return src.bounds.right
            elif attribute == 'yurcorner':
                return src.bounds.top
            elif attribute == 'nodata':
                return src.nodata
            else:
                print(f"Warning: Unknown attribute '{attribute}'")
                return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def collect_attributes(file_pattern, attribute='nrows', max_files=None):
    """
    Collect attribute values from all files matching the pattern.
    
    Parameters:
    -----------
    file_pattern : str
        Glob pattern for files (e.g., "/path/*.asc")
    attribute : str
        Attribute to collect
    max_files : int or None
        Maximum number of files to process (None for all)
    
    Returns:
    --------
    values : list
        List of attribute values
    filenames : list
        List of corresponding file names
    """
    print(f"\nSearching for files matching: {file_pattern}")
    
    # Get list of matching files
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        print(f"Error: No files found matching pattern: {file_pattern}")
        return [], []
    
    print(f"Found {len(files)} files")
    
    if max_files and len(files) > max_files:
        print(f"Limiting to first {max_files} files")
        files = files[:max_files]
    
    # Collect attribute values
    print(f"\nReading '{attribute}' from each file...")
    values = []
    filenames = []
    
    for i, filepath in enumerate(files, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(files)} files...")
        
        value = read_ascii_attribute(filepath, attribute)
        if value is not None:
            values.append(value)
            filenames.append(Path(filepath).name)
    
    print(f"  Successfully read {len(values)}/{len(files)} files")
    
    return values, filenames


def plot_histogram(values, attribute, output_file=None, title=None):
    """
    Create histogram of attribute values.
    
    Parameters:
    -----------
    values : list
        List of attribute values
    attribute : str
        Name of the attribute
    output_file : str or None
        Path to save the plot (if None, display only)
    title : str or None
        Custom title for the plot
    """
    print("\nCreating histogram...")
    
    if not values:
        print("Error: No values to plot")
        return
    
    # Calculate statistics
    values_array = np.array(values)
    mean_val = np.mean(values_array)
    median_val = np.median(values_array)
    std_val = np.std(values_array)
    min_val = np.min(values_array)
    max_val = np.max(values_array)
    unique_vals = len(np.unique(values_array))
    
    print(f"\nStatistics for '{attribute}':")
    print(f"  Count: {len(values)}")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Median: {median_val:.4f}")
    print(f"  Std Dev: {std_val:.4f}")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")
    print(f"  Unique values: {unique_vals}")
    
    # If only one unique value, use a bar plot instead
    if unique_vals <= 10:
        print(f"\nValue counts:")
        counter = Counter(values)
        for val, count in sorted(counter.items()):
            print(f"  {val}: {count} files ({100*count/len(values):.1f}%)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine number of bins
    if unique_vals <= 20:
        # Use discrete bins for small number of unique values
        bins = sorted(np.unique(values_array))
        if len(bins) > 1:
            bin_width = np.min(np.diff(bins))
            bins = np.append(bins - bin_width/2, bins[-1] + bin_width/2)
        else:
            bins = [bins[0] - 0.5, bins[0] + 0.5]
    else:
        # Use auto bins for continuous data
        bins = 'auto'
    
    # Create histogram
    n, bins_edges, patches = ax.hist(values_array, bins=bins, 
                                      color='steelblue', alpha=0.7, 
                                      edgecolor='black', linewidth=1.2)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_val:.2f}')
    
    # Labels and title
    ax.set_xlabel(f'{attribute}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Files)', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Distribution of {attribute} in ASCII Files\n(n={len(values)} files)', 
                    fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    stats_text = f'Min: {min_val:.2f}\nMax: {max_val:.2f}\nStd: {std_val:.2f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Plot histogram of ASCII raster file attributes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_ascii_histogram.py "/group/moniergrp/SpatialCIMIS/Tx/2010/*.asc"
  python plot_ascii_histogram.py "/path/*.asc" --attribute ncols
  python plot_ascii_histogram.py "/path/*.asc" --attribute cellsize_x --output cellsize_hist.png
  python plot_ascii_histogram.py "/path/Tx*.asc" --max-files 1000 --title "Tx Files"
        """
    )
    
    parser.add_argument('pattern', type=str,
                       help='File pattern with wildcard (e.g., "/path/*.asc")')
    parser.add_argument('--attribute', '-a', type=str, default='nrows',
                       choices=['nrows', 'ncols', 'cellsize_x', 'cellsize_y', 
                               'xllcorner', 'yllcorner', 'xurcorner', 'yurcorner', 'nodata'],
                       help='Attribute to plot (default: nrows)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (if not specified, display plot)')
    parser.add_argument('--max-files', '-m', type=int, default=None,
                       help='Maximum number of files to process')
    parser.add_argument('--title', '-t', type=str, default=None,
                       help='Custom title for the plot')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ASCII RASTER HISTOGRAM PLOTTER")
    print("="*70)
    print(f"\nPattern: {args.pattern}")
    print(f"Attribute: {args.attribute}")
    if args.max_files:
        print(f"Max files: {args.max_files}")
    if args.output:
        print(f"Output: {args.output}")
    
    # Collect attribute values
    values, filenames = collect_attributes(args.pattern, args.attribute, args.max_files)
    
    if not values:
        print("\nError: No valid data found")
        sys.exit(1)
    
    # Create histogram
    plot_histogram(values, args.attribute, args.output, args.title)
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()


