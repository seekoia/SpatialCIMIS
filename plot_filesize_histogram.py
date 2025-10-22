#!/usr/bin/env python3
"""
Plot histogram of file sizes for a group of files.

Usage:
    python plot_filesize_histogram.py "<file_pattern>"
    
Examples:
    python plot_filesize_histogram.py "/group/moniergrp/SpatialCIMIS/ascii/ETo*.asc"
    python plot_filesize_histogram.py "/path/to/files/*.nc"
"""

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_file_sizes(pattern):
    """Get file sizes for all files matching the pattern."""
    print(f"\nSearching for files matching: {pattern}")
    
    # Get list of matching files
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"Error: No files found matching pattern: {pattern}")
        return [], []
    
    print(f"Found {len(files)} files")
    
    # Get file sizes
    print("\nReading file sizes...")
    sizes = []
    filenames = []
    
    for i, filepath in enumerate(files, 1):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(files)} files...")
        
        try:
            size = os.path.getsize(filepath)
            sizes.append(size)
            filenames.append(Path(filepath).name)
        except Exception as e:
            print(f"  Warning: Could not get size for {filepath}: {e}")
    
    print(f"  Successfully read {len(sizes)}/{len(files)} file sizes")
    
    return sizes, filenames


def format_size(size_bytes):
    """Format size in bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def plot_histogram(sizes, pattern, output_file=None):
    """Create histogram of file sizes."""
    print("\nCreating histogram...")
    
    if not sizes:
        print("Error: No file sizes to plot")
        return
    
    # Convert to numpy array
    sizes_array = np.array(sizes)
    
    # Calculate statistics
    total_size = np.sum(sizes_array)
    mean_size = np.mean(sizes_array)
    median_size = np.median(sizes_array)
    std_size = np.std(sizes_array)
    min_size = np.min(sizes_array)
    max_size = np.max(sizes_array)
    
    print(f"\nFile Size Statistics:")
    print(f"  Number of files: {len(sizes)}")
    print(f"  Total size: {format_size(total_size)}")
    print(f"  Mean size: {format_size(mean_size)}")
    print(f"  Median size: {format_size(median_size)}")
    print(f"  Std Dev: {format_size(std_size)}")
    print(f"  Min size: {format_size(min_size)}")
    print(f"  Max size: {format_size(max_size)}")
    
    # Determine appropriate unit for plotting
    if max_size > 1024**3:  # GB
        plot_unit = 'GB'
        plot_divisor = 1024**3
    elif max_size > 1024**2:  # MB
        plot_unit = 'MB'
        plot_divisor = 1024**2
    elif max_size > 1024:  # KB
        plot_unit = 'KB'
        plot_divisor = 1024
    else:
        plot_unit = 'B'
        plot_divisor = 1
    
    sizes_scaled = sizes_array / plot_divisor
    mean_scaled = mean_size / plot_divisor
    median_scaled = median_size / plot_divisor
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create histogram
    n, bins, patches = ax.hist(sizes_scaled, bins=50, 
                                color='steelblue', alpha=0.7, 
                                edgecolor='black', linewidth=1.2)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_scaled, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {format_size(mean_size)}')
    ax.axvline(median_scaled, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {format_size(median_size)}')
    
    # Labels and title
    ax.set_xlabel(f'File Size ({plot_unit})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Files)', fontsize=12, fontweight='bold')
    
    # Title with pattern
    pattern_short = Path(pattern).name if len(pattern) > 60 else pattern
    title = f'Distribution of File Sizes\n{pattern_short}\n'
    title += f'({len(sizes)} files, Total: {format_size(total_size)})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(fontsize=11, loc='upper right')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    stats_text = f'Min: {format_size(min_size)}\n'
    stats_text += f'Max: {format_size(max_size)}\n'
    stats_text += f'Std: {format_size(std_size)}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        # Auto-generate filename
        output_file = 'filesize_histogram.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    
    plt.close()
    
    # Print size distribution
    print(f"\nSize Distribution:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        size_p = np.percentile(sizes_array, p)
        print(f"  {p}th percentile: {format_size(size_p)}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python plot_filesize_histogram.py \"<file_pattern>\" [output_file]")
        print("\nExamples:")
        print("  python plot_filesize_histogram.py \"/group/moniergrp/SpatialCIMIS/ascii/ETo*.asc\"")
        print("  python plot_filesize_histogram.py \"/path/*.nc\" output.png")
        sys.exit(1)
    
    pattern = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("="*70)
    print("FILE SIZE HISTOGRAM")
    print("="*70)
    print(f"\nPattern: {pattern}")
    if output_file:
        print(f"Output: {output_file}")
    
    # Get file sizes
    sizes, filenames = get_file_sizes(pattern)
    
    if not sizes:
        print("\nError: No valid file sizes found")
        sys.exit(1)
    
    # Create histogram
    plot_histogram(sizes, pattern, output_file)
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()


