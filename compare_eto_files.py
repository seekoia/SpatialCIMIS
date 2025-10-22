#!/usr/bin/env python3
"""
Find and plot two ETo ASCII files with different column counts.

This script searches for ETo files with 500 and 510 columns,
then creates a comparison plot using the ASCII comparison script.
"""

import glob
import rasterio
from pathlib import Path
import subprocess
import sys


def find_files_with_different_dimensions(pattern, max_files=1000):
    """Find files with different column counts."""
    print(f"\nSearching for files matching: {pattern}")
    
    files = sorted(glob.glob(pattern))
    print(f"Total files found: {len(files)}")
    
    if not files:
        print("Error: No files found")
        return {}
    
    print(f"\nChecking dimensions (max {max_files} files)...")
    
    dimension_map = {}  # {ncols: filepath}
    
    for i, filepath in enumerate(files[:max_files]):
        if i % 100 == 0:
            print(f"  Checked {i}/{min(max_files, len(files))} files...")
        
        try:
            with rasterio.open(filepath) as src:
                ncols = src.width
                nrows = src.height
                
                # Store first file with this dimension
                if ncols not in dimension_map:
                    dimension_map[ncols] = filepath
                    print(f"  ✓ Found {ncols} columns: {Path(filepath).name}")
                
                # Stop if we found at least 2 different column counts
                if len(dimension_map) >= 2:
                    print(f"\n  Found {len(dimension_map)} different dimensions, stopping search.")
                    break
        except Exception as e:
            pass
    
    print(f"\nTotal unique column counts found: {len(dimension_map)}")
    for ncols, filepath in sorted(dimension_map.items()):
        with rasterio.open(filepath) as src:
            print(f"  {ncols} columns × {src.height} rows: {Path(filepath).name}")
    
    return dimension_map


def main():
    """Main function."""
    print("="*70)
    print("ETO FILE COMPARISON BY COLUMN COUNT")
    print("="*70)
    
    # Search for files with different dimensions in 2010
    pattern = "/group/moniergrp/SpatialCIMIS/ascii/Eto.2010*.asc"
    
    found_files = find_files_with_different_dimensions(pattern, max_files=1000)
    
    print(f"\n{'='*70}")
    print("SEARCH RESULTS")
    print(f"{'='*70}")
    
    if len(found_files) < 2:
        print("\nWarning: Could not find files with different dimensions")
        print("Found files:")
        for ncols, filepath in found_files.items():
            print(f"  {ncols} columns: {filepath}")
        
        if len(found_files) == 1:
            print("\nOnly found one unique dimension. Cannot compare.")
            sys.exit(1)
        else:
            print("\nNo files found!")
            sys.exit(1)
    else:
        files_list = list(found_files.values())
    
    print(f"\nFiles to compare:")
    for i, filepath in enumerate(files_list, 1):
        with rasterio.open(filepath) as src:
            print(f"  File {i}: {Path(filepath).name}")
            print(f"    Columns: {src.width}, Rows: {src.height}")
            print(f"    Path: {filepath}")
    
    # Run comparison script
    print(f"\n{'='*70}")
    print("CREATING COMPARISON PLOT")
    print(f"{'='*70}\n")
    
    cmd = [
        "python",
        "compare_ascii_footprints.py",
        files_list[0],
        files_list[1]
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n{'='*70}")
        print("SUCCESS!")
        print(f"{'='*70}")
        print("\nComparison plot created: ascii_comparison.png")
        print("Difference plot created: ascii_comparison_difference.png")
    else:
        print(f"\nError: Comparison script failed with exit code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()

