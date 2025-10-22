#!/usr/bin/env python3
"""
Quick test to verify grid standardization works on Rs 2010
Tests the 500m → 2km resolution change
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spatial_cimis_processing import (
    read_and_reproject_asc, 
    create_target_transform, 
    TARGET_GRID,
    read_asc_header
)

def test_grid_standardization():
    """Test that files with different resolutions get reprojected correctly"""
    
    print("="*60)
    print("Quick Test: Rs 2010 Grid Standardization")
    print("="*60)
    
    data_path = "/group/moniergrp/SpatialCIMIS/ascii/"
    target_transform = create_target_transform()
    target_shape = (TARGET_GRID['nrows'], TARGET_GRID['ncols'])
    
    print(f"\nTarget Grid: {TARGET_GRID['ncols']}×{TARGET_GRID['nrows']} @ {TARGET_GRID['cellsize']}m")
    print()
    
    # Test files
    test_files = [
        ("Rs.2010-01-15.asc", "January (500m resolution)"),
        ("Rs.2010-07-15.asc", "July (500m resolution)"),
        ("Rs.2010-08-15.asc", "August (2km resolution)"),
        ("Rs.2010-12-15.asc", "December (2km resolution)"),
    ]
    
    results = []
    
    for filename, description in test_files:
        filepath = os.path.join(data_path, filename)
        
        print(f"Testing: {description}")
        print(f"  File: {filename}")
        
        if not os.path.exists(filepath):
            print(f"  ✗ File not found!")
            results.append(False)
            continue
        
        try:
            # Read header
            header = read_asc_header(filepath)
            print(f"  Source: {header['ncols']}×{header['nrows']} @ {header['cellsize']}m")
            
            # Reproject
            data = read_and_reproject_asc(filepath, target_transform, target_shape)
            
            # Check results
            valid_cells = np.sum(data != -9999)
            print(f"  Output: {data.shape[1]}×{data.shape[0]} (target grid)")
            print(f"  Valid cells: {valid_cells:,}")
            
            if len(data[data != -9999]) > 0:
                print(f"  Value range: {data[data != -9999].min():.2f} to {data[data != -9999].max():.2f} MJ/m²/day")
            
            # Verify shape
            if data.shape == target_shape:
                print(f"  ✓ Correctly reprojected to target grid")
                results.append(True)
            else:
                print(f"  ✗ Wrong shape: {data.shape} != {target_shape}")
                results.append(False)
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append(False)
        
        print()
    
    # Summary
    print("="*60)
    print("Summary")
    print("="*60)
    
    for (filename, description), success in zip(test_files, results):
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {description}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ Grid standardization is working correctly!")
        print("  Files with different resolutions are properly reprojected to target grid.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    exit(test_grid_standardization())

