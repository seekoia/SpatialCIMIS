#!/usr/bin/env python3
"""
Test script to verify grid standardization works correctly
Tests with files that have different dimensions and resolutions
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import os

# Define target grid
TARGET_GRID = {
    'ncols': 500,
    'nrows': 552,
    'xllcorner': -400000,
    'yllcorner': -650000,
    'cellsize': 2000.0,
    'crs': 'EPSG:3310'
}

def create_target_transform():
    """Create the affine transform for target grid"""
    return from_bounds(
        TARGET_GRID['xllcorner'],
        TARGET_GRID['yllcorner'],
        TARGET_GRID['xllcorner'] + TARGET_GRID['ncols'] * TARGET_GRID['cellsize'],
        TARGET_GRID['yllcorner'] + TARGET_GRID['nrows'] * TARGET_GRID['cellsize'],
        TARGET_GRID['ncols'],
        TARGET_GRID['nrows']
    )

def read_asc_header(filename):
    """Read header from ASCII file"""
    header = {}
    with open(filename, 'r') as f:
        for i in range(6):
            line = f.readline().strip().split()
            if len(line) >= 2:
                key = line[0].lower()
                value = float(line[1]) if '.' in line[1] or 'e' in line[1].lower() else int(line[1])
                header[key] = value
    return header

def test_file(filename, description):
    """Test reading and reprojecting a single file"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"File: {os.path.basename(filename)}")
    print(f"{'='*60}")
    
    if not os.path.exists(filename):
        print(f"  ❌ File not found: {filename}")
        return False
    
    try:
        # Read header
        header = read_asc_header(filename)
        print(f"  Source grid:")
        print(f"    Dimensions: {header['ncols']} × {header['nrows']}")
        print(f"    Cell size: {header['cellsize']} m")
        print(f"    Corner: ({header['xllcorner']}, {header['yllcorner']})")
        
        # Read with rasterio
        with rasterio.open(filename) as src:
            data = src.read(1)
            src_transform = src.transform
            
            print(f"  Source data:")
            print(f"    Shape: {data.shape}")
            print(f"    Valid values: {np.sum(data != -9999)}")
            print(f"    Min/Max: {data[data != -9999].min():.2f} / {data[data != -9999].max():.2f}")
            
            # Create target grid
            target_transform = create_target_transform()
            target_shape = (TARGET_GRID['nrows'], TARGET_GRID['ncols'])
            target_data = np.full(target_shape, -9999, dtype=np.float32)
            
            # Reproject
            reproject(
                source=data,
                destination=target_data,
                src_transform=src_transform,
                src_crs=TARGET_GRID['crs'],
                dst_transform=target_transform,
                dst_crs=TARGET_GRID['crs'],
                resampling=Resampling.bilinear,
                src_nodata=-9999,
                dst_nodata=-9999
            )
            
            print(f"  Target data:")
            print(f"    Shape: {target_data.shape}")
            print(f"    Valid values: {np.sum(target_data != -9999)}")
            if np.sum(target_data != -9999) > 0:
                print(f"    Min/Max: {target_data[target_data != -9999].min():.2f} / {target_data[target_data != -9999].max():.2f}")
            
            # Check if reprojection was needed
            if data.shape == target_shape and src_transform == target_transform:
                print(f"  ✓ No reprojection needed (already on target grid)")
            else:
                print(f"  ✓ Successfully reprojected to target grid")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Run tests on different file types"""
    
    print("="*60)
    print("Grid Standardization Test Suite")
    print("="*60)
    print(f"\nTarget Grid Configuration:")
    print(f"  Dimensions: {TARGET_GRID['ncols']} × {TARGET_GRID['nrows']}")
    print(f"  Cell size: {TARGET_GRID['cellsize']} m")
    print(f"  Corner: ({TARGET_GRID['xllcorner']}, {TARGET_GRID['yllcorner']})")
    print(f"  CRS: {TARGET_GRID['crs']}")
    
    data_path = "/group/moniergrp/SpatialCIMIS/ascii/"
    
    # Test cases covering different scenarios
    test_cases = [
        # Standard grid (no reprojection needed)
        (f"{data_path}Tx.2024-01-01.asc", "Standard 500×552 grid (2km)"),
        
        # Different extent (510×560)
        (f"{data_path}ETo.2012-06-15.asc", "Larger extent 510×560 grid (2km)"),
        
        # Different resolution (2000×2208 at 500m)
        (f"{data_path}Rs.2010-01-15.asc", "High resolution 2000×2208 grid (500m)"),
        
        # Standard after resolution change
        (f"{data_path}Rs.2010-08-15.asc", "Standard 500×552 after resolution change"),
    ]
    
    results = []
    for filename, description in test_cases:
        success = test_file(filename, description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for description, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {description}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Grid standardization is working correctly.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check errors above.")

if __name__ == "__main__":
    main()

