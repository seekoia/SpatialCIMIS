#!/usr/bin/env python3
"""
Test script to process just one day of Tx data to debug the processing
"""

import numpy as np
import rasterio
import os

# Paths
data_path = "/group/moniergrp/SpatialCIMIS/ascii/"
variable = 'Tx'

# Grid dimensions
ny = 552
nx = 500

# Test one file
filename = data_path + variable + '.2010-01-01.asc'

print("="*60)
print("Testing single day processing")
print("="*60)

print(f"\nFile: {filename}")
print(f"File exists: {os.path.isfile(filename)}")

if os.path.isfile(filename):
    # Read the file
    src = rasterio.open(filename)
    tmp = src.read(1)
    
    print(f"\nOriginal data:")
    print(f"  Shape: {tmp.shape}")
    print(f"  Dtype: {tmp.dtype}")
    print(f"  NoData value: {src.nodata}")
    print(f"  Min: {tmp.min()}, Max: {tmp.max()}")
    print(f"  Count of nodata (-9999): {np.sum(tmp == -9999)}")
    print(f"  Count of valid data: {np.sum(tmp != -9999)}")
    
    # Test MASK creation logic
    print(f"\n--- MASK CREATION LOGIC ---")
    tmp_mask = tmp.copy()
    tmp_mask[tmp_mask == 0] = -9999
    print(f"After step 1 (convert 0 to -9999):")
    print(f"  Count of -9999: {np.sum(tmp_mask == -9999)}")
    
    tmp_mask[tmp_mask != -9999] = 1
    print(f"After step 2 (convert non-9999 to 1):")
    print(f"  Count of 1: {np.sum(tmp_mask == 1)}")
    print(f"  Count of -9999: {np.sum(tmp_mask == -9999)}")
    
    tmp_mask[tmp_mask == -9999] = 0
    print(f"After step 3 (convert -9999 to 0):")
    print(f"  Count of 1: {np.sum(tmp_mask == 1)}")
    print(f"  Count of 0: {np.sum(tmp_mask == 0)}")
    print(f"  Sum (total valid cells): {tmp_mask.sum()}")
    
    # Test DATA processing logic
    print(f"\n--- DATA PROCESSING LOGIC ---")
    tmp_data = tmp.copy()
    
    # Original approach (wrong - converts 0°C to nodata)
    tmp_data_wrong = tmp.copy()
    tmp_data_wrong[tmp_data_wrong == 0] = -9999
    print(f"WRONG approach (convert 0 to -9999):")
    print(f"  Valid values: {np.sum(tmp_data_wrong != -9999)}")
    print(f"  Lost data: {np.sum(tmp == 0)}")
    
    # Correct approach (preserve 0°C)
    tmp_data_correct = tmp.copy()
    # Don't modify data, just use the original nodata value
    print(f"\nCORRECT approach (preserve 0°C):")
    print(f"  Valid values: {np.sum(tmp_data_correct != -9999)}")
    print(f"  Min temp: {tmp_data_correct[tmp_data_correct != -9999].min():.2f}°C")
    print(f"  Max temp: {tmp_data_correct[tmp_data_correct != -9999].max():.2f}°C")
    
    # Show sample values
    print(f"\nSample values from valid cells (first 10):")
    valid_values = tmp[tmp != -9999]
    print(valid_values[:10])
    
    print("\n" + "="*60)
    print("✓ Test complete!")
    print("="*60)
    
else:
    print(f"ERROR: File not found!")







