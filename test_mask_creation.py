#!/usr/bin/env python3
"""
Test mask creation with just 2 days to debug the issue
"""

import numpy as np
import rasterio
import os
from spatial_cimis_data_processing import dates_daily
import calendar

data_path = "/group/moniergrp/SpatialCIMIS/ascii/"
variable = 'Tx'
ny = 552
nx = 500

# Test with just 2010 (one year)
yr = 2010
ntime = 365 + calendar.isleap(yr)

# Generate dates
day, month, year = dates_daily(yr, 1, 1, ntime, 0)

# Create array for this year only
data_tmp = np.zeros((ntime, ny, nx))

print("Testing mask creation for first 5 days of 2010...")
print("="*60)

for t in range(5):  # Just first 5 days
    filename = data_path + variable + '.' + str(int(year[t])).zfill(4) + '-' + str(int(month[t])).zfill(2) + '-' + str(int(day[t])).zfill(2) + '.asc'
    
    print(f"\nDay {t}: {int(year[t])}-{int(month[t]):02d}-{int(day[t]):02d}")
    print(f"  File: {os.path.basename(filename)}")
    print(f"  Exists: {os.path.isfile(filename)}")
    
    if os.path.isfile(filename):
        src = rasterio.open(filename)
        tmp = src.read(1)
        
        print(f"  Original: valid={np.sum(tmp != -9999):,}, nodata={np.sum(tmp == -9999):,}")
        
        # Apply mask creation logic
        tmp[tmp == 0] = -9999
        print(f"  After 0→-9999: valid={np.sum(tmp != -9999):,}, nodata={np.sum(tmp == -9999):,}")
        
        tmp[tmp != -9999] = 1
        print(f"  After valid→1: ones={np.sum(tmp == 1):,}, -9999={np.sum(tmp == -9999):,}")
        
        tmp[tmp == -9999] = 0
        print(f"  After -9999→0: ones={np.sum(tmp == 1):,}, zeros={np.sum(tmp == 0):,}")
        print(f"  Sum (should be ~1.68M): {tmp.sum():.0f}")
        
        # Store in data_tmp
        if tmp.shape[0] == ny:
            data_tmp[t, :, :] = tmp
        else:
            data_tmp[t, :, :] = tmp[3:555, 5:505]
        
        print(f"  data_tmp[{t}] sum: {data_tmp[t].sum():.0f}")

# Sum across days
yearly_sum = np.sum(data_tmp, axis=0)
print(f"\n{'='*60}")
print(f"After summing {5} days:")
print(f"  yearly_sum shape: {yearly_sum.shape}")
print(f"  yearly_sum max: {yearly_sum.max():.0f}")
print(f"  yearly_sum sum: {yearly_sum.sum():.0f}")
print(f"  Cells with 5 days of data: {np.sum(yearly_sum == 5):,}")
print(f"  Cells with 0 days of data: {np.sum(yearly_sum == 0):,}")
print(f"  {'='*60}")

if yearly_sum.max() > 0:
    print(f"\n✓ Mask creation logic WORKS!")
else:
    print(f"\n✗ Mask creation logic FAILED - max is 0!")







