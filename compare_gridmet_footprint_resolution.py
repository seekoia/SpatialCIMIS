#!/usr/bin/env python3
"""
Compare footprint and resolution of different variables from gridMET NetCDF files.
"""

import os
import xarray as xr
import numpy as np
from datetime import datetime

# Dates to check
DATES = [
    '2005-01-15',
    '2010-01-15',
    '2018-01-15',
    '2020-01-15'
]

# Variables to check (gridMET variables found in directory)
VARIABLES = ['pr', 'tmmn', 'tmmx', 'srad', 'vs', 'pet', 'rmax', 'rmin', 'sph']

# Base directory
GRIDMET_DIR = '/group/moniergrp/gridMET/'


def get_netcdf_metadata(filepath, target_date):
    """Read NetCDF file and return spatial metadata for a specific date."""
    try:
        with xr.open_dataset(filepath) as ds:
            # Get coordinate information
            if 'lon' in ds.coords and 'lat' in ds.coords:
                lon = ds.lon.values
                lat = ds.lat.values
                
                # Calculate bounds
                lon_min = float(np.min(lon))
                lon_max = float(np.max(lon))
                lat_min = float(np.min(lat))
                lat_max = float(np.max(lat))
                
                # Calculate resolution
                if len(lon) > 1:
                    lon_res = float(np.abs(lon[1] - lon[0]))
                else:
                    lon_res = 0.0
                    
                if len(lat) > 1:
                    lat_res = float(np.abs(lat[1] - lat[0]))
                else:
                    lat_res = 0.0
                
                # Get dimensions
                nlon = len(lon)
                nlat = len(lat)
                
                # Check if the target date exists in the dataset
                date_exists = False
                if 'day' in ds.coords:
                    # Convert target date to day of year
                    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
                    day_of_year = target_dt.timetuple().tm_yday
                    
                    # Check if this day exists in the dataset
                    if day_of_year <= len(ds.day):
                        date_exists = True
                elif 'time' in ds.coords:
                    # Try to find the date in time coordinate
                    try:
                        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
                        if target_dt in ds.time.values:
                            date_exists = True
                    except:
                        pass
                
                if not date_exists:
                    print(f"    WARNING: Date {target_date} not found in {filepath}")
                
                metadata = {
                    'nlon': nlon,
                    'nlat': nlat,
                    'lon_min': lon_min,
                    'lon_max': lon_max,
                    'lat_min': lat_min,
                    'lat_max': lat_max,
                    'lon_res': lon_res,
                    'lat_res': lat_res,
                    'crs': str(ds.crs) if 'crs' in ds.attrs else 'Unknown',
                    'date_exists': date_exists
                }
                
                return metadata
            else:
                print(f"    WARNING: No lon/lat coordinates found in {filepath}")
                return None
                
    except Exception as e:
        print(f"    ERROR reading {filepath}: {e}")
        return None


def find_gridmet_files(date_str):
    """Find gridMET files for a given date (annual files)."""
    files = {}
    year = date_str.split('-')[0]
    
    # Look for annual files in the gridMET directory
    for var in VARIABLES:
        filepath = os.path.join(GRIDMET_DIR, f"{var}_{year}.nc")
        if os.path.exists(filepath):
            files[var] = filepath
    
    return files


def compare_date(date_str):
    """Compare all variables for a specific date."""
    print(f"\n{'='*120}")
    print(f"Date: {date_str}")
    print(f"{'='*120}")
    
    # Find files for this date
    files = find_gridmet_files(date_str)
    
    if not files:
        print("  No gridMET files found for this date")
        return
    
    print(f"  Found {len(files)} files:")
    for var, filepath in files.items():
        print(f"    {var}: {os.path.basename(filepath)}")
    
    results = {}
    
    for var, filepath in files.items():
        print(f"  Reading {var}...")
        metadata = get_netcdf_metadata(filepath, date_str)
        if metadata:
            results[var] = metadata
    
    if not results:
        print("  No valid files could be read for this date")
        return
    
    # Print comparison table
    print(f"\n{'Variable':<10} {'nlon':<8} {'nlat':<8} {'lon_res':<10} {'lat_res':<10} {'lon_min':<12} {'lon_max':<12} {'lat_min':<12} {'lat_max':<12} {'Date OK':<8}")
    print("-" * 128)
    
    for var in sorted(results.keys()):
        m = results[var]
        date_ok = "Yes" if m.get('date_exists', False) else "No"
        print(f"{var:<10} {m['nlon']:<8} {m['nlat']:<8} {m['lon_res']:<10.6f} {m['lat_res']:<10.6f} "
              f"{m['lon_min']:<12.6f} {m['lon_max']:<12.6f} {m['lat_min']:<12.6f} {m['lat_max']:<12.6f} {date_ok:<8}")
    
    # Check for differences
    if len(results) > 1:
        first_var = list(results.keys())[0]
        first_meta = results[first_var]
        
        differences = []
        for var in results.keys():
            if var == first_var:
                continue
            m = results[var]
            
            if m['nlon'] != first_meta['nlon']:
                differences.append(f"{var}: nlon differs ({m['nlon']} vs {first_meta['nlon']})")
            if m['nlat'] != first_meta['nlat']:
                differences.append(f"{var}: nlat differs ({m['nlat']} vs {first_meta['nlat']})")
            if abs(m['lon_res'] - first_meta['lon_res']) > 1e-6:
                differences.append(f"{var}: lon_res differs ({m['lon_res']:.6f} vs {first_meta['lon_res']:.6f})")
            if abs(m['lat_res'] - first_meta['lat_res']) > 1e-6:
                differences.append(f"{var}: lat_res differs ({m['lat_res']:.6f} vs {first_meta['lat_res']:.6f})")
            if abs(m['lon_min'] - first_meta['lon_min']) > 1e-6:
                differences.append(f"{var}: lon_min differs ({m['lon_min']:.6f} vs {first_meta['lon_min']:.6f})")
            if abs(m['lon_max'] - first_meta['lon_max']) > 1e-6:
                differences.append(f"{var}: lon_max differs ({m['lon_max']:.6f} vs {first_meta['lon_max']:.6f})")
            if abs(m['lat_min'] - first_meta['lat_min']) > 1e-6:
                differences.append(f"{var}: lat_min differs ({m['lat_min']:.6f} vs {first_meta['lat_min']:.6f})")
            if abs(m['lat_max'] - first_meta['lat_max']) > 1e-6:
                differences.append(f"{var}: lat_max differs ({m['lat_max']:.6f} vs {first_meta['lat_max']:.6f})")
        
        if differences:
            print(f"\n  Differences found:")
            for diff in differences:
                print(f"    {diff}")
        else:
            print(f"\n  All variables have identical footprint and resolution")
    
    # Show CRS information
    crs_info = set()
    for var, m in results.items():
        if 'crs' in m:
            crs_info.add(m['crs'])
    
    if crs_info:
        print(f"\n  Coordinate Reference Systems found:")
        for crs in sorted(crs_info):
            print(f"    {crs}")


def main():
    print("gridMET Variable Footprint and Resolution Comparison")
    print("="*120)
    print(f"Checking directory: {GRIDMET_DIR}")
    print(f"Variables: {', '.join(VARIABLES)}")
    print(f"Dates: {', '.join(DATES)}")
    
    for date_str in DATES:
        compare_date(date_str)
    
    print(f"\n{'='*120}")
    print("Comparison complete")


if __name__ == '__main__':
    main()
