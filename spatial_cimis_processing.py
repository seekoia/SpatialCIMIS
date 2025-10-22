#!/usr/bin/env python3
"""
Spatial CIMIS Data Processing Script with Grid Standardization

This script processes Spatial CIMIS data from ASCII files into NetCDF format.
It handles ASCII files with varying grid dimensions, resolutions, and extents by
reprojecting all data to a common target grid.

Key improvements:
- Automatically detects and handles different grid dimensions and resolutions
- Uses rasterio.warp.reproject for proper spatial regridding
- Supports multiple variables (Tx, ETo, Tn, Rs, etc.)

Author: UC Davis Global Environmental Change Lab
Date: 2024
"""

import numpy as np
import netCDF4 as nc
import math
import datetime as dt
from datetime import timedelta, date
import calendar
import xarray as xr
import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from netCDF4 import date2num, num2date
from datetime import datetime
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
import pyproj
from pyproj import Transformer
import matplotlib.patches as mpatches
try:
    import cartopy.crs as ccrs
except ImportError:
    ccrs = None
    print("Warning: cartopy not available. Some plotting features may be limited.")
import pandas as pd
import dask
import argparse
import sys


# Define standard target grid parameters
TARGET_GRID = {
    'ncols': 500,
    'nrows': 552,
    'xllcorner': -400000,
    'yllcorner': -650000,
    'cellsize': 2000.0,
    'crs': 'EPSG:3310'  # California Albers
}


def read_asc_header(filename):
    """
    Read header information from ASCII grid file.
    
    Parameters:
    -----------
    filename : str
        Path to ASCII file
        
    Returns:
    --------
    dict : Dictionary with header parameters
    """
    header = {}
    with open(filename, 'r') as f:
        for i in range(6):
            line = f.readline().strip().split()
            if len(line) >= 2:
                key = line[0].lower()
                value = float(line[1]) if '.' in line[1] or 'e' in line[1].lower() else int(line[1])
                header[key] = value
    return header


def create_target_transform():
    """
    Create the affine transform for the target grid.
    
    Returns:
    --------
    rasterio.Affine : Transform for target grid
    """
    return from_bounds(
        TARGET_GRID['xllcorner'],
        TARGET_GRID['yllcorner'],
        TARGET_GRID['xllcorner'] + TARGET_GRID['ncols'] * TARGET_GRID['cellsize'],
        TARGET_GRID['yllcorner'] + TARGET_GRID['nrows'] * TARGET_GRID['cellsize'],
        TARGET_GRID['ncols'],
        TARGET_GRID['nrows']
    )


def read_and_reproject_asc(filename, target_transform, target_shape):
    """
    Read ASCII file and reproject to target grid if needed.
    
    Parameters:
    -----------
    filename : str
        Path to ASCII file
    target_transform : rasterio.Affine
        Target grid transform
    target_shape : tuple
        Target grid shape (nrows, ncols)
        
    Returns:
    --------
    numpy.ndarray : Data reprojected to target grid
    """
    try:
        # Read with rasterio
        with rasterio.open(filename) as src:
            data = src.read(1)
            src_transform = src.transform
            src_crs = src.crs if src.crs else TARGET_GRID['crs']
            
            # Check if regridding is needed
            needs_reproject = (
                data.shape != target_shape or
                src_transform != target_transform
            )
            
            if not needs_reproject:
                # Data already on target grid
                return data
            
            # Initialize target array
            target_data = np.full(target_shape, -9999, dtype=np.float32)
            
            # Reproject to target grid
            reproject(
                source=data,
                destination=target_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=target_transform,
                dst_crs=TARGET_GRID['crs'],
                resampling=Resampling.bilinear,
                src_nodata=-9999,
                dst_nodata=-9999
            )
            
            return target_data
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return np.full(target_shape, -9999, dtype=np.float32)


def dates_daily(y0, m0, d0, mtot, noleap):
    """
    Function that calculates arrays of dates (hour, day, month, year)
    
    Parameters:
    -----------
    y0 : int
        Initial year
    m0 : int
        Initial month
    d0 : int
        Initial day
    mtot : int
        Total number of elements in the time series
    noleap : int
        Flag to indicate whether or not leap years should be considered.
        noleap = 0 --> time series contains Feb 29
        noleap = 1 --> time series does not contain Feb 29
    
    Returns:
    --------
    day : numpy.ndarray
        Array with values of days
    month : numpy.ndarray
        Array with values of months
    year : numpy.ndarray
        Array with values of years
    """
    day = np.zeros((mtot))
    month = np.zeros((mtot))
    year = np.zeros((mtot))
    start_date = date(y0, m0, d0)
    deltad = timedelta(days=1)
    single_date = start_date
    dt = 0
    
    while dt < mtot:
        day[dt] = single_date.day
        month[dt] = single_date.month
        year[dt] = single_date.year
        
        if noleap == 0:
            single_date = single_date + deltad
        if noleap == 1:
            if year[dt] % 4 != 0:
                single_date = single_date + deltad
            if year[dt] % 4 == 0:
                if month[dt] != 2:
                    single_date = single_date + deltad
                if month[dt] == 2:
                    if day[dt] < 28:
                        single_date = single_date + deltad
                    if day[dt] == 28:
                        single_date = single_date + 2 * deltad
        dt = dt + 1
    
    return day, month, year


def dates_of_year(y0, noleap):
    """
    Function that calculates arrays of dates for a full year
    
    Parameters:
    -----------
    y0 : int
        Year
    noleap : int
        Flag to indicate whether or not leap years should be considered.
        noleap = 0 --> time series contains Feb 29
        noleap = 1 --> time series does not contain Feb 29
    
    Returns:
    --------
    day : numpy.ndarray
        Array with values of days
    month : numpy.ndarray
        Array with values of months
    year : numpy.ndarray
        Array with values of years
    """
    if noleap == 0:
        ntime = 365 + calendar.isleap(y0)
    else:
        ntime = 365
        
    start_date = date(y0, 1, 1)
    day = np.zeros((ntime))
    month = np.zeros((ntime))
    year = np.zeros((ntime))
    deltad = timedelta(days=1)
    single_date = start_date
    dt = 0
    
    while dt < ntime:
        day[dt] = single_date.day
        month[dt] = single_date.month
        year[dt] = single_date.year
        
        if noleap == 0:
            single_date = single_date + deltad
        if noleap == 1:
            if year[dt] % 4 != 0:
                single_date = single_date + deltad
            if year[dt] % 4 == 0:
                if month[dt] != 2:
                    single_date = single_date + deltad
                if month[dt] == 2:
                    if day[dt] < 28:
                        single_date = single_date + deltad
                    if day[dt] == 28:
                        single_date = single_date + 2 * deltad
        dt = dt + 1
    
    return day, month, year


def create_variable_mask(data_path, netcdf_path, variable, yr_start=2004, yr_end=2024):
    """
    Create a mask for grid cells that have sufficient data coverage (90% non-NaN values).
    Uses standardized grid reprojection.
    
    Parameters:
    -----------
    data_path : str
        Path to ASCII data files
    netcdf_path : str
        Path to output NetCDF files
    variable : str
        Variable name (e.g., 'Tx', 'ETo', 'Tn', 'Rs')
    yr_start : int
        Start year for processing (default: 2004)
    yr_end : int
        End year for processing (default: 2024)
    """
    print(f"Creating {variable} mask with grid standardization...")
    
    years = range(yr_start, yr_end + 1)
    nyr = len(years)
    ny = TARGET_GRID['nrows']
    nx = TARGET_GRID['ncols']
    
    # Create target transform
    target_transform = create_target_transform()
    target_shape = (ny, nx)
    
    # Create an empty 3D array to store the data
    data = np.zeros((nyr, ny, nx))
    
    # Loop over years to create one netcdf file per year
    for i in range(nyr):
        yr = years[i]
        print(f"  Processing year {yr}...")
        
        # Calculate the number of days in the dataset
        ntime = 365 + calendar.isleap(yr)
        
        # Generate daily dates for the year
        day, month, year = dates_daily(yr, 1, 1, ntime, 0)
        
        # Create an empty array to store daily data
        data_tmp = np.zeros((ntime, ny, nx))
        
        for t in range(ntime):
            # Construct the filename for the ASCII data file
            filename = (data_path + variable + '.' + str(int(year[t])).zfill(4) + '-' + 
                       str(int(month[t])).zfill(2) + '-' + str(int(day[t])).zfill(2) + '.asc')
            
            # Check if the file exists
            if os.path.isfile(filename):
                tmp = read_and_reproject_asc(filename, target_transform, target_shape)
                tmp[tmp == 0] = -9999
                tmp[tmp != -9999] = 1
                tmp[tmp == -9999] = 0
                data_tmp[t, :, :] = tmp
            else:
                data_tmp[t, :, :] = np.zeros((ny, nx))
        
        # Sum the daily data to obtain yearly data
        data[i, :, :] = np.sum(data_tmp, axis=0)
    
    # Calculate a mask by summing the yearly data and applying a threshold
    mask = np.sum(data, axis=0)
    mask = mask / nyr
    
    # Adjust threshold based on data availability
    threshold = 350 if nyr > 1 else max(50, mask.max() * 0.5)
    print(f"  Using mask threshold: {threshold:.0f} days (max available: {mask.max():.0f})")
    
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 1
    
    # Create coordinate arrays for target grid
    # X goes left to right (low to high)
    x = np.arange(nx) * TARGET_GRID['cellsize'] + TARGET_GRID['xllcorner'] + TARGET_GRID['cellsize']/2
    # Y goes top to bottom (high to low) to match rasterio convention
    y_top = TARGET_GRID['yllcorner'] + TARGET_GRID['nrows'] * TARGET_GRID['cellsize']
    y = y_top - np.arange(ny) * TARGET_GRID['cellsize'] - TARGET_GRID['cellsize']/2
    
    # Create a mesh grid for coordinates
    xl, yl = np.meshgrid(x, y)
    
    # Use PyProj to transform coordinates to latitude and longitude
    transformer = Transformer.from_crs(TARGET_GRID['crs'], "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xl, yl)
    
    # Get the current UTC time as a string
    dtnow = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Create an xarray dataset to store the data and metadata
    xrds = xr.Dataset(
        coords=dict(
            y=(['y'], y),
            x=(['x'], x)
        ),
        data_vars=dict(
            lat=(['y', 'x'], lat),
            lon=(['y', 'x'], lon),
            mask=(['y', 'x'], mask)
        ),
        attrs=dict(
            title=f'Spatial CIMIS {variable} Mask',
            summary=f'Mask for the daily {variable} at a 2 km spatial resolution over California obtained from the California Irrigation Management Information System (CIMIS) Spatial Model. All data has been regridded to a common grid.',
            source='Spatial CIMIS data',
            processing_level='Processed with grid standardization',
            creator_type='group',
            creator_name='UC Davis Global Environmental Change Lab',
            date_created=dtnow,
            history=f'File created at {dtnow} using xarray in Python with rasterio reprojection',
            geospatial_bounds_crs='EPSG:3310'
        )
    )
    
    # Assign attributes to variables
    xrds['y'].attrs = {
        'standard_name': 'projection_y_coordinate',
        'long_name': 'Y coordinate (North)',
        'units': 'meter',
        'coverage_content_type': 'coordinate'
    }
    xrds['x'].attrs = {
        'standard_name': 'projection_x_coordinate',
        'long_name': 'X coordinate (East)',
        'units': 'meter',
        'coverage_content_type': 'coordinate'
    }
    xrds['lat'].attrs = {
        'standard_name': 'latitude',
        'long_name': 'Latitude',
        'units': 'degree_north',
        'coverage_content_type': 'referenceInformation'
    }
    xrds['lon'].attrs = {
        'standard_name': 'longitude',
        'long_name': 'Longitude',
        'units': 'degree_east',
        'coverage_content_type': 'referenceInformation'
    }
    xrds['mask'].attrs = {
        'standard_name': 'binary_mask',
        'long_name': f'Mask of grid cells with Spatial CIMIS {variable} data; 1 = data, 0 = no data.',
        'units': '1',
        'coverage_content_type': 'thematicClassification'
    }
    
    # Save the xarray dataset to a NetCDF file
    xrds.to_netcdf(netcdf_path + f'spatial_cimis_{variable.lower()}_mask.nc')
    print(f'{variable} mask creation completed!')


def process_variable_data(data_path, netcdf_path, variable, yr_start=2004, yr_end=2023, max_gap_days=7, use_mask=False):
    """
    Process variable data: read ASCII files, reproject to common grid, optionally apply mask,
    interpolate missing values, and output as NetCDF files.
    
    Parameters:
    -----------
    data_path : str
        Path to ASCII data files
    netcdf_path : str
        Path to output NetCDF files
    variable : str
        Variable name (e.g., 'Tx', 'ETo', 'Tn', 'Rs')
    yr_start : int
        Start year for processing (default: 2004)
    yr_end : int
        End year for processing (default: 2023)
    max_gap_days : int
        Maximum number of consecutive missing days to interpolate over (default: 7)
    use_mask : bool
        Whether to apply a mask (default: False)
    """
    print(f"Processing {variable} data with grid standardization...")
    
    years = range(yr_start, yr_end + 1)
    nyr = len(years)
    ny = TARGET_GRID['nrows']
    nx = TARGET_GRID['ncols']
    
    # Create target transform
    target_transform = create_target_transform()
    target_shape = (ny, nx)
    
    # Try to load mask if requested and available
    mask = None
    if use_mask:
        mask_file = netcdf_path + f'spatial_cimis_{variable.lower()}_mask.nc'
        if os.path.exists(mask_file):
            print(f"  Loading mask from {mask_file}")
            ds = xr.open_dataset(mask_file)
            mask = ds['mask']
        else:
            print(f"  Warning: Mask file not found ({mask_file}). Processing without mask.")
            use_mask = False
    else:
        print(f"  Processing without mask (use_mask=False)")
    
    # Loop over years to create one NetCDF file per year
    for i in range(nyr):
        yr = years[i]
        print(f"  Processing year {yr}...")
        
        # Scan available files for this year
        available_dates = []
        ndays_year = 365 + calendar.isleap(yr)
        day_yr, month_yr, year_yr = dates_daily(yr, 1, 1, ndays_year, 0)
        
        for t in range(ndays_year):
            filename = (data_path + variable + '.' + str(int(year_yr[t])).zfill(4) + '-' + 
                       str(int(month_yr[t])).zfill(2) + '-' + str(int(day_yr[t])).zfill(2) + '.asc')
            if os.path.isfile(filename):
                available_dates.append(t)
        
        if len(available_dates) == 0:
            print(f"  Warning: No data files found for year {yr}. Skipping.")
            continue
        
        print(f"    Found {len(available_dates)} days with data out of {ndays_year} days")
        
        # Determine if we should use a padded window for interpolation
        # Only use 3-year window if:
        # 1. Good coverage (>50%)
        # 2. Processing multiple years (more context for interpolation)
        use_padding = (len(available_dates) > ndays_year * 0.5) and (nyr > 1)
        
        if use_padding:
            print(f"    Using 3-year window for interpolation (better edge handling)")
            ntime = 3 * 365 + calendar.isleap(yr - 1) + calendar.isleap(yr) + calendar.isleap(yr + 1)
            day, month, year = dates_daily(yr - 1, 1, 1, ntime, 0)
        else:
            reason = f"{len(available_dates)/ndays_year*100:.1f}% coverage" if nyr == 1 else "partial coverage"
            print(f"    Using single year ({reason})")
            ntime = ndays_year
            day, month, year = day_yr, month_yr, year_yr
        
        # Create an empty array to store daily data
        data = np.zeros((ntime, ny, nx))
        
        for t in range(ntime):
            # Construct the filename for the ASCII data file
            filename = (data_path + variable + '.' + str(int(year[t])).zfill(4) + '-' + 
                       str(int(month[t])).zfill(2) + '-' + str(int(day[t])).zfill(2) + '.asc')
            
            # Check if the file exists
            if os.path.isfile(filename):
                tmp = read_and_reproject_asc(filename, target_transform, target_shape)
                tmp[tmp == 0] = -9999
                # Apply mask only if requested
                if use_mask and mask is not None:
                    tmp[mask == 0] = -9999
                data[t, :, :] = tmp
            else:
                data[t, :, :] = np.full((ny, nx), -9999)
        
        # Create coordinate arrays for target grid
        # X goes left to right (low to high)
        x = np.arange(nx) * TARGET_GRID['cellsize'] + TARGET_GRID['xllcorner'] + TARGET_GRID['cellsize']/2
        # Y goes top to bottom (high to low) to match rasterio convention
        y_top = TARGET_GRID['yllcorner'] + TARGET_GRID['nrows'] * TARGET_GRID['cellsize']
        y = y_top - np.arange(ny) * TARGET_GRID['cellsize'] - TARGET_GRID['cellsize']/2
        
        xl, yl = np.meshgrid(x, y)
        transformer = Transformer.from_crs(TARGET_GRID['crs'], "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xl, yl)
        
        # Get the current UTC time as a string
        dtnow = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Define time units and create time values
        time_units = "days since 2004-01-01"
        dates = [dt.datetime(int(i), int(j), int(k)) for i, j, k in zip(year, month, day)]
        times = date2num(dates, time_units)
        
        # Create a new xarray dataset
        xrds = xr.Dataset(
            coords=dict(
                time=(['time'], times),
                y=(['y'], y),
                x=(['x'], x)
            ),
            data_vars=dict(
                lat=(['y', 'x'], lat),
                lon=(['y', 'x'], lon),
                **{variable: (['time', 'y', 'x'], data)}
            ),
            attrs=dict(
                title=f'Spatial CIMIS {variable}',
                summary=f'Daily {variable} at a 2 km spatial resolution over California. All data regridded to common grid.',
                source='Spatial CIMIS data',
                processing_level='Processed with grid standardization',
                creator_type='group',
                creator_name='UC Davis Global Environmental Change Lab',
                date_created=dtnow,
                history=f'File created at {dtnow} using xarray and rasterio reprojection',
                geospatial_bounds_crs='EPSG:3310'
            )
        )
        
        # Apply mask to exclude no data values
        xrds[variable] = xrds[variable].where(xrds[variable] != -9999.)
        
        # Interpolate only small gaps
        print(f"    Interpolating gaps up to {max_gap_days} days...")
        try:
            xrds[variable] = xrds[variable].interpolate_na(
                dim="time", 
                method="linear", 
                limit=max_gap_days,
                use_coordinate=True
            )
        except:
            try:
                xrds[variable] = xrds[variable].interpolate_na(
                    dim="time", 
                    method="linear",
                    limit=max_gap_days
                )
            except:
                print(f"    Warning: Could not interpolate. Keeping NaN values.")
                pass
        
        # Assign attributes to variables
        xrds['time'].attrs = {
            'standard_name': 'time',
            'long_name': 'Time',
            'units': time_units,
            'calendar': 'gregorian',
            'coverage_content_type': 'coordinate'
        }
        xrds['y'].attrs = {
            'standard_name': 'projection_y_coordinate',
            'long_name': 'Y coordinate (North)',
            'units': 'meter',
            'coverage_content_type': 'coordinate'
        }
        xrds['x'].attrs = {
            'standard_name': 'projection_x_coordinate',
            'long_name': 'X coordinate (East)',
            'units': 'meter',
            'coverage_content_type': 'coordinate'
        }
        xrds['lat'].attrs = {
            'standard_name': 'latitude',
            'long_name': 'Latitude',
            'units': 'degree_north',
            'coverage_content_type': 'referenceInformation'
        }
        xrds['lon'].attrs = {
            'standard_name': 'longitude',
            'long_name': 'Longitude',
            'units': 'degree_east',
            'coverage_content_type': 'referenceInformation'
        }
        
        # Set variable-specific attributes
        if variable == 'ETo':
            xrds[variable].attrs = {
                'standard_name': 'reference_evapotranspiration',
                'long_name': 'Daily Reference Evapotranspiration (ETo)',
                'units': 'mm/day',
                'coverage_content_type': 'physicalMeasurement'
            }
        elif variable == 'Tx':
            xrds[variable].attrs = {
                'standard_name': 'air_temperature',
                'long_name': 'Daily Maximum Temperature (Tx)',
                'units': 'degrees_C',
                'coverage_content_type': 'physicalMeasurement'
            }
        elif variable == 'Tn':
            xrds[variable].attrs = {
                'standard_name': 'air_temperature',
                'long_name': 'Daily Minimum Temperature (Tn)',
                'units': 'degrees_C',
                'coverage_content_type': 'physicalMeasurement'
            }
        elif variable == 'Rs':
            xrds[variable].attrs = {
                'standard_name': 'surface_downwelling_shortwave_flux_in_air',
                'long_name': 'Daily Solar Radiation (Rs)',
                'units': 'MJ/m2/day',
                'coverage_content_type': 'physicalMeasurement'
            }
        else:
            xrds[variable].attrs = {
                'standard_name': 'unknown',
                'long_name': f'Daily {variable}',
                'units': 'unknown',
                'coverage_content_type': 'physicalMeasurement'
            }
        
        # Select subset for current year
        if use_padding:
            ds = xrds.isel(time=range(365 + calendar.isleap(yr - 1), 
                                      365 + calendar.isleap(yr - 1) + 365 + calendar.isleap(yr)))
        else:
            ds = xrds
        
        # Save the dataset
        ds.to_netcdf(netcdf_path + f'spatial_cimis_{variable.lower()}_{yr}.nc')
        print(f'    Completed year {yr}')
    
    print(f'{variable} data processing completed!')


def create_2d_files(netcdf_path, output_path, variable, start_year=2021, end_year=2023):
    """
    Create 2D (grid, time) daily variable NetCDF files.
    
    Parameters:
    -----------
    netcdf_path : str
        Path to input NetCDF files
    output_path : str
        Path to output NetCDF files
    variable : str
        Variable name (e.g., 'Tx', 'ETo', 'Tn', 'Rs')
    start_year : int
        Start year for processing (default: 2021)
    end_year : int
        End year for processing (default: 2023)
    """
    print(f"Creating 2D {variable} files...")
    
    for yr in range(start_year, end_year + 1):
        ds = xr.open_dataset(netcdf_path + f'spatial_cimis_{variable.lower()}_{yr}.nc')
        var_data = ds[variable]
        var_2d = var_data.stack(grid=("y", "x"))
        data = var_2d.transpose('grid', 'time')
        
        ncfile = Dataset(output_path + f'spatial_cimis_{variable.lower()}_2d_{yr}.nc', 
                        mode='w', format='NETCDF4_CLASSIC')
        
        grid_dim = ncfile.createDimension('grid', data.shape[0])
        time_dim = ncfile.createDimension('time', data.shape[1])
        
        ncfile.title = f'Spatial CIMIS {variable}'
        ncfile.description = f'Daily {variable} at 2 km resolution. Data regridded to common grid.'
        ncfile.author = 'UC Davis Global Environmental Change Lab'
        ncfile.history = 'Created ' + datetime.today().strftime('%Y-%m-%d')
        
        grid = ncfile.createVariable('grid', np.float32, ('grid',))
        time = ncfile.createVariable('time', np.float64, ('time',))
        time.units = 'days since 2004-01-01'
        time.long_name = 'time'
        
        var = ncfile.createVariable(variable, np.float64, ('grid', 'time'), fill_value=-9999)
        
        if variable == 'ETo':
            var.units = 'mm/day'
            var.standard_name = 'reference_evapotranspiration'
            var.description = 'Daily Reference Evapotranspiration'
        elif variable == 'Tx':
            var.units = 'degrees_C'
            var.standard_name = 'air_temperature'
            var.description = 'Daily Maximum Temperature'
        elif variable == 'Tn':
            var.units = 'degrees_C'
            var.standard_name = 'air_temperature'
            var.description = 'Daily Minimum Temperature'
        elif variable == 'Rs':
            var.units = 'MJ/m2/day'
            var.standard_name = 'surface_downwelling_shortwave_flux_in_air'
            var.description = 'Daily Solar Radiation'
        else:
            var.units = 'unknown'
            var.standard_name = 'unknown'
            var.description = f'Daily {variable}'
        
        grid[:] = range(data.shape[0])
        time[:] = range(data.shape[1])
        var[:, :] = data.to_numpy()
        
        ncfile.close()
    
    print(f'2D {variable} files creation completed!')


def create_daily_climatology(netcdf_path, output_path, variable):
    """Create daily climatology from variable data."""
    print(f"Creating daily climatology for {variable}...")
    
    ds = xr.open_mfdataset(netcdf_path + f'spatial_cimis_{variable.lower()}_20*.nc', 
                           combine='nested', concat_dim="time")
    var_data = ds[variable]
    
    try:
        var_daily_clim = var_data.groupby('time.dayofyear').mean(dim='time')
    except AttributeError:
        var_daily_clim = var_data.groupby('time.dt.dayofyear').mean(dim='time')
    
    var_daily_clim_1d = var_daily_clim.stack(grid=("y", "x"))
    data = var_daily_clim_1d.transpose('grid', 'dayofyear')
    
    ncfile = Dataset(output_path + f'spatial_cimis_{variable.lower()}_daily_clim_2d_2004-2022.nc', 
                    mode='w', format='NETCDF4_CLASSIC')
    
    grid_dim = ncfile.createDimension('grid', data.shape[0])
    time_dim = ncfile.createDimension('time', data.shape[1])
    
    ncfile.title = f'Daily Annual Cycle of Spatial CIMIS {variable}'
    ncfile.description = f'Daily {variable} annual cycle at 2 km resolution'
    ncfile.author = 'UC Davis Global Environmental Change Lab'
    ncfile.history = 'Created ' + datetime.today().strftime('%Y-%m-%d')
    
    grid = ncfile.createVariable('grid', np.float32, ('grid',))
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'day of year'
    time.long_name = 'time'
    
    var = ncfile.createVariable(variable, np.float64, ('grid', 'time'), fill_value=-9999)
    
    if variable == 'ETo':
        var.units = 'mm/day'
        var.standard_name = 'reference_evapotranspiration'
        var.description = 'Daily Reference Evapotranspiration'
    elif variable == 'Tx':
        var.units = 'degrees_C'
        var.standard_name = 'air_temperature'
        var.description = 'Daily Maximum Temperature'
    elif variable == 'Tn':
        var.units = 'degrees_C'
        var.standard_name = 'air_temperature'
        var.description = 'Daily Minimum Temperature'
    elif variable == 'Rs':
        var.units = 'MJ/m2/day'
        var.standard_name = 'surface_downwelling_shortwave_flux_in_air'
        var.description = 'Daily Solar Radiation'
    else:
        var.units = 'unknown'
        var.standard_name = 'unknown'
        var.description = f'Daily {variable}'
    
    grid[:] = range(data.shape[0])
    time[:] = range(366)
    var[:, :] = data.to_numpy()
    
    ncfile.close()
    print(f'Daily climatology for {variable} completed!')


def create_monthly_climatology(netcdf_path, output_path, variable):
    """Create monthly climatology from variable data."""
    print(f"Creating monthly climatology for {variable}...")
    
    ds = xr.open_mfdataset(netcdf_path + f'spatial_cimis_{variable.lower()}_20*.nc', 
                           combine='nested', concat_dim="time")
    var_data = ds[variable]
    
    try:
        var_monthly_clim = var_data.groupby('time.month').mean(dim='time')
    except AttributeError:
        var_monthly_clim = var_data.groupby('time.dt.month').mean(dim='time')
    
    var_monthly_clim_1d = var_monthly_clim.stack(grid=("y", "x"))
    data = var_monthly_clim_1d.transpose('grid', 'month')
    
    ncfile = Dataset(output_path + f'spatial_cimis_{variable.lower()}_monthly_clim_2d_2004-2022.nc', 
                    mode='w', format='NETCDF4_CLASSIC')
    
    grid_dim = ncfile.createDimension('grid', data.shape[0])
    time_dim = ncfile.createDimension('time', data.shape[1])
    
    ncfile.title = f'Monthly Annual Cycle of Spatial CIMIS {variable}'
    ncfile.description = f'Monthly {variable} annual cycle at 2 km resolution'
    ncfile.author = 'UC Davis Global Environmental Change Lab'
    ncfile.history = 'Created ' + datetime.today().strftime('%Y-%m-%d')
    
    grid = ncfile.createVariable('grid', np.float32, ('grid',))
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'month of year'
    time.long_name = 'time'
    
    var = ncfile.createVariable(variable, np.float64, ('grid', 'time'), fill_value=-9999)
    
    if variable == 'ETo':
        var.units = 'mm/day'
        var.standard_name = 'reference_evapotranspiration'
        var.description = 'Monthly Reference Evapotranspiration'
    elif variable == 'Tx':
        var.units = 'degrees_C'
        var.standard_name = 'air_temperature'
        var.description = 'Monthly Maximum Temperature'
    elif variable == 'Tn':
        var.units = 'degrees_C'
        var.standard_name = 'air_temperature'
        var.description = 'Monthly Minimum Temperature'
    elif variable == 'Rs':
        var.units = 'MJ/m2/day'
        var.standard_name = 'surface_downwelling_shortwave_flux_in_air'
        var.description = 'Monthly Solar Radiation'
    else:
        var.units = 'unknown'
        var.standard_name = 'unknown'
        var.description = f'Monthly {variable}'
    
    grid[:] = range(data.shape[0])
    time[:] = range(12)
    var[:, :] = data.to_numpy()
    
    ncfile.close()
    print(f'Monthly climatology for {variable} completed!')


def load_config(config_file='scp_config.txt'):
    """Load configuration from a text file."""
    config = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if value.lower() in ['true', 'false']:
                        config[key] = value.lower() == 'true'
                    elif value.isdigit():
                        config[key] = int(value)
                    else:
                        config[key] = value
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Using defaults.")
    return config


def main():
    """Main function to orchestrate the Spatial CIMIS processing workflow."""
    
    parser = argparse.ArgumentParser(description='Spatial CIMIS Data Processing with Grid Standardization')
    parser.add_argument('--config', type=str, default='scp_config.txt',
                       help='Configuration file path')
    parser.add_argument('--variable', type=str,
                       help='Variable to process (e.g., Tx, ETo, Tn, Rs)')
    parser.add_argument('--data-path', type=str,
                       help='Path to ASCII data files')
    parser.add_argument('--netcdf-path', type=str,
                       help='Path to NetCDF files')
    parser.add_argument('--output-path', type=str,
                       help='Path to output files')
    parser.add_argument('--start-year', type=int,
                       help='Start year for processing')
    parser.add_argument('--end-year', type=int,
                       help='End year for processing')
    parser.add_argument('--mask-years', type=str,
                       help='Years for mask creation (format: start-end)')
    parser.add_argument('--process-mask', action='store_true',
                       help='Process mask creation')
    parser.add_argument('--process-data', action='store_true',
                       help='Process variable data')
    parser.add_argument('--process-2d', action='store_true',
                       help='Process 2D files')
    parser.add_argument('--process-daily-clim', action='store_true',
                       help='Process daily climatology')
    parser.add_argument('--process-monthly-clim', action='store_true',
                       help='Process monthly climatology')
    parser.add_argument('--process-all', action='store_true',
                       help='Run all processing steps')
    parser.add_argument('--use-mask', action='store_true',
                       help='Apply mask during data processing (default: False)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set parameters with command line override
    variable = args.variable if args.variable is not None else config.get('variable', 'Tx')
    data_path = args.data_path if args.data_path is not None else config.get('data_path', '/group/moniergrp/SpatialCIMIS/ascii/')
    netcdf_path = args.netcdf_path if args.netcdf_path is not None else config.get('netcdf_path', '/group/moniergrp/SpatialCIMIS/netcdf/')
    output_path = args.output_path if args.output_path is not None else config.get('output_path', '/home/salba/SpatialCIMIS/data/')
    start_year = args.start_year if args.start_year is not None else config.get('start_year', 2004)
    end_year = args.end_year if args.end_year is not None else config.get('end_year', 2023)
    mask_years = args.mask_years if args.mask_years is not None else config.get('mask_years', '2004-2024')
    
    # Process flags
    process_mask = args.process_mask or config.get('process_mask', False)
    process_data = args.process_data or config.get('process_data', False)
    process_2d = args.process_2d or config.get('process_2d', False)
    process_daily_clim = args.process_daily_clim or config.get('process_daily_clim', False)
    process_monthly_clim = args.process_monthly_clim or config.get('process_monthly_clim', False)
    process_all = args.process_all or config.get('process_all', False)
    use_mask = args.use_mask or config.get('use_mask', False)
    
    print("="*60)
    print("Spatial CIMIS Processing with Grid Standardization")
    print("="*60)
    print(f"Configuration:")
    print(f"  Variable: {variable}")
    print(f"  Data path: {data_path}")
    print(f"  NetCDF path: {netcdf_path}")
    print(f"  Output path: {output_path}")
    print(f"  Start year: {start_year}")
    print(f"  End year: {end_year}")
    print(f"  Mask years: {mask_years}")
    print(f"  Use mask: {use_mask}")
    print(f"  Target grid: {TARGET_GRID['ncols']}x{TARGET_GRID['nrows']} at {TARGET_GRID['cellsize']}m")
    print()
    
    # Parse mask years
    mask_start, mask_end = map(int, mask_years.split('-'))
    
    try:
        if process_all or process_mask:
            create_variable_mask(data_path, netcdf_path, variable, mask_start, mask_end)
        
        if process_all or process_data:
            process_variable_data(data_path, netcdf_path, variable, start_year, end_year, use_mask=use_mask)
        
        if process_all or process_2d:
            create_2d_files(netcdf_path, output_path, variable, start_year, end_year)
        
        if process_all or process_daily_clim:
            create_daily_climatology(netcdf_path, output_path, variable)
        
        if process_all or process_monthly_clim:
            create_monthly_climatology(netcdf_path, output_path, variable)
        
        if not any([process_mask, process_data, process_2d, 
                   process_daily_clim, process_monthly_clim, process_all]):
            print("No processing steps specified. Use --help to see available options.")
            print("Use --process-all to run all processing steps.")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    print("="*60)
    print("All processing completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

