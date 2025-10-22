#!/usr/bin/env python3
"""
Spatial CIMIS Data Processing Script

This script processes Spatial CIMIS data from ASCII files and converts them to NetCDF format.
It includes functions for date calculations, mask creation, data processing, and output generation.

Original notebook: SpatialCIMIS-data-processing-Copy1.ipynb
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
import rasterio
import pyproj
from pyproj import Transformer
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import pandas as pd
import dask


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
        noleap = 0 --> time series contain Feb 29
        noleap = 1 --> time series does not contain Feb 29
    
    Returns:
    --------
    day : array
        Array with values of days
    month : array
        Array with values of months
    year : array
        Array with values of years
    
    Example:
    --------
    >>> day, month, year = dates_daily(1980, 1, 1, 365, 0)
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
    Function that calculates arrays of dates (hour, day, month, year)
    
    Parameters:
    -----------
    y0 : int
        Initial year
    noleap : int
        Flag to indicate whether or not leap years should be considered.
        noleap = 0 --> time series contain Feb 29
        noleap = 1 --> time series does not contain Feb 29
    
    Returns:
    --------
    day : array
        Array with values of days
    month : array
        Array with values of months
    year : array
        Array with values of years
    
    Example:
    --------
    >>> day, month, year = dates_daily(1980, 1, 1, 365, 0)
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


def create_mask(variable='Tx'):
    """
    Create a mask to ensure that only grid cells with 90% of non-NaN values are kept.
    This deals with the issue of change in grid coverage change on Oct 23 2018.
    
    Parameters:
    -----------
    variable : str
        Variable name (Tx, ETo, Tn, Tdew, U2, etc.)
    """
    # Define the start and end years for processing
    yr_start = 2004
    yr_end = 2024
    years = range(yr_start, yr_end + 1)
    nyr = len(years)

    # Define file paths and grid dimensions
    data_path = "/group/moniergrp/SpatialCIMIS/ascii/"
    netcdf_path = '/group/moniergrp/SpatialCIMIS/netcdf/'
    ny = 552
    nx = 500

    # Create an empty 3D array to store the data
    data = np.zeros((nyr, ny, nx))

    # Loop over years to create one netcdf file per year
    for i in range(nyr):
        yr = years[i]
        
        print(f"  Processing year {yr} ({i+1}/{nyr})...")
        
        # Calculate the number of days in the dataset
        ntime = 365 + calendar.isleap(yr)
        
        # Generate daily dates for the year
        day, month, year = dates_daily(yr, 1, 1, ntime, 0)
        
        # Create an empty array to store daily data
        data_tmp = np.zeros((ntime, ny, nx))
        
        for t in range(ntime):
            # Construct the filename for the ASCII data file
            filename = data_path + variable + '.' + str(int(year[t])).zfill(4) + '-' + str(int(month[t])).zfill(2) + '-' + str(int(day[t])).zfill(2) + '.asc'
            
            # Check if the file exists
            if os.path.isfile(filename):
                src = rasterio.open(filename)
                tmp = src.read(1)
                # Use the ORIGINAL notebook logic that works:
                # This converts: data→1, nodata→0 for mask creation
                tmp[tmp == 0] = -9999
                tmp[tmp != -9999] = 1
                tmp[tmp == -9999] = 0
                if tmp.shape[0] == ny:
                    data_tmp[t, :, :] = tmp
                else:
                    data_tmp[t, :, :] = tmp[3:555, 5:505]  # Trimming the data for days when the grid is 560 x 510
            else:
                data_tmp[t, :, :] = np.zeros((ny, nx))
        
        # Sum the daily data to obtain yearly data
        data[i, :, :] = np.sum(data_tmp, axis=0)
        print(f"    Year {yr}: max daily count = {data[i, :, :].max():.0f}, sum = {data[i, :, :].sum():.0f}")

    # Calculate a mask by summing the yearly data and applying a threshold
    mask = np.sum(data, axis=0)
    mask = mask / nyr
    
    # Use 350 days threshold (original notebook value)
    # For variables where 0 is valid, adjust threshold dynamically
    threshold = 350
    print(f"  Before threshold: mask max = {mask.max():.0f}")
    print(f"  Using mask threshold: {threshold:.0f} days")
    
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 1
    
    print(f"  After threshold: valid cells = {np.sum(mask == 1):,}")

    # Extract geospatial information from the source raster
    # Use the last successfully opened raster for transform information
    if 'src' in locals():
        b = src.transform
    else:
        # Fallback transform if no files were found
        b = (2000.0, 0.0, -2400000.0, 0.0, -2000.0, 2200000.0)
    x = np.arange(nx) * b[0] + b[2]
    y = np.arange(ny) * b[4] + b[5]

    # Create a mesh grid for coordinates
    xl, yl = np.meshgrid(x + 1000, y + 1000)

    # Use PyProj to transform coordinates to latitude and longitude
    transformer = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xl, yl)

    # Get the current UTC time as a string
    dtnow = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Create an xarray dataset to store the data and metadata
    xrds = xr.Dataset(
        # Assign coordinates
        coords=dict(
            y=(['y'], y),
            x=(['x'], x)
        ),
        # Assign variables
        data_vars=dict(
            lat=(['y', 'x'], lat),
            lon=(['y', 'x'], lon),
            mask=(['y', 'x'], mask)
        ),
        # Assign global attributes
        attrs=dict(
            title='Spatial CIMIS Maximum temperature (Tx) Mask',
            summary='Mask for the daily Maximum temperature (Tx) at a 2 km spatial resolution over California obtained from the California Irrigation Management Information System (CIMIS) Spatial Model developed at the University of California, Davis. The data is converted from daily ASCII files into yearly netcdf files starting with 2004.',
            source='Spatial CIMIS data',
            processing_level='Raw data',
            creator_type='group',
            creator_name='UC Davis Global Environmental Change Lab',
            date_created=dtnow,
            history=f'File created at {dtnow} using xarray in Python',
            geospatial_bounds_crs='EPSG:3310'
        )
    )

    # Assign attributes to variables in the xarray dataset
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
        'long_name': 'Mask of grid cells with Spatial CIMIS ETo data; 1 = data, 0 = no data.',
        'units': '1',
        'coverage_content_type': 'thematicClassification'
    }

    # Save the xarray dataset to a NetCDF file
    output_file = netcdf_path + f'spatial_cimis_{variable.lower()}_mask.nc'
    xrds.to_netcdf(output_file)

    # Print a confirmation message
    print(f'  Mask saved to: {output_file}')
    print(f'{variable} mask creation completed!')


def process_spatial_cimis_data(variable='Tx'):
    """
    Read the Spatial CIMIS ASCII files, extract for all grid cells falling within 
    the Spatial CIMIS mask, interpolate all missing and zero values using temporal 
    spline interpolation, and output as netcdf files.
    
    Parameters:
    -----------
    variable : str
        Variable name (Tx, ETo, Tn, Tdew, U2, etc.)
    """
    # Define the start and end years for processing
    yr_start = 2004
    yr_end = 2023
    years = range(yr_start, yr_end + 1)
    nyr = len(years)

    # Define file paths and grid dimensions
    data_path = "/group/moniergrp/SpatialCIMIS/ascii/"
    netcdf_path = '/group/moniergrp/SpatialCIMIS/netcdf/'
    ny = 552
    nx = 500

    # Open the existing xarray dataset to retrieve the mask
    ds = xr.open_dataset(netcdf_path + f'spatial_cimis_{variable.lower()}_mask.nc')
    mask = ds['mask']

    # Loop over years to create one NetCDF file per year
    for i in range(nyr):
        yr = years[i]
        
        # Calculate the number of days in the dataset for the current year
        ntime = 3 * 365 + calendar.isleap(yr - 1) + calendar.isleap(yr) + calendar.isleap(yr + 1)
        day, month, year = dates_daily(yr - 1, 1, 1, ntime, 0)
        
        # Create an empty array to store daily data
        data = np.zeros((ntime, ny, nx))
        
        for t in range(ntime):
            # Construct the filename for the ASCII data file
            filename = data_path + variable + '.' + str(int(year[t])).zfill(4) + '-' + str(int(month[t])).zfill(2) + '-' + str(int(day[t])).zfill(2) + '.asc'
            
            # Check if the file exists
            if os.path.isfile(filename): 
                src = rasterio.open(filename)
                tmp = src.read(1)
                tmp[tmp == 0] = -9999
                
                # Check the shape and apply the mask
                if tmp.shape[0] == ny:
                    tmp[mask == 0] = -9999
                    data[t, :, :] = tmp
                else:
                    data_tmp = tmp[3:555, 5:505]  # Trimming the data for days when the grid is 560 x 510
                    data_tmp[mask == 0] = -9999
                    data[t, :, :] = data_tmp
            else:
                data[t, :, :] = np.full((ny, nx), -9999)

        # Get the grid dimensions and transform coordinates
        ny = 552
        nx = 500
        # Use the last successfully opened raster for transform information
        if 'src' in locals():
            b = src.transform
        else:
            # Fallback transform if no files were found
            b = (2000.0, 0.0, -2400000.0, 0.0, -2000.0, 2200000.0)
        x = np.arange(nx) * b[0] + b[2]
        y = np.arange(ny) * b[4] + b[5]
        
        xl, yl = np.meshgrid(x + 1000, y + 1000)
        transformer = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xl, yl)

        # Get the current UTC time as a string
        dtnow = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

        # Define time units and create time values
        time_units = "days since 2004-01-01"
        dates = [dt.datetime(int(i), int(j), int(k)) for i, j, k in zip(year, month, day)]
        times = date2num(dates, time_units)
        
        # Create a new xarray dataset to store the data and metadata
        xrds = xr.Dataset(
        # Assign coordinates
            coords=dict(
                time=(['time'], times),
                y=(['y'], y),
                x=(['x'], x)
            ),
        # Assign variables
        data_vars=dict(
            lat=(['y', 'x'], lat),
            lon=(['y', 'x'], lon),
            **{variable: (['time', 'y', 'x'], data)}
        ),
        # Assign global attributes
            attrs=dict(
                title=f'Spatial CIMIS {variable}',
                summary=f'Daily {variable} at a 2 km spatial resolution over California obtained from the California Irrigation Management Information System (CIMIS) Spatial Model developed at the University of California, Davis. The data is converted from daily ASCII files into yearly NetCDF files starting with 2004.',
                source='Spatial CIMIS data',
                processing_level='Raw data',
                creator_type='group',
                creator_name='UC Davis Global Environmental Change Lab',
                date_created=dtnow,
                history=f'File created at {dtnow} using xarray in Python',
                geospatial_bounds_crs='EPSG:3310'
            )
        )
        
        # Apply a mask to exclude no data values and interpolate missing values
        xrds[variable] = xrds[variable].where(xrds[variable] != -9999.)
        xrds[variable] = xrds[variable].interpolate_na(dim="time", method="linear")

        # Assign attributes to variables in the xarray dataset
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
        # Define variable-specific attributes
        var_attrs = {
            'Tx': {
                'standard_name': 'air_temperature',
                'long_name': 'Daily Maximum Temperature',
                'units': 'C',
                'coverage_content_type': 'physicalMeasurement'
            },
            'ETo': {
                'standard_name': 'reference_evapotranspiration',
                'long_name': 'Daily Reference Evapotranspiration',
                'units': 'mm/day',
                'coverage_content_type': 'physicalMeasurement'
            },
            'Tn': {
                'standard_name': 'air_temperature',
                'long_name': 'Daily Minimum Temperature',
                'units': 'C',
                'coverage_content_type': 'physicalMeasurement'
            },
            'Tdew': {
                'standard_name': 'dew_point_temperature',
                'long_name': 'Daily Dew Point Temperature',
                'units': 'C',
                'coverage_content_type': 'physicalMeasurement'
            },
            'U2': {
                'standard_name': 'wind_speed',
                'long_name': 'Daily Wind Speed at 2m',
                'units': 'm/s',
                'coverage_content_type': 'physicalMeasurement'
            }
        }
        
        xrds[variable].attrs = var_attrs.get(variable, {
            'standard_name': variable.lower(),
            'long_name': f'Daily {variable}',
            'units': 'unknown',
            'coverage_content_type': 'physicalMeasurement'
        })

        # Select a subset of data for the current year
        ds = xrds.isel(time=range(365 + calendar.isleap(yr - 1), 365 + calendar.isleap(yr - 1) + 365 + calendar.isleap(yr)))

        # Save the dataset to a NetCDF file for the current year
        ds.to_netcdf(netcdf_path + f'spatial_cimis_{variable.lower()}_{yr}.nc')
        print(f'Done with year {yr} for variable {variable}')
        
    print('Done!')


def create_2d_daily_netcdf(variable='Tx'):
    """
    Create 2D (grid, time) daily netcdf files.
    
    Parameters:
    -----------
    variable : str
        Variable name (Tx, ETo, Tn, Tdew, U2, etc.)
    """
    netcdf_path = '/group/moniergrp/SpatialCIMIS/netcdf/'
    output_path = '/home/salba/SpatialCIMIS/data/'

    # Loop over years to create one netcdf file per year
    for yr in range(2021, 2023):
        ds = xr.open_dataset(netcdf_path + f'spatial_cimis_{variable.lower()}_{yr}.nc')
        var_data = ds[variable]
        var_2d = var_data.stack(grid=("y", "x"))
        data = var_2d.transpose('grid', 'time')

        ncfile = Dataset(output_path + f'spatial_cimis_{variable.lower()}_2d_{yr}.nc', mode='w', format='NETCDF4_CLASSIC') 

        grid_dim = ncfile.createDimension('grid', data.shape[0])     # grid axis
        time_dim = ncfile.createDimension('time', data.shape[1]) # unlimited axis (can be appended to).
            
        ncfile.title = f'Spatial CIMIS {variable}'
        ncfile.description = f'Daily {variable} at a 2 km spatial resolution over California obtained from the California Irrigation Management Information System (CIMIS) Spatial Model developed at the University of California, Davis. The data is converted from daily ASCII files into yearly netcdf files starting with 2004.'
        ncfile.author = 'UC Davis Global Environmental Change Lab'
        ncfile.history = 'Created ' + datetime.today().strftime('%Y-%m-%d')
        
        grid = ncfile.createVariable('grid', np.float32, ('grid',))
        time = ncfile.createVariable('time', np.float64, ('time',))
        time.units = 'days since 2004-01-01'
        time.long_name = 'time'
        
        # Define a 3D variable to hold the data
        var_nc = ncfile.createVariable(variable, np.float64, ('grid', 'time'), fill_value=-9999) # note: unlimited dimension is leftmost
        
        # Set variable-specific attributes
        var_attrs = {
            'Tx': {'units': 'C', 'standard_name': 'air_temperature', 'description': 'Daily Maximum Temperature'},
            'ETo': {'units': 'mm/day', 'standard_name': 'reference_evapotranspiration', 'description': 'Daily Reference Evapotranspiration'},
            'Tn': {'units': 'C', 'standard_name': 'air_temperature', 'description': 'Daily Minimum Temperature'},
            'Tdew': {'units': 'C', 'standard_name': 'dew_point_temperature', 'description': 'Daily Dew Point Temperature'},
            'U2': {'units': 'm/s', 'standard_name': 'wind_speed', 'description': 'Daily Wind Speed at 2m'}
        }
        
        attrs = var_attrs.get(variable, {'units': 'unknown', 'standard_name': variable.lower(), 'description': f'Daily {variable}'})
        var_nc.units = attrs['units']
        var_nc.standard_name = attrs['standard_name']
        var_nc.description = attrs['description']
        
        # Write grid number.
        # Note: the ":" is necessary in these "write" statements
        grid[:] = range(data.shape[0])

        time[:] = range(data.shape[1])
        # Write the data.  This writes the whole 3D netCDF variable all at once.
        var_nc[:, :] = data.to_numpy()  # Appends data along unlimited dimension
        
        # close the Dataset.
        ncfile.close()

    print('Done!')


def create_daily_climatology(variable='Tx'):
    """
    Create daily climatology from multi-year dataset.
    
    Parameters:
    -----------
    variable : str
        Variable name (Tx, ETo, Tn, Tdew, U2, etc.)
    """
    netcdf_path = '/group/moniergrp/SpatialCIMIS/netcdf/'
    output_path = '/home/salba/SpatialCIMIS/data/'

    ds = xr.open_mfdataset(netcdf_path + f'spatial_cimis_{variable.lower()}_20*.nc', combine='nested', concat_dim="time")
    var_data = ds[variable]
    var_daily_clim = var_data.groupby('time.dayofyear').mean(dim='time')
    var_daily_clim_1d = var_daily_clim.stack(grid=("y", "x"))
    data = var_daily_clim_1d.transpose('grid', 'dayofyear')

    ncfile = Dataset(output_path + f'spatial_cimis_{variable.lower()}_daily_clim_2d_2004-2022.nc', mode='w', format='NETCDF4_CLASSIC') 

    grid_dim = ncfile.createDimension('grid', data.shape[0])     # grid axis
    time_dim = ncfile.createDimension('time', data.shape[1]) # unlimited axis (can be appended to).
        
    ncfile.title = f'Daily Annual Cycle of Spatial CIMIS {variable}'
    ncfile.description = f'Daily {variable} annual cycle at a 2 km spatial resolution over California obtained from the California Irrigation Management Information System (CIMIS) Spatial Model developed at the University of California, Davis. The data is converted from daily ASCII files into yearly netcdf files starting with 2004.'
    ncfile.author = 'UC Davis Global Environmental Change Lab'
    ncfile.history = 'Created ' + datetime.today().strftime('%Y-%m-%d')
        
    grid = ncfile.createVariable('grid', np.float32, ('grid',))
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'day of year'
    time.long_name = 'time'
    
    # Define a 3D variable to hold the data
    var_nc = ncfile.createVariable(variable, np.float64, ('grid', 'time'), fill_value=-9999) # note: unlimited dimension is leftmost
    
    # Set variable-specific attributes
    var_attrs = {
        'Tx': {'units': 'C', 'standard_name': 'air_temperature', 'description': 'Daily Maximum Temperature'},
        'ETo': {'units': 'mm/day', 'standard_name': 'reference_evapotranspiration', 'description': 'Daily Reference Evapotranspiration'},
        'Tn': {'units': 'C', 'standard_name': 'air_temperature', 'description': 'Daily Minimum Temperature'},
        'Tdew': {'units': 'C', 'standard_name': 'dew_point_temperature', 'description': 'Daily Dew Point Temperature'},
        'U2': {'units': 'm/s', 'standard_name': 'wind_speed', 'description': 'Daily Wind Speed at 2m'}
    }
    
    attrs = var_attrs.get(variable, {'units': 'unknown', 'standard_name': variable.lower(), 'description': f'Daily {variable}'})
    var_nc.units = attrs['units']
    var_nc.standard_name = attrs['standard_name']
    var_nc.description = attrs['description']
        
    # Write grid number.
    # Note: the ":" is necessary in these "write" statements
    grid[:] = range(data.shape[0])

    time[:] = range(366)
    # Write the data.  This writes the whole 3D netCDF variable all at once.
    var_nc[:, :] = data.to_numpy()  # Appends data along unlimited dimension
        
    # close the Dataset.
    ncfile.close()

    print('Done!')


def create_monthly_climatology(variable='Tx'):
    """
    Create monthly climatology from multi-year dataset.
    
    Parameters:
    -----------
    variable : str
        Variable name (Tx, ETo, Tn, Tdew, U2, etc.)
    """
    netcdf_path = '/group/moniergrp/SpatialCIMIS/netcdf/'
    output_path = '/home/salba/SpatialCIMIS/data/'

    ds = xr.open_mfdataset(netcdf_path + f'spatial_cimis_{variable.lower()}_20*.nc', combine='nested', concat_dim="time")
    var_data = ds[variable]
    var_monthly_clim = var_data.groupby('time.month').mean(dim='time')
    var_monthly_clim_1d = var_monthly_clim.stack(grid=("y", "x"))
    data = var_monthly_clim_1d.transpose('grid', 'month')

    ncfile = Dataset(output_path + f'spatial_cimis_{variable.lower()}_monthly_clim_2d_2004-2022.nc', mode='w', format='NETCDF4_CLASSIC') 

    grid_dim = ncfile.createDimension('grid', data.shape[0])     # grid axis
    time_dim = ncfile.createDimension('time', data.shape[1]) # unlimited axis (can be appended to).
        
    ncfile.title = f'Monthly Annual Cycle of Spatial CIMIS {variable}'
    ncfile.description = f'Monthly {variable} annual cycle at a 2 km spatial resolution over California obtained from the California Irrigation Management Information System (CIMIS) Spatial Model developed at the University of California, Davis. The data is converted from daily ASCII files into yearly netcdf files starting with 2004.'
    ncfile.author = 'UC Davis Global Environmental Change Lab'
    ncfile.history = 'Created ' + datetime.today().strftime('%Y-%m-%d')
        
    grid = ncfile.createVariable('grid', np.float32, ('grid',))
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'month of year'
    time.long_name = 'time'
    
    # Define a 3D variable to hold the data
    var_nc = ncfile.createVariable(variable, np.float64, ('grid', 'time'), fill_value=-9999) # note: unlimited dimension is leftmost
    
    # Set variable-specific attributes
    var_attrs = {
        'Tx': {'units': 'C', 'standard_name': 'air_temperature', 'description': 'Daily Maximum Temperature'},
        'ETo': {'units': 'mm/day', 'standard_name': 'reference_evapotranspiration', 'description': 'Daily Reference Evapotranspiration'},
        'Tn': {'units': 'C', 'standard_name': 'air_temperature', 'description': 'Daily Minimum Temperature'},
        'Tdew': {'units': 'C', 'standard_name': 'dew_point_temperature', 'description': 'Daily Dew Point Temperature'},
        'U2': {'units': 'm/s', 'standard_name': 'wind_speed', 'description': 'Daily Wind Speed at 2m'}
    }
    
    attrs = var_attrs.get(variable, {'units': 'unknown', 'standard_name': variable.lower(), 'description': f'Daily {variable}'})
    var_nc.units = attrs['units']
    var_nc.standard_name = attrs['standard_name']
    var_nc.description = attrs['description']
        
    # Write grid number.
    # Note: the ":" is necessary in these "write" statements
    grid[:] = range(data.shape[0])

    time[:] = range(12)
    # Write the data.  This writes the whole 3D netCDF variable all at once.
    var_nc[:, :] = data.to_numpy()  # Appends data along unlimited dimension
        
    # close the Dataset.
    ncfile.close()

    print('Done!')


def main():
    """
    Main function to run the Spatial CIMIS data processing pipeline.
    """
    import sys
    
    # Get variable from command line argument (required)
    if len(sys.argv) < 2:
        print("Error: Variable argument is required.")
        print("Usage: python spatial_cimis_data_processing.py <variable>")
        print(f"Supported variables: {', '.join(['Tx', 'ETo', 'Tn', 'Tdew', 'U2'])}")
        sys.exit(1)
    
    variable_input = sys.argv[1]
    
    # Normalize variable name (keep proper case)
    variable_map = {
        'TX': 'Tx', 'tx': 'Tx', 'Tx': 'Tx',
        'ETO': 'ETo', 'eto': 'ETo', 'ETo': 'ETo',
        'TN': 'Tn', 'tn': 'Tn', 'Tn': 'Tn',
        'TDEW': 'Tdew', 'tdew': 'Tdew', 'Tdew': 'Tdew',
        'U2': 'U2', 'u2': 'U2'
    }
    
    variable = variable_map.get(variable_input, variable_input)
    valid_variables = ['Tx', 'ETo', 'Tn', 'Tdew', 'U2']
    if variable not in valid_variables:
        print(f"Error: Variable '{variable}' not supported.")
        print(f"Supported variables: {', '.join(valid_variables)}")
        sys.exit(1)
    
    print(f"Starting Spatial CIMIS data processing for variable: {variable}")
    
    # Step 1: Create mask (skip if already exists)
    mask_file = '/group/moniergrp/SpatialCIMIS/netcdf/' + f'spatial_cimis_{variable.lower()}_mask.nc'
    if os.path.exists(mask_file):
        print(f"Mask file already exists: {mask_file}")
        print(f"Skipping mask creation...")
    else:
        print(f"Creating mask for {variable}...")
        create_mask(variable)
    
    # Step 2: Process spatial CIMIS data
    print(f"Processing spatial CIMIS data for {variable}...")
    process_spatial_cimis_data(variable)
    
    # Step 3: Create 2D daily netcdf files
    print("Creating 2D daily netcdf files...")
    create_2d_daily_netcdf(variable)
    
    # Step 4: Create daily climatology
    print("Creating daily climatology...")
    create_daily_climatology(variable)
    
    # Step 5: Create monthly climatology
    print("Creating monthly climatology...")
    create_monthly_climatology(variable)
    
    print("All processing complete!")


if __name__ == "__main__":
    main()
