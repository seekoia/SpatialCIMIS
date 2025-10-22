#!/usr/bin/env python3
"""
Create Tx mask - based directly on SpatialCIMIS-data-processing-Copy1.ipynb
"""

import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from netCDF4 import date2num, num2date
import datetime as dt
from datetime import datetime, timedelta, date
import rasterio
import pyproj
from pyproj import Transformer
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import calendar

def dates_daily(y0, m0, d0, mtot, noleap):
    """
    Function that calculates arrays of dates (hour, day,month, year)
    Imput:
       y0:     initial year [integer]
       m0:     initial month [integer]
       d0:     initial day [integer]
       mtot:   total number of elements in the time series [integer]
       noleap: flag to indicate whether or not leap years should be considered.
               noleap = 0 --> time searies contain Feb 29
               noleap = 1 --> time series does not contain Feb 29
    Output:
       day:    array with values of days
       month:  array with values of months
       year:   array with values of years
    Example:
    --------
      >>> day,month,year=dates_daily(1980,1,1,365,0)
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


# EXACT code from notebook Cell 2
print("Creating Tx mask...")

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
    
    print(f"Processing year {yr} ({i+1}/{nyr})...")
    
    # Calculate the number of days in the dataset
    ntime = 365 + calendar.isleap(yr)
    
    # Generate daily dates for the year
    day, month, year = dates_daily(yr, 1, 1, ntime, 0)
    
    # Create an empty array to store daily data
    data_tmp = np.zeros((ntime, ny, nx))
    
    for t in range(ntime):
        # Construct the filename for the ASCII data file
        filename = data_path + 'Tx.' + str(int(year[t])).zfill(4) + '-' + str(int(month[t])).zfill(2) + '-' + str(int(day[t])).zfill(2) + '.asc'
        
        # Check if the file exists
        if os.path.isfile(filename):
            src = rasterio.open(filename)
            tmp = src.read(1)
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
    
    print(f"  Year {yr}: max daily count = {data[i, :, :].max():.0f}, sum = {data[i, :, :].sum():.0f}")

# Calculate a mask by summing the yearly data and applying a threshold
mask = np.sum(data, axis=0)
print(f"\nBefore division: mask max = {mask.max():.0f}, sum = {mask.sum():.0f}")

mask = mask / nyr
print(f"After division by {nyr}: mask max = {mask.max():.0f}")

mask[mask < 350] = 0
mask[mask >= 350] = 1

print(f"After threshold (350): valid cells = {np.sum(mask == 1):,}")

# Extract geospatial information from the source raster
b = src.transform
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
output_file = '/home/salba/SpatialCIMIS/test_tx_mask.nc'
xrds.to_netcdf(output_file)

# Print a confirmation message
print(f'\n{"="*60}')
print(f'Mask saved to: {output_file}')
print(f'Valid cells in mask: {np.sum(mask == 1):,}')
print(f'{"="*60}')
print('Done!')







