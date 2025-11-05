#!/usr/bin/env python3
"""
Unified Spatial CIMIS Data Analysis and Extraction Script (GridMET Grid Version)

This script handles:
1. Loading station data from CIMIS CSV files
2. Loading Spatial CIMIS NetCDF data (reprojected to GridMET grid)
3. Loading GridMET data (optional)
4. Computing climatologies and statistics (optional)
5. Extracting nearest pixels for stations (optional)
6. Matching and comparing datasets
7. Saving processed results

Configuration is read from a text file (analysis_config.txt by default)

Key differences from original:
- Uses GridMET-projected Spatial CIMIS data (EPSG:4326)
- No need to reproject stations to EPSG:3310
- Both datasets use same coordinate system for easier comparison
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import rioxarray as rio
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False
    print("Warning: rioxarray not available. Some reprojection features will be limited.")

from rasterio.enums import Resampling
from shapely.geometry import box


def load_config(config_file='analysis_config.txt'):
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
                    
                    # Convert boolean strings
                    if value.lower() in ['true', 'false']:
                        config[key] = value.lower() == 'true'
                    # Convert integers
                    elif value.isdigit():
                        config[key] = int(value)
                    # Keep strings as is
                    else:
                        config[key] = value
        print(f"✓ Configuration loaded from: {config_file}")
    except FileNotFoundError:
        print(f"Warning: Config file {config_file} not found. Using defaults.")
        config = get_default_config()
    
    return config


def get_default_config():
    """Return default configuration parameters."""
    return {
        'variable': 'ETo',
        'station_csv_path': '/group/moniergrp/SpatialCIMIS/CIMIS/',
        'station_list_file': 'CIMIS_Stations.csv',
        'spatial_netcdf_path': '/group/moniergrp/SpatialCIMIS/netcdf/test/',  # GridMET-projected data
        'gridmet_netcdf_path': '/group/moniergrp/gridMET/',
        'output_path': '/home/salba/SpatialCIMIS/output/',
        'start_date': '2004-01-01',
        'end_date': '2024-01-01',
        'min_station_coverage': 600,
        'compute_climatology': True,
        'match_gridmet': True,
        'extract_station_pixels': True
    }


def load_station_list(config):
    """Load and process CIMIS station list."""
    print("\n1. Loading station list...")
    
    station_file = config.get('station_list_file', 'CIMIS_Stations.csv')
    station_list = pd.read_csv(station_file)
    
    # Clean up latitude/longitude if needed
    if 'HmsLatitude' in station_list.columns:
        station_list['HmsLatitude'] = station_list['HmsLatitude'].str.split('/ ').str[-1]
        station_list['HmsLongitude'] = station_list['HmsLongitude'].str.split('/ ').str[-1]
        station_list.columns = ['Latitude' if x=='HmsLatitude' else x for x in station_list.columns]
        station_list.columns = ['Longitude' if x=='HmsLongitude' else x for x in station_list.columns]
    
    # Convert to numeric
    station_list['Latitude'] = pd.to_numeric(station_list['Latitude'], errors='coerce')
    station_list['Longitude'] = pd.to_numeric(station_list['Longitude'], errors='coerce')
    
    # Clean station names
    station_list['Name'] = station_list['Name'].str.replace(" ", "")
    station_list['Name'] = station_list['Name'].str.replace("/", "")
    station_list['Name'] = station_list['Name'].str.replace(".", "")
    station_list['Name'] = station_list['Name'].str.replace("-", "")
    
    # Convert dates
    station_list['DisconnectDate'] = station_list['DisconnectDate'].astype('datetime64[ns]')
    station_list['ConnectDate'] = station_list['ConnectDate'].astype('datetime64[ns]')
    
    # Create GeoDataFrame (stations are already in EPSG:4326)
    stations = gpd.GeoDataFrame(
        station_list, 
        geometry=gpd.points_from_xy(station_list.Longitude, station_list.Latitude),
        crs="EPSG:4326"
    )
    
    print(f"  Loaded {len(station_list)} stations")
    
    return station_list, stations


def filter_stations_by_date(station_list, start_date, end_date):
    """Filter stations that were active during the specified date range."""
    filtered = station_list.loc[
        (station_list['ConnectDate'] <= start_date) &
        (station_list['DisconnectDate'] > end_date)
    ]
    
    print(f"  Filtered to {len(filtered)} stations active during {start_date} to {end_date}")
    
    return filtered


def load_station_data(config, station_list, variable_col_map):
    """Load station data from CSV files."""
    print(f"\n2. Loading station {config['variable']} data...")
    
    # Create date array
    dates = np.arange(config['start_date'], config['end_date'], dtype='datetime64[D]')
    dates = dates.astype('datetime64[ns]')
    
    # Initialize dataframe
    station_data = pd.DataFrame(dates, columns=['Date'])
    
    path_cimis = config.get('station_csv_path', '/group/moniergrp/SpatialCIMIS/CIMIS/')
    
    # Get filenames
    filenames = station_list['Name'].astype(str) + '.csv'
    files = os.listdir(path_cimis) if os.path.exists(path_cimis) else []
    filenames = [str(i) for i in filenames if i in files]
    
    print(f"  Found {len(filenames)} station files")
    
    # Variable column name mapping
    var_col = variable_col_map.get(config['variable'], config['variable'])
    
    # Iterate over all files
    for f in filenames:
        filepath = path_cimis + f
        
        try:
            stationdat = pd.read_csv(filepath)
            
            # Normalize column names based on variable
            for old_name, new_name in variable_col_map.items():
                if old_name in stationdat.columns:
                    stationdat.columns = [new_name if x == old_name else x for x in stationdat.columns]
            
            # Normalize other column names
            stationdat.columns = ['Station' if x == 'Stn Id' else x for x in stationdat.columns]
            stationdat['Date'] = stationdat['Date'].astype('datetime64[ns]')
            
            # Filter by date range
            stationdat = stationdat.loc[
                (stationdat['Date'] < config['end_date']) &
                (stationdat['Date'] >= config['start_date'])
            ]
            stationdat.reset_index(drop=True, inplace=True)
            
            if len(stationdat) > 0:
                stationnum = str(stationdat['Station'].iloc[0])
                stat_data = stationdat[['Date', var_col]].rename(columns={var_col: stationnum})
                station_data = pd.merge(station_data, stat_data, how='outer', on='Date')
        
        except Exception as e:
            print(f"    Warning: Could not load {f}: {e}")
            continue
    
    # Process the dataframe
    station_data = station_data.rename(columns={'Date': 'date'})
    station_data.set_index('date', inplace=True, drop=True)
    
    # Drop stations with too many missing values
    min_coverage = config.get('min_station_coverage', 600)
    station_data = station_data.dropna(thresh=len(station_data) - min_coverage, axis=1)
    
    print(f"  Loaded data for {len(station_data.columns)} stations")
    print(f"  Date range: {station_data.index[0]} to {station_data.index[-1]}")
    print(f"  Total days: {len(station_data)}")
    
    return station_data


def load_spatial_cimis_data_gridmet(config):
    """Load Spatial CIMIS NetCDF data (GridMET-projected version)."""
    print(f"\n3. Loading Spatial CIMIS {config['variable']} data (GridMET grid)...")
    
    netcdf_path = config.get('spatial_netcdf_path', '/group/moniergrp/SpatialCIMIS/netcdf/test/')
    variable = config['variable']
    
    # Load all yearly files (GridMET-projected version)
    pattern = netcdf_path + f'spatial_cimis_{variable.lower()}_20*_gridmet.nc'
    print(f"  Pattern: {pattern}")
    
    ds = xr.open_mfdataset(pattern, combine='nested', concat_dim="time")
    var_data = ds[variable]
    
    # Set CRS to EPSG:4326 (GridMET projection)
    if HAS_RIOXARRAY:
        var_data.rio.write_crs("EPSG:4326", inplace=True)
        var_data.rio.write_nodata(np.nan, inplace=True)
    
    # Get lat/lon (already in EPSG:4326)
    spatial_lat = ds['lat']
    spatial_lon = ds['lon']
    
    print(f"  Loaded {len(ds.time)} time steps")
    print(f"  Grid shape: {var_data.shape}")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"  CRS: EPSG:4326 (GridMET grid)")
    
    return var_data, spatial_lat, spatial_lon, ds


def load_gridmet_data(config):
    """
    Load GridMET NetCDF data using xarray with dask for memory efficiency.
    No unit conversion - xarray auto-applies scale_factor from NetCDF metadata.
    """
    print(f"\n4. Loading GridMET {config['variable']} data...")
    
    gridmet_path = config.get('gridmet_netcdf_path', '/group/moniergrp/gridMET/')
    variable = config['variable']
    
    # Variable name mapping for GridMET files
    gridmet_var_map = {
        'ETo': 'pet',
        'Tx': 'tmmx',
        'Tn': 'tmmn',
        'Rs': 'srad'
    }
    
    gridmet_var = gridmet_var_map.get(variable, variable.lower())
    
    # Create list of GridMET file paths for years
    start_year = int(config.get('start_date', '2004-01-01').split('-')[0])
    end_year = int(config.get('end_date', '2024-01-01').split('-')[0])
    
    print(f"  Variable: {gridmet_var}")
    print(f"  Path: {gridmet_path}")
    print(f"  Years: {start_year} to {end_year-1}")
    
    # Build explicit list of files for the year range (not wildcards)
    file_list = [os.path.join(gridmet_path, f"{gridmet_var}_{year}.nc") 
                 for year in range(start_year, end_year)]
    
    # Filter to existing files only
    file_list = [f for f in file_list if os.path.exists(f)]
    
    if len(file_list) == 0:
        print(f"  Error: No GridMET files found for years {start_year}-{end_year-1}")
        return None, None
    
    print(f"  Found {len(file_list)} files for years {start_year}-{end_year-1}")
    
    try:
        # Open all GridMET files at once with dask (lazy loading)
        print(f"  Opening GridMET files with dask chunking...")
        gridmet_ds = xr.open_mfdataset(
            file_list,
            combine='nested',
            concat_dim='day',
            parallel=True,
            chunks={'day': 365, 'lat': 100, 'lon': 100}
        )
        
        # Get the data variable (handle different possible names)
        if gridmet_var in gridmet_ds.data_vars:
            gridmet_data = gridmet_ds[gridmet_var]
        elif 'potential_evapotranspiration' in gridmet_ds.data_vars:
            gridmet_data = gridmet_ds['potential_evapotranspiration']
        elif 'surface_downwelling_shortwave_flux_in_air' in gridmet_ds.data_vars:
            gridmet_data = gridmet_ds['surface_downwelling_shortwave_flux_in_air']
            print(f"  Using variable: surface_downwelling_shortwave_flux_in_air")
        elif 'air_temperature' in gridmet_ds.data_vars and variable in ['Tx', 'Tn']:
            gridmet_data = gridmet_ds['air_temperature']
            print(f"  Using variable: air_temperature")
        elif 'temperature' in gridmet_ds.data_vars and variable in ['Tx', 'Tn']:
            gridmet_data = gridmet_ds['temperature']
            print(f"  Using variable: temperature")
        else:
            data_vars = [v for v in gridmet_ds.data_vars 
                        if v not in ['lat', 'lon', 'crs', 'spatial_ref', 'day']]
            if data_vars:
                gridmet_data = gridmet_ds[data_vars[0]]
                print(f"  Using variable: {data_vars[0]}")
            else:
                print(f"  Error: Could not find {gridmet_var} in GridMET files")
                print(f"  Available variables: {list(gridmet_ds.data_vars)}")
                gridmet_ds.close()
                return None, None
        
        # No unit conversion needed - xarray auto-applies scale_factor from NetCDF metadata
        
        # Clip to California if rioxarray is available
        if HAS_RIOXARRAY:
            print(f"  Clipping to California bounding box...")
            california_box = box(-124.41060660766607, 32.5342307609976,
                               -114.13445790587905, 42.00965914828148)
            
            gridmet_data.rio.write_crs("EPSG:4326", inplace=True)
            gridmet_data.rio.write_nodata(np.nan, inplace=True)
            gridmet_data = gridmet_data.rio.clip([california_box], drop=True)
        else:
            # Manual clipping if rioxarray not available
            lat_slice = slice(49.5, 32.0)
            lon_slice = slice(-125.0, -114.0)
            gridmet_data = gridmet_data.sel(lat=lat_slice, lon=lon_slice)
        
        print(f"  GridMET data loaded: {gridmet_data.shape}")
        if HAS_RIOXARRAY:
            print(f"  CRS: {gridmet_data.rio.crs}")
        
        return gridmet_data, gridmet_ds
        
    except Exception as e:
        print(f"  Error loading GridMET data: {e}")
        print(f"  Falling back to year-by-year loading...")
        
        gridmet_data = load_gridmet_yearly(gridmet_path, gridmet_var, start_year, end_year)
        return gridmet_data, None


def load_gridmet_yearly(gridmet_path, gridmet_var, start_year, end_year):
    """Fallback method to load GridMET data year by year."""
    all_data = []
    
    for year in range(start_year, end_year):
        file_path = os.path.join(gridmet_path, f"{gridmet_var}_{year}.nc")
        
        if not os.path.exists(file_path):
            continue
        
        try:
            ds = xr.open_dataset(file_path)
            
            # Get the data variable
            if gridmet_var in ds.data_vars:
                data_year = ds[gridmet_var]
            elif 'potential_evapotranspiration' in ds.data_vars:
                data_year = ds['potential_evapotranspiration']
            elif 'surface_downwelling_shortwave_flux_in_air' in ds.data_vars:
                data_year = ds['surface_downwelling_shortwave_flux_in_air']
            elif 'air_temperature' in ds.data_vars and variable in ['Tx', 'Tn']:
                data_year = ds['air_temperature']
            elif 'temperature' in ds.data_vars and variable in ['Tx', 'Tn']:
                data_year = ds['temperature']
            else:
                data_vars = [v for v in ds.data_vars 
                            if v not in ['lat', 'lon', 'crs', 'spatial_ref', 'day']]
                data_year = ds[data_vars[0]] if data_vars else None
            
            if data_year is None:
                continue
            
            # No unit conversion - xarray handles it automatically
            all_data.append(data_year)
            
            ds.close()
            
            if (year - start_year + 1) % 5 == 0:
                print(f"    Loaded {year - start_year + 1}/{end_year - start_year} years")
        
        except Exception as e:
            print(f"    Warning: Could not load {year}: {e}")
            continue
    
    if not all_data:
        print("  Error: No data loaded")
        return None
    
    # Concatenate all years
    combined_data = xr.concat(all_data, dim='day')
    
    if HAS_RIOXARRAY:
        combined_data.rio.write_crs("EPSG:4326", inplace=True)
        combined_data.rio.write_nodata(np.nan, inplace=True)
    
    print(f"  GridMET data loaded: {combined_data.shape}")
    
    return combined_data


def extract_station_pixels_gridmet(filtered_stations, spatial_data, gridmet_data, config):
    """Extract nearest pixel time series for each station (GridMET grid version)."""
    print(f"\n5. Extracting nearest pixels for {len(filtered_stations)} stations...")
    
    # Both datasets are now in EPSG:4326, so no reprojection needed
    stations_gdf = gpd.GeoDataFrame(
        filtered_stations,
        geometry=gpd.points_from_xy(filtered_stations.Longitude, filtered_stations.Latitude),
        crs="EPSG:4326"
    )
    
    # Get time coordinates from Spatial CIMIS
    time_coord = spatial_data.time.values
    
    print("  Extracting from Spatial CIMIS (GridMET grid)...")
    
    # Extract Spatial CIMIS pixels for each station (using lat/lon coordinates)
    spatial_station_data = {}
    for idx, row in filtered_stations.iterrows():
        station_num = str(row['StationNbr'])
        try:
            # Use lat/lon coordinates directly (no reprojection needed)
            lon_coord = row['Longitude']
            lat_coord = row['Latitude']
            spatial_value = spatial_data.sel(lon=lon_coord, lat=lat_coord, method='nearest')
            spatial_station_data[station_num] = spatial_value.values
        except Exception as e:
            print(f"    Warning: Could not extract Spatial CIMIS for station {station_num}: {e}")
            continue
    
    spatial_df = pd.DataFrame(spatial_station_data)
    spatial_df['date'] = time_coord
    spatial_df.set_index('date', inplace=True)
    
    print(f"  ✓ Extracted Spatial CIMIS for {len(spatial_df.columns)} stations")
    
    # Extract GridMET if available
    gridmet_df = None
    if gridmet_data is not None:
        print("  Extracting from GridMET...")
        
        # Try vectorized extraction with dask
        try:
            station_lons = xr.DataArray(filtered_stations['Longitude'].values, dims='station')
            station_lats = xr.DataArray(filtered_stations['Latitude'].values, dims='station')
            
            gridmet_stations = gridmet_data.sel(lon=station_lons, lat=station_lats, method='nearest')
            gridmet_stations_computed = gridmet_stations.compute()
            
            gridmet_df = pd.DataFrame(
                gridmet_stations_computed.values.T,
                index=gridmet_stations_computed['day'].values,
                columns=[str(row['StationNbr']) for _, row in filtered_stations.iterrows()]
            )
            gridmet_df.index.name = 'date'
            
            print(f"  ✓ Extracted GridMET for {len(gridmet_df.columns)} stations")
            
        except Exception as e:
            print(f"    Warning: Vectorized extraction failed: {e}")
            print(f"    Extracting station by station...")
            
            # Fallback: extract station by station
            gridmet_station_data = {}
            for idx, row in filtered_stations.iterrows():
                station_num = str(row['StationNbr'])
                try:
                    gridmet_value = gridmet_data.sel(
                        lon=row['Longitude'], 
                        lat=row['Latitude'], 
                        method='nearest'
                    )
                    gridmet_station_data[station_num] = gridmet_value.values
                except Exception as e:
                    print(f"      Warning: Could not extract GridMET for station {station_num}: {e}")
                    continue
            
            if gridmet_station_data:
                # Get time dimension name
                time_dim = 'day' if 'day' in gridmet_data.dims else 'time'
                gridmet_df = pd.DataFrame(gridmet_station_data)
                gridmet_df['date'] = gridmet_data[time_dim].values
                gridmet_df.set_index('date', inplace=True)
                print(f"  ✓ Extracted GridMET for {len(gridmet_df.columns)} stations")
    
    return spatial_df, gridmet_df


def compute_climatology_gridmet(var_data, config):
    """Compute temporal climatology (mean over time) for GridMET-projected data."""
    print(f"\n6. Computing Spatial CIMIS climatology for {config['variable']} (GridMET grid)...")
    
    clim = var_data.mean(dim='time')
    
    if HAS_RIOXARRAY:
        try:
            clim.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
        except:
            pass
        clim.rio.write_crs("EPSG:4326", inplace=True)
        clim.rio.write_nodata(np.nan, inplace=True)
    
    # Save
    output_path = config.get('output_path', '/home/salba/SpatialCIMIS/output/')
    os.makedirs(output_path, exist_ok=True)
    
    output_file = config.get('spatial_mean_output_gridmet', f'spatial_mean_{config["variable"]}_gridmet.nc')
    output_file = output_file.replace('{variable}', config['variable'].lower())
    output_filepath = output_path + output_file
    
    # Convert to dataset for saving with CRS
    if isinstance(clim, xr.DataArray):
        clim_ds = clim.to_dataset(name=clim.name if clim.name else config['variable'])
    else:
        clim_ds = clim
    
    # Add CRS as attribute
    clim_ds.attrs['crs'] = 'EPSG:4326'
    clim_ds.attrs['grid_mapping'] = 'crs'
    
    clim_ds.to_netcdf(output_filepath)
    
    print(f"  Climatology saved to: {output_filepath}")
    
    return clim


def compute_gridmet_climatology(gridmet_data, config):
    """Compute GridMET climatology."""
    if gridmet_data is None:
        return None
    
    print(f"\n7. Computing GridMET climatology...")
    
    gridmet_clim = gridmet_data.mean(dim="day")
    
    if HAS_RIOXARRAY:
        gridmet_clim.rio.write_crs("EPSG:4326", inplace=True)
    
    # Save
    output_path = config.get('output_path', '/home/salba/SpatialCIMIS/output/')
    output_file = config.get('gridmet_mean_output', f'gridmet_mean_{config["variable"]}.nc')
    output_file = output_file.replace('{variable}', config['variable'].lower())
    output_filepath = output_path + output_file
    
    # Convert to dataset for saving with CRS
    if isinstance(gridmet_clim, xr.DataArray):
        gridmet_ds = gridmet_clim.to_dataset(
            name=gridmet_clim.name if gridmet_clim.name else 'gridmet_' + config['variable'].lower()
        )
    else:
        gridmet_ds = gridmet_clim
    
    # Add CRS as attribute
    gridmet_ds.attrs['crs'] = 'EPSG:4326'
    gridmet_ds.attrs['grid_mapping'] = 'crs'
    
    gridmet_ds.to_netcdf(output_filepath)
    print(f"  GridMET climatology saved to: {output_filepath}")
    
    return gridmet_clim


def save_station_data(station_data, config):
    """Save station data to CSV."""
    output_path = config.get('output_path', '/home/salba/SpatialCIMIS/output/')
    os.makedirs(output_path, exist_ok=True)
    
    output_file = config.get('station_data_output', f'station_{config["variable"]}_data.csv')
    output_file = output_file.replace('{variable}', config['variable'].lower())
    output_filepath = output_path + output_file
    
    station_data.to_csv(output_filepath)
    print(f"  Station data saved to: {output_filepath}")


def save_extracted_pixels(spatial_df, gridmet_df, filtered_stations, config):
    """Save extracted pixel time series to CSV."""
    print(f"\n8. Saving extracted pixel time series...")
    
    output_path = config.get('output_path', '/home/salba/SpatialCIMIS/output/')
    variable = config['variable']
    
    # Save Spatial CIMIS station pixels (GridMET grid version)
    spatial_output = os.path.join(output_path, f'spatial_cimis_station_{variable.lower()}_gridmet.csv')
    spatial_df.to_csv(spatial_output)
    print(f"  Saved Spatial CIMIS data (GridMET grid): {spatial_output}")
    
    # Save GridMET station pixels if available
    if gridmet_df is not None:
        gridmet_output = os.path.join(output_path, f'gridmet_station_{variable.lower()}.csv')
        gridmet_df.to_csv(gridmet_output)
        print(f"  Saved GridMET data: {gridmet_output}")
    
    # Save station metadata
    station_metadata_output = os.path.join(output_path, f'station_metadata_{variable.lower()}_gridmet.csv')
    filtered_stations[['StationNbr', 'Name', 'Latitude', 'Longitude', 'Elevation', 
                       'ConnectDate', 'DisconnectDate']].to_csv(station_metadata_output, index=False)
    print(f"  Saved station metadata: {station_metadata_output}")


def main():
    """Main analysis workflow."""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'analysis_config.txt'
    
    print("="*70)
    print("Unified Spatial CIMIS Data Analysis (GridMET Grid Version)")
    print("="*70)
    
    # Load configuration
    config = load_config(config_file)
    
    print(f"\nConfiguration:")
    print(f"  Variable: {config.get('variable', 'ETo')}")
    print(f"  Date range: {config.get('start_date', '2004-01-01')} to {config.get('end_date', '2024-01-01')}")
    print(f"  Output path: {config.get('output_path', 'output/')}")
    print(f"  Compute climatology: {config.get('compute_climatology', True)}")
    print(f"  Match GridMET: {config.get('match_gridmet', False)}")
    print(f"  Extract station pixels: {config.get('extract_station_pixels', True)}")
    print(f"  Using GridMET-projected Spatial CIMIS data")
    
    # Variable-specific column name mappings for station data
    variable_col_map = {
        'ETo (mm)': 'DayAsceEto',
        'DayAsceEto.Value': 'DayAsceEto',
        'Sol Rad (W/sq.m)': 'DaySolRadAvg',
        'DaySolRadAvg.Value': 'DaySolRadAvg',
        'ETo': 'DayAsceEto',
        'Rs': 'DaySolRadAvg',
        'Tx': 'DayAirTmpMax',
        'Tn': 'DayAirTmpMin'
    }
    
    try:
        # Load station list
        station_list, stations_gdf = load_station_list(config)
        
        # Filter stations by date range
        filtered_stations = filter_stations_by_date(
            station_list,
            config.get('start_date', '2004-01-01'),
            config.get('end_date', '2024-01-01')
        )
        
        # Load station data
        station_data = load_station_data(config, filtered_stations, variable_col_map)
        
        # Save station data
        save_station_data(station_data, config)
        
        # Load Spatial CIMIS data (GridMET-projected version)
        spatial_data, spatial_lat, spatial_lon, spatial_ds = load_spatial_cimis_data_gridmet(config)
        
        # Load GridMET data if requested
        gridmet_data = None
        gridmet_ds = None
        gridmet_clim = None
        if config.get('match_gridmet', False):
            gridmet_data, gridmet_ds = load_gridmet_data(config)
            
            # Compute GridMET climatology if requested
            if gridmet_data is not None and config.get('compute_climatology', True):
                gridmet_clim = compute_gridmet_climatology(gridmet_data, config)
        
        # Compute Spatial CIMIS climatology if requested
        if config.get('compute_climatology', True):
            spatial_clim = compute_climatology_gridmet(spatial_data, config)
        
        # Extract station pixels if requested
        if config.get('extract_station_pixels', True):
            spatial_pixels, gridmet_pixels = extract_station_pixels_gridmet(
                filtered_stations, spatial_data, gridmet_data, config
            )
            save_extracted_pixels(spatial_pixels, gridmet_pixels, filtered_stations, config)
        
        print(f"\n{'='*70}")
        print("✓ Analysis complete!")
        print(f"{'='*70}")
        
        # Save summary
        summary_file = config.get('output_path', 'output/') + 'analysis_summary_gridmet.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Spatial CIMIS Unified Analysis Summary (GridMET Grid Version)\n")
            f.write(f"{'='*70}\n")
            f.write(f"Variable: {config['variable']}\n")
            f.write(f"Date range: {config.get('start_date')} to {config.get('end_date')}\n")
            f.write(f"Stations loaded: {len(station_data.columns)}\n")
            f.write(f"Spatial grid shape: {spatial_data.shape}\n")
            f.write(f"Spatial CRS: EPSG:4326 (GridMET grid)\n")
            if gridmet_data is not None:
                f.write(f"GridMET data loaded: Yes\n")
                f.write(f"GridMET grid shape: {gridmet_data.shape}\n")
            else:
                f.write(f"GridMET data loaded: No\n")
            f.write(f"Climatology computed: {config.get('compute_climatology', True)}\n")
            f.write(f"Station pixels extracted: {config.get('extract_station_pixels', True)}\n")
            f.write(f"Analysis completed: {datetime.now()}\n")
        
        print(f"\nSummary saved to: {summary_file}")
        
        # Close datasets
        spatial_ds.close()
        if gridmet_ds is not None:
            gridmet_ds.close()
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
