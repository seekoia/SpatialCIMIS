#!/usr/bin/env python3
"""
Spatial CIMIS Data I/O and Analysis Script

This script handles:
1. Loading station data from CIMIS CSV files
2. Loading Spatial CIMIS NetCDF data
3. Loading GridMET data (optional)
4. Computing climatologies and statistics
5. Matching and comparing datasets
6. Saving processed results

Configuration is read from a text file (analysis_config.txt by default)
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
    """
    Load configuration from a text file.
    
    Parameters:
    -----------
    config_file : str
        Path to configuration file
        
    Returns:
    --------
    dict : Configuration parameters
    """
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
        'spatial_netcdf_path': '/group/moniergrp/SpatialCIMIS/netcdf/',
        'output_path': '/home/salba/SpatialCIMIS/output/',
        'start_date': '2004-01-01',
        'end_date': '2024-01-01',
        'min_station_coverage': 600,
        'compute_climatology': True
    }


def load_station_list(config):
    """
    Load and process CIMIS station list.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
        
    Returns:
    --------
    pandas.DataFrame : Processed station list
    geopandas.GeoDataFrame : Station locations with geometry
    """
    print("\n1. Loading station list...")
    
    station_file = config.get('station_list_file', 'CIMIS_Stations.csv')
    
    station_list = pd.read_csv(station_file)
    
    # Clean up latitude/longitude if needed
    if 'HmsLatitude' in station_list.columns:
        station_list['HmsLatitude'] = station_list['HmsLatitude'].str.split('/ ').str[-1]
        station_list['HmsLongitude'] = station_list['HmsLongitude'].str.split('/ ').str[-1]
        station_list.columns = ['Latitude' if x=='HmsLatitude' else x for x in station_list.columns]
        station_list.columns = ['Longitude' if x=='HmsLongitude' else x for x in station_list.columns]
    
    # Clean station names
    station_list['Name'] = station_list['Name'].str.replace(" ", "")
    station_list['Name'] = station_list['Name'].str.replace("/", "")
    station_list['Name'] = station_list['Name'].str.replace(".", "")
    station_list['Name'] = station_list['Name'].str.replace("-", "")
    
    # Convert dates
    station_list['DisconnectDate'] = station_list['DisconnectDate'].astype('datetime64[ns]')
    station_list['ConnectDate'] = station_list['ConnectDate'].astype('datetime64[ns]')
    
    # Create GeoDataFrame
    stations = gpd.GeoDataFrame(
        station_list, 
        geometry=gpd.points_from_xy(station_list.Longitude, station_list.Latitude),
        crs="EPSG:4326"
    )
    
    print(f"  Loaded {len(station_list)} stations")
    
    return station_list, stations


def filter_stations_by_date(station_list, start_date, end_date):
    """
    Filter stations that were active during the specified date range.
    
    Parameters:
    -----------
    station_list : pandas.DataFrame
        Station list
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
        
    Returns:
    --------
    pandas.DataFrame : Filtered station list
    """
    filtered = station_list.loc[
        (station_list['ConnectDate'] <= start_date) &
        (station_list['DisconnectDate'] > end_date)
    ]
    
    print(f"  Filtered to {len(filtered)} stations active during {start_date} to {end_date}")
    
    return filtered


def load_station_data(config, station_list, variable_col_map):
    """
    Load station data from CSV files.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    station_list : pandas.DataFrame
        List of stations to load
    variable_col_map : dict
        Mapping of variable names to CSV column names
        
    Returns:
    --------
    pandas.DataFrame : Station data
    """
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


def load_spatial_cimis_data(config):
    """
    Load Spatial CIMIS NetCDF data.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
        
    Returns:
    --------
    xarray.DataArray : Variable data
    xarray.DataArray : Latitude
    xarray.DataArray : Longitude
    """
    print(f"\n3. Loading Spatial CIMIS {config['variable']} data...")
    
    netcdf_path = config.get('spatial_netcdf_path', '/group/moniergrp/SpatialCIMIS/netcdf/')
    variable = config['variable']
    
    # Load all yearly files
    pattern = netcdf_path + f'spatial_cimis_{variable.lower()}_20*.nc'
    print(f"  Pattern: {pattern}")
    
    ds = xr.open_mfdataset(pattern, combine='nested', concat_dim="time")
    var_data = ds[variable]
    
    # Set CRS if rioxarray is available
    if HAS_RIOXARRAY:
        var_data.rio.write_crs(3310, inplace=True)
        var_data.rio.write_nodata(np.nan, inplace=True)
    
    # Get lat/lon
    spatial_lat = ds['lat'].isel(time=0)
    spatial_lon = ds['lon'].isel(time=0)
    
    print(f"  Loaded {len(ds.time)} time steps")
    print(f"  Grid shape: {var_data.shape}")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    ds.close()
    
    return var_data, spatial_lat, spatial_lon


def load_gridmet_data(config):
    """
    Load GridMET NetCDF data and clip to California.
    Uses xarray.open_mfdataset with dask for memory-efficient loading.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
        
    Returns:
    --------
    xarray.DataArray : GridMET variable data
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
        return None
    
    print(f"  Found {len(file_list)} files for years {start_year}-{end_year-1}")
    
    try:
        # Open all GridMET files at once with dask (lazy loading)
        print(f"  Opening GridMET files with dask chunking...")
        gridmet_ds = xr.open_mfdataset(
            file_list,
            combine='nested',
            concat_dim='day',
            parallel=True,
            chunks={'day': 365, 'lat': 100, 'lon': 100}  # Chunk for lazy loading
        )
        
        # Get the data variable (handle different possible names)
        if gridmet_var in gridmet_ds.data_vars:
            gridmet_data = gridmet_ds[gridmet_var]
        elif 'potential_evapotranspiration' in gridmet_ds.data_vars:
            gridmet_data = gridmet_ds['potential_evapotranspiration']
        else:
            data_vars = [v for v in gridmet_ds.data_vars 
                        if v not in ['lat', 'lon', 'crs', 'spatial_ref']]
            if data_vars:
                gridmet_data = gridmet_ds[data_vars[0]]
                print(f"  Using variable: {data_vars[0]}")
            else:
                print(f"  Error: Could not find {gridmet_var} in GridMET files")
                return None
        
        # No unit conversion needed - xarray auto-applies scale_factor from NetCDF metadata
        # GridMET data is already in correct units (mm for pet, K for temperature, etc.)
        
        # Clip to California if rioxarray is available
        if HAS_RIOXARRAY:
            print(f"  Clipping to California bounding box...")
            california_box = box(-124.41060660766607, 32.5342307609976,
                               -114.13445790587905, 42.00965914828148)
            
            # Set CRS for clipping
            gridmet_data.rio.write_crs("EPSG:4326", inplace=True)
            gridmet_data.rio.write_nodata(np.nan, inplace=True)
            
            # Clip to California (this will trigger dask computation)
            gridmet_data = gridmet_data.rio.clip([california_box], drop=True)
        else:
            # Manual clipping if rioxarray not available
            lat_slice = slice(49.5, 32.0)  # Approximate CA latitude range
            lon_slice = slice(-125.0, -114.0)  # Approximate CA longitude range
            gridmet_data = gridmet_data.sel(lat=lat_slice, lon=lon_slice)
        
        print(f"  GridMET data loaded: {gridmet_data.shape}")
        if HAS_RIOXARRAY:
            print(f"  CRS: {gridmet_data.rio.crs}")
        
        gridmet_ds.close()
        
        return gridmet_data
        
    except Exception as e:
        print(f"  Error loading GridMET data: {e}")
        print(f"  Falling back to year-by-year loading...")
        
        # Fallback: Load year by year
        return load_gridmet_yearly(gridmet_path, gridmet_var, start_year, end_year)


def load_gridmet_yearly(gridmet_path, gridmet_var, start_year, end_year):
    """
    Fallback method to load GridMET data year by year.
    
    Parameters:
    -----------
    gridmet_path : str
        Path to GridMET files
    gridmet_var : str
        Variable name (pet, tmmx, etc.)
    start_year : int
        Start year
    end_year : int
        End year
        
    Returns:
    --------
    xarray.DataArray : Concatenated GridMET data
    """
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
            else:
                data_vars = [v for v in ds.data_vars 
                            if v not in ['lat', 'lon', 'crs', 'spatial_ref']]
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


def compute_gridmet_climatology(gridmet_data, config):
    """
    Compute GridMET climatology.
    
    Parameters:
    -----------
    gridmet_data : xarray.DataArray
        GridMET data
    config : dict
        Configuration parameters
        
    Returns:
    --------
    xarray.DataArray : GridMET climatological mean
    """
    if gridmet_data is None:
        return None
    
    print(f"\n5. Computing GridMET climatology...")
    
    gridmet_clim = gridmet_data.mean(dim="day")
    gridmet_clim.rio.write_crs("EPSG:4326", inplace=True)
    
    # Save
    output_path = config.get('output_path', '/home/salba/SpatialCIMIS/output/')
    output_file = config.get('gridmet_mean_output', f'gridmet_mean_{config["variable"]}.nc')
    output_file = output_file.replace('{variable}', config['variable'].lower())
    output_filepath = output_path + output_file
    
    # Convert to dataset for saving with CRS
    if isinstance(gridmet_clim, xr.DataArray):
        gridmet_ds = gridmet_clim.to_dataset(name=gridmet_clim.name if gridmet_clim.name else 'gridmet_' + config['variable'].lower())
    else:
        gridmet_ds = gridmet_clim
    
    # Add CRS as attribute
    gridmet_ds.attrs['crs'] = 'EPSG:4326'
    gridmet_ds.attrs['grid_mapping'] = 'crs'
    
    gridmet_ds.to_netcdf(output_filepath)
    print(f"  GridMET climatology saved to: {output_filepath}")
    
    return gridmet_clim


def compute_climatology(var_data, config):
    """
    Compute temporal climatology (mean over time).
    
    Parameters:
    -----------
    var_data : xarray.DataArray
        Variable data
    config : dict
        Configuration parameters
        
    Returns:
    --------
    xarray.DataArray : Climatological mean
    """
    print(f"\n6. Computing climatology for {config['variable']}...")
    
    clim = var_data.mean(dim='time')
    
    if HAS_RIOXARRAY:
        # Set spatial dimensions explicitly before setting CRS
        try:
            clim.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
        except:
            pass
        clim.rio.write_crs(3310, inplace=True)
        clim.rio.write_nodata(np.nan, inplace=True)
    
    # Save if requested
    output_path = config.get('output_path', '/home/salba/SpatialCIMIS/output/')
    os.makedirs(output_path, exist_ok=True)
    
    output_file = config.get('spatial_mean_output', f'spatial_mean_{config["variable"]}.nc')
    output_file = output_file.replace('{variable}', config['variable'].lower())
    output_filepath = output_path + output_file
    
    # Convert to dataset for saving with CRS
    if isinstance(clim, xr.DataArray):
        clim_ds = clim.to_dataset(name=clim.name if clim.name else config['variable'])
    else:
        clim_ds = clim
    
    # Add CRS as attribute
    clim_ds.attrs['crs'] = 'EPSG:3310'
    clim_ds.attrs['grid_mapping'] = 'spatial_ref'
    
    clim_ds.to_netcdf(output_filepath)
    
    print(f"  Climatology saved to: {output_filepath}")
    
    return clim


def save_station_data(station_data, config):
    """
    Save station data to CSV.
    
    Parameters:
    -----------
    station_data : pandas.DataFrame
        Station data
    config : dict
        Configuration parameters
    """
    output_path = config.get('output_path', '/home/salba/SpatialCIMIS/output/')
    os.makedirs(output_path, exist_ok=True)
    
    output_file = config.get('station_data_output', f'station_{config["variable"]}_data.csv')
    output_file = output_file.replace('{variable}', config['variable'].lower())
    output_filepath = output_path + output_file
    
    station_data.to_csv(output_filepath)
    print(f"  Station data saved to: {output_filepath}")


def main():
    """Main analysis workflow."""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'analysis_config.txt'
    
    print("="*70)
    print("Spatial CIMIS Data I/O and Analysis")
    print("="*70)
    
    # Load configuration
    config = load_config(config_file)
    
    print(f"\nConfiguration:")
    print(f"  Variable: {config.get('variable', 'ETo')}")
    print(f"  Date range: {config.get('start_date', '2004-01-01')} to {config.get('end_date', '2024-01-01')}")
    print(f"  Output path: {config.get('output_path', 'output/')}")
    
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
        
        # Load Spatial CIMIS data
        spatial_data, spatial_lat, spatial_lon = load_spatial_cimis_data(config)
        
        # Load GridMET data if requested
        gridmet_data = None
        gridmet_clim = None
        if config.get('match_gridmet', False):
            gridmet_data = load_gridmet_data(config)
            if gridmet_data is not None:
                gridmet_clim = compute_gridmet_climatology(gridmet_data, config)
        
        # Compute climatology if requested
        if config.get('compute_climatology', True):
            spatial_clim = compute_climatology(spatial_data, config)
        
        print(f"\n{'='*70}")
        print("✓ Analysis complete!")
        print(f"{'='*70}")
        
        # Save summary
        summary_file = config.get('output_path', 'output/') + 'analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Spatial CIMIS Analysis Summary\n")
            f.write(f"{'='*70}\n")
            f.write(f"Variable: {config['variable']}\n")
            f.write(f"Date range: {config.get('start_date')} to {config.get('end_date')}\n")
            f.write(f"Stations loaded: {len(station_data.columns)}\n")
            f.write(f"Spatial grid shape: {spatial_data.shape}\n")
            if gridmet_data is not None:
                f.write(f"GridMET data loaded: Yes\n")
                f.write(f"GridMET grid shape: {gridmet_data.shape}\n")
            else:
                f.write(f"GridMET data loaded: No\n")
            f.write(f"Analysis completed: {datetime.now()}\n")
        
        print(f"\nSummary saved to: {summary_file}")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

