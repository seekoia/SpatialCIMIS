#!/usr/bin/env python3
"""
Extract nearest pixel time series from Spatial CIMIS and GridMET datasets
for each CIMIS station and save to CSV files.

Based on methodology in cimis_eto.ipynb
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray as rio
from datetime import datetime


def load_config(config_file):
    """Load configuration from text file."""
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config


def load_station_list(station_list_file):
    """Load and preprocess CIMIS station list."""
    print(f"  Loading station list from: {station_list_file}")
    
    station_list = pd.read_csv(station_list_file)
    
    # Parse lat/lon (handle "/ " separator if present)
    if station_list['HmsLatitude'].dtype == 'object':
        station_list['HmsLatitude'] = station_list['HmsLatitude'].str.split('/ ').str[-1]
        station_list['HmsLongitude'] = station_list['HmsLongitude'].str.split('/ ').str[-1]
    
    # Convert to numeric
    station_list['HmsLatitude'] = pd.to_numeric(station_list['HmsLatitude'], errors='coerce')
    station_list['HmsLongitude'] = pd.to_numeric(station_list['HmsLongitude'], errors='coerce')
    
    # Clean up station names
    station_list['Name'] = station_list['Name'].str.replace(" ", "")
    station_list['Name'] = station_list['Name'].str.replace("/", "")
    station_list['Name'] = station_list['Name'].str.replace(".", "")
    station_list['Name'] = station_list['Name'].str.replace("-", "")
    
    # Parse dates
    station_list['DisconnectDate'] = pd.to_datetime(station_list['DisconnectDate'], errors='coerce')
    station_list['ConnectDate'] = pd.to_datetime(station_list['ConnectDate'], errors='coerce')
    
    # Rename columns
    station_list = station_list.rename(columns={
        'HmsLatitude': 'Latitude',
        'HmsLongitude': 'Longitude'
    })
    
    print(f"  Loaded {len(station_list)} stations")
    return station_list


def filter_stations(station_list, start_date, end_date):
    """Filter stations that were active during the analysis period."""
    print(f"  Filtering stations active between {start_date} and {end_date}")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    filtered = station_list.loc[
        (station_list['ConnectDate'] <= start_dt) &
        (station_list['DisconnectDate'] > end_dt)
    ]
    
    print(f"  {len(filtered)} stations active during period")
    return filtered


def load_spatial_cimis(netcdf_path, variable, start_date, end_date):
    """Load Spatial CIMIS data."""
    print(f"\n2. Loading Spatial CIMIS data...")
    print(f"  Path: {netcdf_path}")
    
    # Build file pattern (matching spatial_cimis_analysis.py approach)
    pattern = netcdf_path + f'spatial_cimis_{variable.lower()}_20*.nc'
    print(f"  Pattern: {pattern}")
    
    # Open multiple files
    ds = xr.open_mfdataset(pattern, combine='nested', concat_dim="time")
    data = ds[variable]
    
    # Set CRS
    data.rio.write_crs(3310, inplace=True)
    data.rio.write_nodata(np.nan, inplace=True)
    
    # Subset by time
    data = data.sel(time=slice(start_date, end_date))
    
    print(f"  Loaded {len(data.time)} time steps")
    print(f"  Grid shape: {data.shape}")
    print(f"  Time range: {data.time.values[0]} to {data.time.values[-1]}")
    
    return data, ds


def load_gridmet(netcdf_path, variable, start_date, end_date):
    """Load GridMET data."""
    print(f"\n3. Loading GridMET data...")
    print(f"  Path: {netcdf_path}")
    
    # Map variable names
    variable_map = {
        'ETo': 'pet',
        'Rs': 'srad',
        'Tn': 'tmmn',
        'Tx': 'tmmx'
    }
    
    gridmet_var = variable_map.get(variable, variable.lower())
    print(f"  Variable: {gridmet_var}")
    
    # Determine years to load
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    
    print(f"  Loading years {start_year} to {end_year-1}")
    
    # Create list of file paths
    netcdf_files = [
        os.path.join(netcdf_path, f"{gridmet_var}_{year}.nc") 
        for year in range(start_year, end_year)
    ]
    
    # Check which files exist
    existing_files = [f for f in netcdf_files if os.path.exists(f)]
    
    if len(existing_files) == 0:
        raise FileNotFoundError(f"No GridMET files found for {gridmet_var}")
    
    print(f"  Found {len(existing_files)} files")
    
    # Load and concatenate files
    combined_data = None
    
    for i, file in enumerate(existing_files):
        try:
            # Open raster file
            data = rio.open_rasterio(file, masked=True)
            data.name = gridmet_var
            
            # Concatenate
            if combined_data is None:
                combined_data = data
            else:
                combined_data = xr.concat([combined_data, data], dim='band')
            
            if (i + 1) % 5 == 0:
                print(f"    Processed {i+1}/{len(existing_files)} files")
        
        except Exception as e:
            print(f"    Warning: Could not load {os.path.basename(file)}: {e}")
            continue
    
    if combined_data is None:
        raise ValueError(f"No data loaded for {gridmet_var}")
    
    # Convert units if needed (GridMET ETo is in 0.1mm, convert to mm)
    if variable == 'ETo':
        combined_data = combined_data * 0.1
    
    # Set CRS
    combined_data.rio.write_crs("EPSG:4326", inplace=True)
    combined_data.rio.write_nodata(np.nan, inplace=True)
    
    print(f"  Loaded: {combined_data.shape}")
    
    return combined_data


def extract_station_pixels_efficient(stations, spatial_data, gridmet_netcdf_path, variable, start_date, end_date):
    """Extract nearest pixel time series for each station (fast vectorized with dask)."""
    print(f"\n4. Extracting nearest pixels for {len(stations)} stations...")
    
    # Create GeoDataFrame and reproject to EPSG:3310 for Spatial CIMIS
    stations_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.Longitude, stations.Latitude),
        crs="EPSG:4326"
    )
    stations_3310 = stations_gdf.to_crs('epsg:3310')
    
    # Get time coordinates from Spatial CIMIS
    time_coord = spatial_data.time.values
    
    print("  Extracting from Spatial CIMIS (vectorized)...")
    
    # Vectorized extraction for Spatial CIMIS
    station_coords_3310 = np.array([[stations_3310.loc[idx, 'geometry'].x, 
                                      stations_3310.loc[idx, 'geometry'].y] 
                                     for idx in stations.index])
    
    spatial_station_data = {}
    for i, (idx, row) in enumerate(stations.iterrows()):
        station_num = str(row['StationNbr'])
        try:
            xx_3310 = station_coords_3310[i, 0]
            yy_3310 = station_coords_3310[i, 1]
            spatial_value = spatial_data.sel(x=xx_3310, y=yy_3310, method='nearest')
            spatial_station_data[station_num] = spatial_value.values
        except Exception as e:
            print(f"    Warning: Could not extract Spatial CIMIS for station {station_num}: {e}")
            continue
    
    spatial_df = pd.DataFrame(spatial_station_data)
    spatial_df['date'] = time_coord
    spatial_df.set_index('date', inplace=True)
    
    print(f"  ✓ Extracted Spatial CIMIS for {len(spatial_df.columns)} stations")
    
    # Fast GridMET extraction with dask
    print("  Extracting from GridMET (using dask for lazy loading)...")
    
    variable_map = {'ETo': 'pet', 'Rs': 'srad', 'Tn': 'tmmn', 'Tx': 'tmmx'}
    gridmet_var = variable_map.get(variable, variable.lower())
    
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    
    # Build file pattern
    file_pattern = os.path.join(gridmet_netcdf_path, f"{gridmet_var}_*.nc")
    
    try:
        # Open all GridMET files at once with dask (lazy loading, only loads chunks needed)
        print(f"    Opening GridMET files with dask chunking...")
        gridmet_ds = xr.open_mfdataset(
            file_pattern,
            combine='nested',
            concat_dim='day',
            parallel=True,
            chunks={'day': 365, 'lat': 50, 'lon': 50}  # Chunk size for lazy loading
        )
        
        # Get the data variable
        if gridmet_var in gridmet_ds.data_vars:
            gridmet_data = gridmet_ds[gridmet_var]
        else:
            data_vars = [v for v in gridmet_ds.data_vars if v not in ['lat', 'lon', 'crs', 'spatial_ref']]
            gridmet_data = gridmet_ds[data_vars[0]] if data_vars else None
        
        if gridmet_data is None:
            raise ValueError(f"Could not find {gridmet_var} in GridMET files")
        
        # No unit conversion needed - xarray auto-applies scale_factor from NetCDF
        # GridMET pet is already in mm when read with xarray
        
        print(f"    GridMET data shape: {gridmet_data.shape}")
        print(f"    Extracting {len(stations)} station pixels...")
        
        # Vectorized extraction using sel with method='nearest' for all stations at once
        # This is MUCH faster than looping
        station_lons = xr.DataArray(stations['Longitude'].values, dims='station')
        station_lats = xr.DataArray(stations['Latitude'].values, dims='station')
        
        # Extract all stations at once (dask will only load the needed pixels)
        gridmet_stations = gridmet_data.sel(lon=station_lons, lat=station_lats, method='nearest')
        
        print(f"    Computing extraction (this triggers dask loading)...")
        # Compute the result (this is where dask actually loads data)
        gridmet_stations_computed = gridmet_stations.compute()
        
        # Convert to DataFrame
        # gridmet_stations_computed has shape (station, day)
        # We need (day, station) for DataFrame with days as rows
        gridmet_df = pd.DataFrame(
            gridmet_stations_computed.values.T,  # (station, day) -> (day, station)
            index=gridmet_stations_computed['day'].values,
            columns=[str(row['StationNbr']) for _, row in stations.iterrows()]
        )
        gridmet_df.index.name = 'date'
        
        gridmet_ds.close()
        
        print(f"  ✓ Extracted GridMET for {len(gridmet_df.columns)} stations")
        
    except Exception as e:
        print(f"  ✗ Error extracting GridMET: {e}")
        print(f"    Falling back to year-by-year processing...")
        
        # Fallback to year-by-year if dask fails
        gridmet_df = extract_gridmet_yearly(stations, gridmet_netcdf_path, gridmet_var, 
                                            variable, start_year, end_year)
    
    return spatial_df, gridmet_df


def extract_gridmet_yearly(stations, gridmet_netcdf_path, gridmet_var, variable, start_year, end_year):
    """Fallback: Extract GridMET year by year."""
    all_gridmet_data = []
    all_times = []
    
    for year in range(start_year, end_year):
        file_path = os.path.join(gridmet_netcdf_path, f"{gridmet_var}_{year}.nc")
        
        if not os.path.exists(file_path):
            continue
        
        try:
            ds = xr.open_dataset(file_path)
            
            if gridmet_var in ds.data_vars:
                data_year = ds[gridmet_var]
            else:
                data_vars = [v for v in ds.data_vars if v not in ['lat', 'lon', 'crs', 'spatial_ref']]
                data_year = ds[data_vars[0]] if data_vars else None
            
            if data_year is None:
                continue
            
            # No unit conversion needed - xarray auto-applies scale_factor
            
            year_station_data = {}
            for idx, row in stations.iterrows():
                station_num = str(row['StationNbr'])
                try:
                    station_ts = data_year.sel(lon=row['Longitude'], lat=row['Latitude'], method='nearest')
                    year_station_data[station_num] = station_ts.values
                except:
                    continue
            
            all_gridmet_data.append(pd.DataFrame(year_station_data))
            
            if 'day' in data_year.dims:
                all_times.extend(data_year['day'].values)
            elif 'time' in data_year.dims:
                all_times.extend(data_year['time'].values)
            
            ds.close()
            
            if (year - start_year + 1) % 5 == 0:
                print(f"    Processed {year - start_year + 1}/{end_year - start_year} years")
        
        except Exception as e:
            continue
    
    if all_gridmet_data:
        gridmet_df = pd.concat(all_gridmet_data, ignore_index=True)
        gridmet_df['date'] = all_times
        gridmet_df.set_index('date', inplace=True)
        return gridmet_df
    else:
        return pd.DataFrame()


def main():
    """Main extraction function."""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'analysis_config.txt'
    
    print("="*70)
    print("Extract Nearest Pixels for CIMIS Stations")
    print("="*70)
    
    # Load configuration
    config = load_config(config_file)
    
    variable = config.get('variable', 'ETo')
    station_list_file = config.get('station_list_file', 'CIMIS_Stations.csv')
    spatial_netcdf_path = config.get('spatial_netcdf_path', '/group/moniergrp/SpatialCIMIS/netcdf/')
    gridmet_netcdf_path = config.get('gridmet_netcdf_path', '/group/moniergrp/gridMET/')
    output_path = config.get('output_path', '/home/salba/SpatialCIMIS/output/')
    start_date = config.get('start_date', '2004-01-01')
    end_date = config.get('end_date', '2024-01-01')
    
    print(f"\nConfiguration:")
    print(f"  Variable: {variable}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Output path: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load station list
    print(f"\n1. Loading station data...")
    station_list = load_station_list(station_list_file)
    
    # Filter stations by date range
    filtered_stations = filter_stations(station_list, start_date, end_date)
    
    # Load Spatial CIMIS data
    spatial_data, spatial_ds = load_spatial_cimis(spatial_netcdf_path, variable, start_date, end_date)
    
    # Extract nearest pixels (GridMET processed year-by-year for memory efficiency)
    spatial_station_data, gridmet_station_data = extract_station_pixels_efficient(
        filtered_stations, spatial_data, gridmet_netcdf_path, variable, start_date, end_date
    )
    
    # Save to CSV
    print(f"\n5. Saving results...")
    
    spatial_output = os.path.join(output_path, f'spatial_cimis_station_{variable.lower()}.csv')
    gridmet_output = os.path.join(output_path, f'gridmet_station_{variable.lower()}.csv')
    
    spatial_station_data.to_csv(spatial_output)
    print(f"  Saved Spatial CIMIS data: {spatial_output}")
    
    gridmet_station_data.to_csv(gridmet_output)
    print(f"  Saved GridMET data: {gridmet_output}")
    
    # Save station metadata
    station_metadata_output = os.path.join(output_path, f'station_metadata_{variable.lower()}.csv')
    filtered_stations[['StationNbr', 'Name', 'Latitude', 'Longitude', 'Elevation', 
                       'ConnectDate', 'DisconnectDate']].to_csv(station_metadata_output, index=False)
    print(f"  Saved station metadata: {station_metadata_output}")
    
    # Print summary statistics
    print(f"\n6. Summary:")
    print(f"  Number of stations: {len(spatial_station_data.columns)}")
    print(f"  Number of days: {len(spatial_station_data)}")
    print(f"  Spatial CIMIS shape: {spatial_station_data.shape}")
    print(f"  GridMET shape: {gridmet_station_data.shape}")
    
    print("\n" + "="*70)
    print("Extraction complete!")
    print("="*70)
    
    # Close dataset
    spatial_ds.close()


if __name__ == "__main__":
    main()

