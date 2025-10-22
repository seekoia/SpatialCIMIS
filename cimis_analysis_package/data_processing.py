# cimis_analysis_package/data_processing.py

import pandas as pd
import numpy as np
import os
import xarray as xr
import rioxarray as rio
from rasterio.enums import Resampling

# Import configuration constants
from . import config

def filter_stations_by_date(stations_gdf, 
                            connect_date_filter_str=config.CONNECT_DATE_FILTER, 
                            disconnect_date_filter_str=config.DISCONNECT_DATE_FILTER):
    """
    Filters stations GeoDataFrame based on connection and disconnection dates.
    Args:
        stations_gdf (gpd.GeoDataFrame): Input GeoDataFrame of stations.
        connect_date_filter_str (str): Earliest connection date string.
        disconnect_date_filter_str (str): Latest disconnection date string.
    Returns:
        gpd.GeoDataFrame or None: Filtered GeoDataFrame, or original if error.
    """
    if stations_gdf is None:
        print("stations_gdf is None in filter_stations_by_date.")
        return None
    try:
        connect_date_filter = pd.to_datetime(connect_date_filter_str)
        disconnect_date_filter = pd.to_datetime(disconnect_date_filter_str)
        
        # Ensure date columns are datetime
        stations_gdf['ConnectDate'] = pd.to_datetime(stations_gdf['ConnectDate'], errors='coerce')
        stations_gdf['DisconnectDate'] = pd.to_datetime(stations_gdf['DisconnectDate'], errors='coerce')

        filtered_stations = stations_gdf[
            (stations_gdf['ConnectDate'].notna()) & 
            (stations_gdf['ConnectDate'] <= connect_date_filter) &
            (
                (stations_gdf['DisconnectDate'].isna()) | 
                (stations_gdf['DisconnectDate'] > disconnect_date_filter)
            )
        ].copy() # Use .copy() to avoid SettingWithCopyWarning on the slice
        print(f"Filtered stations by date: {len(filtered_stations)} remaining.")
        return filtered_stations
    except Exception as e:
        print(f"Error filtering stations by date: {e}")
        return stations_gdf # Return original if filtering fails, or handle more gracefully

def load_station_data_from_csvs(filtered_stations_df, target_var_config, 
                                cimis_data_path=config.CIMIS_DATA_PATH,
                                start_date_str=config.START_DATE_FILTER, 
                                end_date_str=config.END_DATE_FILTER):
    """
    Loads data for the target variable for filtered stations from individual station CSVs.
    Args:
        filtered_stations_df (pd.DataFrame): DataFrame of filtered stations.
        target_var_config (dict): Configuration for the target variable.
        cimis_data_path (str): Path to the directory containing CIMIS station CSVs.
        start_date_str (str): Start date for data filtering.
        end_date_str (str): End date for data filtering.
    Returns:
        pd.DataFrame: DataFrame with the target variable data for stations.
    """
    if filtered_stations_df is None or filtered_stations_df.empty:
        print("No filtered stations to process for data loading.")
        return pd.DataFrame()

    cimis_col_name = target_var_config["cimis_csv_column_name"]
    # The output column name in the final DataFrame will be the station ID (station_col_id)
    # The 'cimis_csv_output_name' is more for identifying the variable being processed.

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str) # Exclusive end date for np.arange
    # Create a full date range index for consistent merging
    dates_index = pd.date_range(start=start_date, end=pd.to_datetime(end_date_str) - pd.Timedelta(days=1), freq='D')

    station_var_df = pd.DataFrame(index=dates_index)
    processed_files_count = 0
    
    # Prepare a list of available CSV files in uppercase for case-insensitive matching
    try:
        available_files_upper = {f.upper() for f in os.listdir(cimis_data_path) if f.endswith('.csv')}
    except FileNotFoundError:
        print(f"Error: CIMIS data path not found: {cimis_data_path}")
        return pd.DataFrame()

    for _, row in filtered_stations_df.iterrows():
        station_name = str(row.get('Name', 'UnknownStation')) # Get station name, default if not found
        station_nbr = row.get('StationNbr', None)
        
        try:
            # Prioritize StationNbr for column naming if it's a valid integer-like string
            station_col_id = str(int(float(station_nbr))) if pd.notna(station_nbr) else station_name
        except (ValueError, TypeError):
            station_col_id = station_name # Fallback to station name if StationNbr is not convertible

        # Construct filename for matching (case-insensitive) and actual opening
        filename_to_match = station_name.upper() + '.CSV'
        actual_filename_to_open = station_name + '.csv' # Use original case for os.path.join

        if filename_to_match in available_files_upper:
            file_path = os.path.join(cimis_data_path, actual_filename_to_open)
            try:
                # Specify common NA values
                station_data_raw = pd.read_csv(file_path, parse_dates=['Date'], 
                                            na_values=['None', '--', '', ' ', '  ', 'NA', 'N/A', 'NaN'])
                
                # Standardize column names (case-insensitive check)
                # Create a mapping from lowercased column names to original names
                col_map_lower = {c.lower().strip(): c for c in station_data_raw.columns}
                
                # Determine which column to use for the target variable
                current_col_to_use = None
                if cimis_col_name.lower().strip() in col_map_lower:
                    current_col_to_use = col_map_lower[cimis_col_name.lower().strip()]
                else: # Try alternative names if primary is not found
                    renamed_col_options = {
                        'eto (mm)': 'dayasceeto', 'sol rad (w/sq.m)': 'daysolradavg',
                        'dayasceeto.value': 'dayasceeto', 'daysolradavg.value': 'daysolradavg'
                    } # lowercased potential alternative names
                    
                    # Check if cimis_col_name (e.g. "ETo (mm)") maps to an alternative (e.g. "dayasceeto")
                    alt_key_lower = cimis_col_name.lower().strip()
                    if alt_key_lower in renamed_col_options:
                        potential_alt_name_lower = renamed_col_options[alt_key_lower]
                        if potential_alt_name_lower in col_map_lower:
                            current_col_to_use = col_map_lower[potential_alt_name_lower]
                
                if not current_col_to_use:
                    # print(f"Warning: Target column '{cimis_col_name}' or alternatives not found in {actual_filename_to_open}.")
                    station_data_raw[station_col_id] = np.nan # Ensure column exists for this station
                else:
                     station_data_raw[station_col_id] = pd.to_numeric(station_data_raw[current_col_to_use], errors='coerce')

                # Filter by date, remove duplicates, and set index
                station_data_filtered = station_data_raw[['Date', station_col_id]].copy()
                station_data_filtered = station_data_filtered[(station_data_filtered['Date'] >= start_date) & 
                                                              (station_data_filtered['Date'] <= dates_index.max())] # Inclusive of start, up to last day of range
                station_data_filtered = station_data_filtered.drop_duplicates(subset=['Date'])
                station_data_filtered = station_data_filtered.set_index('Date')
                
                # Merge with the main DataFrame, ensuring alignment on the full date index
                if not station_data_filtered.empty:
                    # station_var_df = station_var_df.join(station_data_filtered[[station_col_id]], how='left')
                    # A direct assignment is often cleaner if station_col_id is unique and df is pre-indexed
                    station_var_df[station_col_id] = station_data_filtered[station_col_id].reindex(dates_index)

                    processed_files_count += 1
            except FileNotFoundError: # Should not happen if filename_match is in available_files_upper
                 print(f"File not found error for {file_path} (should have been caught).")
            except Exception as e:
                print(f"Error processing file {actual_filename_to_open} for station {station_name} (ID: {station_col_id}), var {target_var_config['cimis_csv_output_name']}: {e}")
        else:
            # This message can be very verbose if many stations in filtered_stations_df don't have corresponding CSVs
            # print(f"File {filename_to_match} for station {station_name} (ID: {station_col_id}) not found in {cimis_data_path}. Skipping.")
            pass
            
    print(f"Processed {processed_files_count} station data files for variable '{target_var_config['cimis_csv_output_name']}'.")
    
    # Drop columns (stations) that are mostly NaN
    # Original threshold was len(df) - 600. A percentage might be more robust.
    # e.g., drop if more than 90% NaN, so thresh is 10% of length.
    min_valid_count = len(station_var_df) * 0.1 
    station_var_df.dropna(axis=1, thresh=min_valid_count, inplace=True)
    
    station_var_df.index.name = 'date'
    print(f"Station '{target_var_config['cimis_csv_output_name']}' data shape: {station_var_df.shape} after loading and cleaning.")
    return station_var_df

def get_taylor_list_stations(full_station_gdf, station_var_df):
    """
    Creates a GeoDataFrame of stations that are present as columns in station_var_df.
    Args:
        full_station_gdf (gpd.GeoDataFrame): The complete list of stations.
        station_var_df (pd.DataFrame): DataFrame containing the variable data,
                                       with station identifiers as columns.
    Returns:
        gpd.GeoDataFrame or None: Filtered GeoDataFrame for Taylor analysis.
    """
    if full_station_gdf is None or station_var_df is None or station_var_df.columns.empty:
        print("Cannot create Taylor list: Missing GDF, variable data, or variable data has no station columns.")
        return None
    
    # Station identifiers from the columns of station_var_df (should be strings)
    var_station_cols_as_str = [str(col) for col in station_var_df.columns]
    
    try:
        # Ensure 'StationNbr' in full_station_gdf is also treated as string for comparison
        taylor_list_gdf = full_station_gdf[
            full_station_gdf['StationNbr'].astype(str).isin(var_station_cols_as_str)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning
    except KeyError:
        print("Error: 'StationNbr' column not found in the provided station GeoDataFrame for Taylor list.")
        return None
    except Exception as e:
        print(f"Error filtering for Taylor list: {e}")
        return None

    if 'StationNbr' in taylor_list_gdf.columns:
        if not taylor_list_gdf['StationNbr'].astype(str).duplicated().any():
            # Set index to StationNbr if it exists and is unique (after casting to str for safety)
            # Keep StationNbr as a column as well, as it's useful.
            taylor_list_gdf.set_index(taylor_list_gdf['StationNbr'].astype(str), inplace=True, drop=False)
        else:
            # If there are duplicates, don't set index or handle duplicates appropriately
            print("Warning: Duplicate StationNbr values found in Taylor list; not setting as index.")
    
    print(f"Taylor list created with {len(taylor_list_gdf)} stations for current variable.")
    return taylor_list_gdf


def load_spatial_gridded_data(netcdf_path_pattern, netcdf_var_name, target_crs_epsg=3310):
    """
    Loads and preprocesses specified variable from Spatial CIMIS type NetCDF files.
    Args:
        netcdf_path_pattern (str): Glob pattern for NetCDF files.
        netcdf_var_name (str): Name of the variable to extract from NetCDF.
        target_crs_epsg (int): Target EPSG code for CRS.
    Returns:
        tuple (xr.DataArray, xr.DataArray) or (None, None): Full timeseries DataArray and
                                                            seasonal mean DataArray, or Nones if error.
    """
    print(f"Loading Spatial gridded data for variable '{netcdf_var_name}' from: {netcdf_path_pattern}")
    try:
        # Using combine='by_coords' is generally safer for time series if coordinates are well-defined
        ds = xr.open_mfdataset(netcdf_path_pattern, combine='by_coords', concat_dim="time", engine="netcdf4")
        
        if netcdf_var_name not in ds.variables:
            print(f"Error: Variable '{netcdf_var_name}' not found in dataset from {netcdf_path_pattern}.")
            print(f"Available variables: {list(ds.variables)}")
            ds.close()
            return None, None
            
        data_arr = ds[netcdf_var_name]
        
        # Ensure CRS is set or written
        if data_arr.rio.crs is None:
            data_arr = data_arr.rio.write_crs(f"EPSG:{target_crs_epsg}", inplace=False)
        
        # Calculate seasonal mean
        data_arr_daily_seas = data_arr.groupby("time.season").mean('time', skipna=True)
        if data_arr_daily_seas.rio.crs is None and data_arr.rio.crs is not None: # Inherit CRS if possible
             data_arr_daily_seas = data_arr_daily_seas.rio.write_crs(data_arr.rio.crs, inplace=False)
        elif data_arr_daily_seas.rio.crs is None: # Fallback if main array also had no CRS
             data_arr_daily_seas = data_arr_daily_seas.rio.write_crs(f"EPSG:{target_crs_epsg}", inplace=False)


        # Set nodata value
        data_arr = data_arr.rio.write_nodata(np.nan, inplace=False)
        data_arr_daily_seas = data_arr_daily_seas.rio.write_nodata(np.nan, inplace=False)
        
        ds.close() # Close the multifile dataset
        print(f"Spatial gridded data for '{netcdf_var_name}' loaded. Shape: {data_arr.shape}")
        return data_arr, data_arr_daily_seas
    except FileNotFoundError:
        print(f"Error: No files found matching pattern '{netcdf_path_pattern}' for variable '{netcdf_var_name}'.")
        return None, None
    except Exception as e:
        print(f"Error loading Spatial gridded data for '{netcdf_var_name}': {e}")
        return None, None


def load_gridmet_gridded_data(netcdf_dir, gridmet_file_prefix, gridmet_internal_var_heuristic, 
                              shapefile_path, start_year=2004, end_year=2023, clip_to_shapefile=True):
    """
    Loads, optionally clips, and combines specified variable from GridMET NetCDF data.
    Args:
        netcdf_dir (str): Directory containing GridMET NetCDF files.
        gridmet_file_prefix (str): Prefix for GridMET filenames (e.g., "srad", "tmmn").
        gridmet_internal_var_heuristic (str): A keyword to help find the actual data variable
                                             name within the NetCDF file (e.g., "radiation", "temperature").
        shapefile_path (str): Path to the shapefile for clipping.
        start_year (int): Start year for data loading.
        end_year (int): End year for data loading.
        clip_to_shapefile (bool): Whether to clip data to the shapefile.
    Returns:
        xr.DataArray or None: Combined and optionally clipped DataArray, or None if error.
    """
    print(f"Loading GridMET gridded data for prefix '{gridmet_file_prefix}' (heuristic: '{gridmet_internal_var_heuristic}') from: {netcdf_dir}")
    california_gdf = None
    if clip_to_shapefile:
        if os.path.exists(shapefile_path):
            california_gdf = gpd.read_file(shapefile_path)
        else:
            print(f"Warning: Shapefile {shapefile_path} not found. Cannot clip GridMET data.")
            # Proceed without clipping if shapefile is missing but clipping was requested
            # clip_to_shapefile = False 

    netcdf_files = [os.path.join(netcdf_dir, f"{gridmet_file_prefix}_{year}.nc") for year in range(start_year, end_year + 1)]
    all_data_list = []

    for file_path in netcdf_files:
        if os.path.exists(file_path):
            try:
                with xr.open_dataset(file_path, engine="netcdf4") as ds_single_year:
                    actual_data_var_in_file = None
                    # Try to find the variable using the heuristic
                    possible_vars = [v for v in ds_single_year.data_vars if gridmet_internal_var_heuristic.lower() in v.lower()]
                    if possible_vars:
                        actual_data_var_in_file = possible_vars[0] # Take the first match
                        if len(possible_vars) > 1:
                            print(f"  Note: Multiple potential variables found in {file_path} for heuristic '{gridmet_internal_var_heuristic}': {possible_vars}. Using '{actual_data_var_in_file}'.")
                    elif gridmet_file_prefix in ds_single_year.data_vars: # Fallback to direct prefix match
                        actual_data_var_in_file = gridmet_file_prefix
                    
                    if not actual_data_var_in_file:
                        print(f"  Warning: Could not determine data variable for '{gridmet_internal_var_heuristic}' or prefix '{gridmet_file_prefix}' in {file_path}. Available: {list(ds_single_year.data_vars)}. Skipping file.")
                        continue

                    data_year = ds_single_year[actual_data_var_in_file].copy() # copy to allow closing dataset
                    
                # Ensure CRS (GridMET is typically EPSG:4326)
                if data_year.rio.crs is None:
                    data_year = data_year.rio.write_crs("EPSG:4326", inplace=False)
                
                if clip_to_shapefile and california_gdf is not None:
                    # Reproject shapefile to data's CRS for clipping if necessary
                    if california_gdf.crs != data_year.rio.crs:
                        california_gdf_reproj = california_gdf.to_crs(data_year.rio.crs)
                    else:
                        california_gdf_reproj = california_gdf
                    data_year = data_year.rio.clip(california_gdf_reproj.geometry, drop=False, all_touched=True)
                
                data_year.name = gridmet_file_prefix # Standardize name for concatenation consistency
                all_data_list.append(data_year)
            except Exception as e_file:
                print(f"  Error processing GridMET file {file_path}: {e_file}")
        else:
            # print(f"Warning: GridMET file not found: {file_path}") # Can be verbose
            pass

    if not all_data_list:
        print(f"No GridMET data loaded for prefix '{gridmet_file_prefix}'.")
        return None
    
    # Determine time dimension for concatenation (usually 'day' for GridMET)
    time_dim_to_concat = 'day' 
    if time_dim_to_concat not in all_data_list[0].dims:
        # Try to find a dimension that looks like time if 'day' is not present
        possible_time_dims = [d for d in all_data_list[0].dims if 'time' in d.lower() or 'day' in d.lower()]
        if possible_time_dims:
            time_dim_to_concat = possible_time_dims[0]
            print(f"  Note: Using '{time_dim_to_concat}' as time dimension for GridMET concatenation.")
        else:
            print(f"Error: Could not determine time dimension for concatenation in GridMET data. First file dims: {all_data_list[0].dims}")
            return None
            
    try:
        combined_data = xr.concat(all_data_list, dim=time_dim_to_concat)
        if combined_data.rio.crs is None: # Ensure CRS on combined data
             combined_data = combined_data.rio.write_crs("EPSG:4326", inplace=False)
        print(f"GridMET gridded data for prefix '{gridmet_file_prefix}' loaded and combined. Shape: {combined_data.shape}")
        return combined_data
    except Exception as e_concat:
        print(f"Error concatenating GridMET data for prefix '{gridmet_file_prefix}': {e_concat}")
        return None


def reproject_and_match_rasters(source_raster, match_raster, 
                                output_path_mean_source, output_path_matched,
                                source_time_dim='time'):
    """
    Calculates mean of source_raster, saves it, then reprojects this mean to match_raster's grid.
    Args:
        source_raster (xr.DataArray): The raster to be reprojected.
        match_raster (xr.DataArray): The raster whose grid will be matched.
        output_path_mean_source (str): Path to save the time-mean of the source_raster (native projection).
        output_path_matched (str): Path to save the reprojected source_raster mean.
        source_time_dim (str): Name of the time dimension in source_raster.
    Returns:
        xr.DataArray or None: The reprojected DataArray, or None if error.
    """
    if source_raster is None or match_raster is None:
        print("Missing source_raster or match_raster for reprojection.")
        return None
    try:
        print(f"Calculating mean of source raster along dimension '{source_time_dim}'...")
        source_mean = source_raster.mean(dim=source_time_dim, skipna=True)
        
        # Inherit CRS from source_raster if source_mean doesn't have it
        if source_mean.rio.crs is None and source_raster.rio.crs is not None:
            source_mean = source_mean.rio.write_crs(source_raster.rio.crs, inplace=False)
        
        config.ensure_dir_exists(os.path.dirname(output_path_mean_source))
        source_mean.to_netcdf(output_path_mean_source)
        print(f"Saved mean source raster (native projection) to {output_path_mean_source}")

        if match_raster.rio.crs is None:
            print("Error: match_raster is missing CRS information, which is needed for reproject_match.")
            return None
            
        print(f"Reprojecting mean source raster to match grid of another raster (CRS: {match_raster.rio.crs})...")
        reprojected_raster = source_mean.rio.reproject_match(
            match_raster, 
            nodata=np.nan, 
            resampling=Resampling.bilinear
        )
        config.ensure_dir_exists(os.path.dirname(output_path_matched))
        reprojected_raster.to_netcdf(output_path_matched)
        print(f"Saved reprojected and matched raster to {output_path_matched}")
        return reprojected_raster
    except Exception as e:
        print(f"Error in reproject_and_match_rasters: {e}")
        return None

def extract_raster_data_at_station_locations(stations_gdf, # GeoDataFrame with station locations and 'StationNbr'
                                             spatial_gridded_da,   # Spatial CIMIS type xarray DataArray (time, y, x)
                                             gridmet_gridded_da,   # GridMET xarray DataArray (day, y, x)
                                             target_station_ids, # List/Series of station IDs (strings) to extract
                                             var_config,         # Current variable's configuration dict
                                             spatial_time_coord='time', 
                                             gridmet_time_coord='day'):
    """
    Extracts time series data from rasters for given station locations and applies scaling.
    Args:
        stations_gdf (gpd.GeoDataFrame): Station metadata including geometry.
        spatial_gridded_da (xr.DataArray, optional): Spatially gridded data (e.g., Spatial CIMIS).
        gridmet_gridded_da (xr.DataArray, optional): Spatially gridded data (e.g., GridMET).
        target_station_ids (iterable): Iterable of station ID strings to extract data for.
        var_config (dict): Configuration for the current variable, including scale factors.
        spatial_time_coord (str): Name of the time coordinate in spatial_gridded_da.
        gridmet_time_coord (str): Name of the time coordinate in gridmet_gridded_da.
    Returns:
        tuple (pd.DataFrame, pd.DataFrame): DataFrames for spatial and GridMET data extracted at stations.
                                            Empty DataFrames if issues or no data.
    """
    if stations_gdf is None or not list(target_station_ids): # Ensure target_station_ids is not empty
        print("Missing station GeoDataFrame or target_station_ids for raster extraction.")
        return pd.DataFrame(), pd.DataFrame()
    if spatial_gridded_da is None and gridmet_gridded_da is None:
        print("Both spatial_gridded_da and gridmet_gridded_da are None. Nothing to extract.")
        return pd.DataFrame(), pd.DataFrame()

    print("Extracting raster data for station locations...")
    spatial_station_series_list = []
    gridmet_station_series_list = []

    # Filter stations_gdf for only those IDs we need, ensuring StationNbr is string for matching
    target_stations_gdf = stations_gdf[stations_gdf['StationNbr'].astype(str).isin(map(str, target_station_ids))].copy()
    if target_stations_gdf.empty:
        print("No matching stations found in GeoDataFrame for the provided target IDs for extraction.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Process Spatial CIMIS type data ---
    if spatial_gridded_da is not None:
        if spatial_gridded_da.rio.crs is None:
            print("Warning: spatial_gridded_da is missing CRS. Cannot reproject stations for extraction.")
        else:
            stations_reproj_spatial = target_stations_gdf.to_crs(spatial_gridded_da.rio.crs)
            for station_id_str in map(str, target_station_ids):
                if station_id_str in stations_reproj_spatial['StationNbr'].astype(str).values:
                    try:
                        station_row = stations_reproj_spatial[stations_reproj_spatial['StationNbr'].astype(str) == station_id_str].iloc[0]
                        geom = station_row.geometry
                        
                        # Ensure x and y coordinate names are correctly identified from the raster
                        x_coord_name = spatial_gridded_da.rio.x_coordinate
                        y_coord_name = spatial_gridded_da.rio.y_coordinate

                        extracted_values = spatial_gridded_da.sel(
                            **{x_coord_name: geom.x, y_coord_name: geom.y}, method='nearest'
                        )
                        series = pd.Series(
                            extracted_values.data * var_config["spatial_cimis_scale_factor"], 
                            index=pd.to_datetime(extracted_values[spatial_time_coord].data), 
                            name=station_id_str
                        )
                        spatial_station_series_list.append(series)
                    except IndexError: # Should not happen if station_id_str is in values
                        print(f"  Warning: Station ID {station_id_str} found in list but not in reprojected GDF for Spatial data (IndexError).")
                    except Exception as e: 
                        print(f"  Error extracting Spatial data for station {station_id_str}: {e}")
    
    # --- Process GridMET type data ---
    if gridmet_gridded_da is not None:
        if gridmet_gridded_da.rio.crs is None:
            print("Warning: gridmet_gridded_da is missing CRS. Cannot reproject stations for extraction.")
        else:
            stations_reproj_gridmet = target_stations_gdf.to_crs(gridmet_gridded_da.rio.crs)
            for station_id_str in map(str, target_station_ids):
                if station_id_str in stations_reproj_gridmet['StationNbr'].astype(str).values:
                    try:
                        station_row = stations_reproj_gridmet[stations_reproj_gridmet['StationNbr'].astype(str) == station_id_str].iloc[0]
                        geom = station_row.geometry

                        x_coord_name = gridmet_gridded_da.rio.x_coordinate
                        y_coord_name = gridmet_gridded_da.rio.y_coordinate

                        extracted_values = gridmet_gridded_da.sel(
                            **{x_coord_name: geom.x, y_coord_name: geom.y}, method='nearest'
                        )
                        series = pd.Series(
                            extracted_values.data * var_config["gridmet_scale_factor"], 
                            index=pd.to_datetime(extracted_values[gridmet_time_coord].data), 
                            name=station_id_str
                        )
                        gridmet_station_series_list.append(series)
                    except IndexError:
                         print(f"  Warning: Station ID {station_id_str} found in list but not in reprojected GDF for GridMET data (IndexError).")
                    except Exception as e: 
                        print(f"  Error extracting GridMET data for station {station_id_str}: {e}")

    # Concatenate all series for each dataset type
    spatial_station_df = pd.concat(spatial_station_series_list, axis=1) if spatial_station_series_list else pd.DataFrame()
    gridmet_station_df = pd.concat(gridmet_station_series_list, axis=1) if gridmet_station_series_list else pd.DataFrame()
    
    if not spatial_station_df.empty: spatial_station_df.index.name = 'date'
    if not gridmet_station_df.empty: gridmet_station_df.index.name = 'date'

    print(f"Raster data extraction complete. Spatial data shape: {spatial_station_df.shape}, GridMET data shape: {gridmet_station_df.shape}")
    return spatial_station_df, gridmet_station_df
