# scripts/main_analysis.py

import os
import sys
import pandas as pd

# Add the project root to the Python path to allow importing from cimis_analysis_package
# This assumes 'scripts/' is one level down from 'cimis_analysis_project/'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from cimis_analysis_package import config
from cimis_analysis_package import data_fetching
from cimis_analysis_package import data_processing
from cimis_analysis_package import analysis_stats
from cimis_analysis_package import plotting

def run_analysis_workflow():
    """
    Main function to orchestrate the CIMIS and Gridded Data Analysis Workflow.
    """
    print("Starting CIMIS and Gridded Data Analysis Workflow...")

    # --- 1. Station List Preparation (once for all variables) ---
    # Use paths from config, but allow override if necessary
    station_csv_file = config.DEFAULT_STATION_CSV_PATH
    shapefile_main = config.DEFAULT_SHAPEFILE_PATH

    if not os.path.exists(station_csv_file):
        print(f"{station_csv_file} not found. Attempting to fetch from API...")
        data_fetching.fetch_and_save_station_list_from_api(output_csv_path=station_csv_file)
    
    full_stations_gdf = data_fetching.load_and_prepare_station_list(csv_path=station_csv_file)
    if full_stations_gdf is None or full_stations_gdf.empty:
        print("CRITICAL: Failed to load or prepare station list. Exiting workflow.")
        return

    active_stations_gdf = data_processing.filter_stations_by_date(full_stations_gdf)
    if active_stations_gdf is None or active_stations_gdf.empty:
        print("CRITICAL: No active stations found after filtering. Exiting workflow.")
        return

    # --- Loop through each variable defined in config.VARIABLES_TO_PROCESS ---
    for var_key_to_process in config.VARIABLES_TO_PROCESS:
        if var_key_to_process not in config.TARGET_VARIABLES_CONFIG:
            print(f"Warning: Variable key '{var_key_to_process}' not found in TARGET_VARIABLES_CONFIG. Skipping.")
            continue
        
        current_var_config = config.TARGET_VARIABLES_CONFIG[var_key_to_process]
        var_name_short = current_var_config["cimis_csv_output_name"]
        var_unit_label = current_var_config["unit"]
        print(f"\n{'='*15} Processing Variable: {var_name_short} ({var_unit_label}) {'='*15}")

        # Define variable-specific output paths
        current_output_netcdf_path = os.path.join(config.OUTPUT_NETCDF_PATH_BASE, var_name_short)
        current_output_csv_path = os.path.join(config.OUTPUT_CSV_PATH_BASE, var_name_short)
        current_output_plot_path = os.path.join(config.OUTPUT_PLOT_PATH_BASE, var_name_short)
        config.ensure_dir_exists(current_output_netcdf_path)
        config.ensure_dir_exists(current_output_csv_path)
        config.ensure_dir_exists(current_output_plot_path)

        # --- 2. Load Station Observation Data for the current variable ---
        print(f"\n--- Loading Station Observations for {var_name_short} ---")
        station_obs_df = data_processing.load_station_data_from_csvs(
            active_stations_gdf, 
            current_var_config,
            cimis_data_path=config.CIMIS_DATA_PATH, # from global config
            start_date_str=config.START_DATE_FILTER,
            end_date_str=config.END_DATE_FILTER
        )
        if station_obs_df.empty:
            print(f"No station observation data loaded for {var_name_short}. Many analyses will be skipped for this variable.")
        else:
            output_file = os.path.join(current_output_csv_path, f'station_{var_name_short}_observed_processed.csv')
            station_obs_df.to_csv(output_file)
            print(f"Saved processed station {var_name_short} observations to {output_file}")

        # --- 3. Create Taylor List (stations with observations for the current variable) ---
        print(f"\n--- Creating Taylor List for {var_name_short} ---")
        taylor_list_gdf_var = data_processing.get_taylor_list_stations(full_stations_gdf, station_obs_df)
        if taylor_list_gdf_var is None or taylor_list_gdf_var.empty:
            print(f"Taylor list for {var_name_short} is empty. Station-specific comparisons will be limited.")
        else:
            print(f"Taylor list for {var_name_short} created with {len(taylor_list_gdf_var)} stations.")


        # --- 4. Load Gridded Datasets for the current variable ---
        print(f"\n--- Loading Gridded Datasets for {var_name_short} ---")
        # Spatial CIMIS type data
        spatial_netcdf_pattern = os.path.join(
            config.NETCDF_SPATIAL_CIMIS_PATH, 
            f'spatial_cimis_{current_var_config["spatial_cimis_netcdf_var"]}_20*.nc' # Example pattern
        )
        spatial_gridded_da, _ = data_processing.load_spatial_gridded_data( # Seasonal part not used directly in main flow here
            spatial_netcdf_pattern,
            current_var_config["spatial_cimis_netcdf_var"]
        )
        
        # GridMET data
        gridmet_gridded_da = data_processing.load_gridmet_gridded_data(
            config.NETCDF_GRIDMET_PATH, 
            current_var_config["gridmet_netcdf_var"], # This is the file prefix
            current_var_config["gridmet_internal_var_name_heuristic"], # Heuristic for var name inside file
            shapefile_main
        )
        
        # Apply scaling factors immediately after loading
        if spatial_gridded_da is not None:
            spatial_gridded_da = spatial_gridded_da * current_var_config["spatial_cimis_scale_factor"]
        if gridmet_gridded_da is not None:
            gridmet_gridded_da = gridmet_gridded_da * current_var_config["gridmet_scale_factor"]

        # --- 5. Reproject and Match Rasters (if both gridded datasets are available) ---
        print(f"\n--- Reprojecting and Matching Rasters for {var_name_short} ---")
        # gridmet_gridded_mean_da = None # Initialize for broader scope
        # spatial_gridded_matched_da = None # Initialize
        if gridmet_gridded_da is not None and spatial_gridded_da is not None:
            # Assuming 'day' is the time dimension for GridMET for mean calculation
            gridmet_gridded_mean_da = gridmet_gridded_da.mean(dim="day", skipna=True) 
            if gridmet_gridded_mean_da.rio.crs is None: # Should be set by loader, but double check
                gridmet_gridded_mean_da = gridmet_gridded_mean_da.rio.write_crs("EPSG:4326", inplace=False)
            
            mean_gm_path = os.path.join(current_output_netcdf_path, f"gridmet_mean_{var_name_short}.nc")
            gridmet_gridded_mean_da.to_netcdf(mean_gm_path)
            print(f"Saved gridmet_mean_{var_name_short}.nc to {mean_gm_path}")

            # Reproject Spatial CIMIS mean to match GridMET mean grid
            spatial_gridded_matched_da = data_processing.reproject_and_match_rasters(
                source_raster=spatial_gridded_da, # Pass the full time-series spatial data
                match_raster=gridmet_gridded_mean_da,
                output_path_mean_source=os.path.join(current_output_netcdf_path, f"spatial_mean_{var_name_short}_native_proj.nc"),
                output_path_matched=os.path.join(current_output_netcdf_path, f"spatial_mean_{var_name_short}_matched_to_gridmet.nc"),
                source_time_dim='time' # Assuming 'time' for Spatial CIMIS time dimension
            )
        else:
            print(f"Skipping raster reprojection for {var_name_short} as one or both gridded datasets are missing.")
            gridmet_gridded_mean_da = None # Ensure it's None if not computed
            spatial_gridded_matched_da = None


        # --- 6. Extract Gridded Data at Station Locations ---
        print(f"\n--- Extracting Gridded Data at Station Locations for {var_name_short} ---")
        spatial_station_extracted_df = pd.DataFrame() # Initialize as empty
        gridmet_station_extracted_df = pd.DataFrame() # Initialize as empty

        if taylor_list_gdf_var is not None and not taylor_list_gdf_var.empty and not station_obs_df.empty:
            target_ids_for_extraction = station_obs_df.columns # Stations that have observations
            
            spatial_station_extracted_df, gridmet_station_extracted_df = \
                data_processing.extract_raster_data_at_station_locations(
                    taylor_list_gdf_var, # Use the variable-specific Taylor list GDF
                    spatial_gridded_da,   # Full time-series Spatial CIMIS DataArray (scaled)
                    gridmet_gridded_da,   # Full time-series GridMET DataArray (scaled)
                    target_ids_for_extraction, 
                    current_var_config # Pass the current variable's config for scale factors (already applied, but good for consistency)
                )
            
            if not spatial_station_extracted_df.empty:
                output_file = os.path.join(current_output_csv_path, f'spatial_{var_name_short}_extracted_at_stations.csv')
                spatial_station_extracted_df.to_csv(output_file)
                print(f"Saved extracted Spatial {var_name_short} data at stations to {output_file}")
            if not gridmet_station_extracted_df.empty:
                output_file = os.path.join(current_output_csv_path, f'gridmet_{var_name_short}_extracted_at_stations.csv')
                gridmet_station_extracted_df.to_csv(output_file)
                print(f"Saved extracted GridMET {var_name_short} data at stations to {output_file}")
        else:
            print(f"Skipping extraction of gridded {var_name_short} data at station locations due to missing pre-requisite data.")
            
        # --- 7. Perform Analyses and Generate Plots for the current variable ---
        print(f"\n--- Performing Analyses and Generating Plots for {var_name_short} ---")
        
        # Scatter Plots
        if not station_obs_df.empty:
            if not spatial_station_extracted_df.empty:
                plotting.plot_variable_scatter_comparison(
                    station_obs_df, spatial_station_extracted_df, "Spatial CIMIS", 
                    var_name_short, var_unit_label, current_output_plot_path
                )
            if not gridmet_station_extracted_df.empty:
                plotting.plot_variable_scatter_comparison(
                    station_obs_df, gridmet_station_extracted_df, "GridMET", 
                    var_name_short, var_unit_label, current_output_plot_path
                )
        
        # Seasonal Statistics and Maps
        if taylor_list_gdf_var is not None and not taylor_list_gdf_var.empty and not station_obs_df.empty:
            if not spatial_station_extracted_df.empty:
                print(f"\nCalculating seasonal stats for Spatial CIMIS ({var_name_short})...")
                spatial_seasonal_stats_data = analysis_stats.calculate_seasonal_stats_for_stations(
                    station_obs_df, spatial_station_extracted_df, taylor_list_gdf_var, var_name_short
                )
                if not spatial_seasonal_stats_data.empty:
                    plotting.plot_seasonal_stats_maps(
                        spatial_seasonal_stats_data, "SpatialCIMIS", var_name_short, var_unit_label, 
                        shapefile_main, current_output_plot_path
                    )

            if not gridmet_station_extracted_df.empty:
                print(f"\nCalculating seasonal stats for GridMET ({var_name_short})...")
                gridmet_seasonal_stats_data = analysis_stats.calculate_seasonal_stats_for_stations(
                    station_obs_df, gridmet_station_extracted_df, taylor_list_gdf_var, var_name_short
                )
                if not gridmet_seasonal_stats_data.empty:
                    plotting.plot_seasonal_stats_maps(
                        gridmet_seasonal_stats_data, "GridMET", var_name_short, var_unit_label, 
                        shapefile_main, current_output_plot_path
                    )
        
        # Station Time Series Plots
        if taylor_list_gdf_var is not None and not taylor_list_gdf_var.empty and \
           not station_obs_df.empty and \
           not spatial_station_extracted_df.empty and \
           not gridmet_station_extracted_df.empty:
            
            stations_to_plot_ts = []
            for st_id in config.DEFAULT_STATION_CANDIDATES_FOR_TIMESERIES: # Use candidates from config
                if all(st_id in df.columns for df in [station_obs_df, spatial_station_extracted_df, gridmet_station_extracted_df]):
                    stations_to_plot_ts.append(st_id)
            
            if not stations_to_plot_ts and not station_obs_df.columns.empty: # Fallback to first available station
                first_avail_st = str(station_obs_df.columns[0])
                if all(first_avail_st in df.columns for df in [spatial_station_extracted_df, gridmet_station_extracted_df]):
                    stations_to_plot_ts.append(first_avail_st)
            
            if stations_to_plot_ts:
                print(f"\nGenerating {var_name_short} time series plots for stations: {stations_to_plot_ts}")
                plotting.aggregate_and_plot_station_timeseries(
                    station_obs_df, spatial_station_extracted_df, gridmet_station_extracted_df,
                    taylor_list_gdf_var, stations_to_plot_ts, 
                    var_name_short, var_unit_label, current_output_plot_path
                )
            else:
                print(f"No suitable stations found for {var_name_short} time series plots (e.g., {config.DEFAULT_STATION_CANDIDATES_FOR_TIMESERIES} or first available).")

        # Taylor Diagrams
        if not station_obs_df.empty:
            print(f"\nCalculating Taylor Diagram statistics for {var_name_short}...")
            spatial_taylor_stats_seasonal, _ = analysis_stats.calculate_taylor_stats_seasonal(
                station_obs_df, spatial_station_extracted_df, "Spatial CIMIS", var_name_short
            )
            gridmet_taylor_stats_seasonal, _ = analysis_stats.calculate_taylor_stats_seasonal(
                station_obs_df, gridmet_station_extracted_df, "GridMET", var_name_short
            )
            
            plotting.plot_taylor_diagram_seasonal(
                spatial_taylor_stats_seasonal, gridmet_taylor_stats_seasonal, 
                var_name_short, var_unit_label, current_output_plot_path
            )
        
        print(f"\n--- Finished processing for variable: {var_name_short} ---")

    print("\nWorkflow execution finished for all selected variables.")

if __name__ == '__main__':
    run_analysis_workflow()
