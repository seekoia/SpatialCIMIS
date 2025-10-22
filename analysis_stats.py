# cimis_analysis_package/analysis_stats.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import skill_metrics as sm # Ensure skillmetrics is installed
import geopandas as gpd # For merging stats with geometry

# No direct config import needed here if var_name passed as arg

def calculate_seasonal_stats_for_stations(station_df, model_df, taylor_list_gdf, var_name):
    """
    Calculates seasonal correlation, bias, and MAE for each station for a given variable.
    Args:
        station_df (pd.DataFrame): DataFrame of observed station data.
        model_df (pd.DataFrame): DataFrame of modeled data (e.g., Spatial CIMIS or GridMET).
        taylor_list_gdf (gpd.GeoDataFrame): GeoDataFrame containing station info including geometry.
                                            Used to merge geometry with stats. Can be None if geometry not needed.
        var_name (str): Name of the variable being analyzed (for record-keeping in output).
    Returns:
        pd.DataFrame or gpd.GeoDataFrame: (Geo)DataFrame with seasonal statistics.
                                          Returns GeoDataFrame if taylor_list_gdf is provided and has geometry.
    """
    if station_df is None or model_df is None or station_df.empty or model_df.empty:
        print(f"Cannot calculate seasonal stats for {var_name}: station_df or model_df is empty or None.")
        return pd.DataFrame() # Return empty DataFrame

    all_seasons_stats_list = []
    seasons_map = {1: 'DJF', 2: 'MAM', 3: 'JJA', 4: 'SON'} # Standard season mapping

    for season_quarter in range(1, 5):
        # Determine months for the current season
        if season_quarter == 1: season_months = [12, 1, 2]  # DJF
        elif season_quarter == 2: season_months = [3, 4, 5] # MAM
        elif season_quarter == 3: season_months = [6, 7, 8] # JJA
        else: season_months = [9, 10, 11] # SON

        # Filter data for the current season
        stat_seasonal = station_df[station_df.index.month.isin(season_months)]
        model_seasonal = model_df[model_df.index.month.isin(season_months)]

        if stat_seasonal.empty or model_seasonal.empty:
            # print(f"  Skipping season {seasons_map[season_quarter]} for {var_name}: no data for station or model.")
            continue

        # Find common stations (columns) between observed and modeled seasonal data
        common_stations = stat_seasonal.columns.intersection(model_seasonal.columns)
        if not common_stations.any():
            # print(f"  Skipping season {seasons_map[season_quarter]} for {var_name}: no common stations.")
            continue
            
        # Align DataFrames to common stations and dates (though dates already filtered by season)
        stat_s = stat_seasonal[common_stations]
        model_s = model_seasonal[common_stations]

        # Calculate Pearson correlation
        correlations = stat_s.corrwith(model_s, axis=0, method='pearson')
        
        # Calculate bias (model - observed)
        biases = (model_s - stat_s).mean(skipna=True) # Ensure NaNs are skipped
        
        # Calculate Mean Absolute Error (MAE)
        maes = pd.Series(index=common_stations, dtype=float)
        for station_col in common_stations:
            # Align individual station series and drop NaNs for MAE calculation
            obs_series = stat_s[station_col].dropna()
            mod_series = model_s[station_col].dropna()
            common_idx_mae = obs_series.index.intersection(mod_series.index)
            
            if not common_idx_mae.empty:
                maes[station_col] = mean_absolute_error(obs_series[common_idx_mae], mod_series[common_idx_mae])
            else:
                maes[station_col] = np.nan
        
        # Create a DataFrame for the current season's stats
        season_stats_df = pd.DataFrame({
            'StationNbr': common_stations.astype(str), # Ensure station IDs are strings
            'Correlation': correlations,
            'Bias': biases,
            'MAE': maes,
            'Season': seasons_map[season_quarter],
            'Variable': var_name # Keep track of which variable these stats are for
        }).reset_index(drop=True) # Reset index to make StationNbr a regular column for merging
        
        all_seasons_stats_list.append(season_stats_df)

    if not all_seasons_stats_list:
        print(f"No seasonal stats computed for {var_name}.")
        return pd.DataFrame() # Return empty DataFrame
    
    # Concatenate stats from all seasons
    final_stats_df = pd.concat(all_seasons_stats_list, ignore_index=True)

    # Merge with geometry information if taylor_list_gdf is provided and valid
    if taylor_list_gdf is not None and isinstance(taylor_list_gdf, gpd.GeoDataFrame) and \
       'StationNbr' in taylor_list_gdf.columns and 'geometry' in taylor_list_gdf.columns:
        try:
            # Ensure StationNbr is string in both for merging
            taylor_list_geom = taylor_list_gdf[['StationNbr', 'geometry']].copy()
            taylor_list_geom['StationNbr'] = taylor_list_geom['StationNbr'].astype(str)
            
            final_stats_gdf = pd.merge(taylor_list_geom, final_stats_df, on='StationNbr', how='right')
            # Convert back to GeoDataFrame if merge was successful and geometry is present
            if 'geometry' in final_stats_gdf.columns:
                 # Drop rows where geometry might be NaN after a right merge if a station in stats_df wasn't in taylor_list_gdf
                final_stats_gdf = final_stats_gdf.dropna(subset=['geometry'])
                if not final_stats_gdf.empty:
                    return gpd.GeoDataFrame(final_stats_gdf, geometry='geometry', crs=taylor_list_gdf.crs)
                else: # If all geometries were NaN
                    print(f"Warning: All geometries were NaN after merging for {var_name}. Returning DataFrame.")
                    return final_stats_df.drop(columns=['geometry'], errors='ignore')

            else: # Should not happen if merge is correct
                return final_stats_df
        except Exception as e_merge:
            print(f"Error merging geometry for seasonal stats of {var_name}: {e_merge}. Returning DataFrame without geometry.")
            return final_stats_df
    else:
        # print(f"Taylor list GDF not provided or invalid; returning seasonal stats for {var_name} as DataFrame.")
        return final_stats_df


def calculate_taylor_stats_seasonal(station_df, model_df, model_name, var_name):
    """
    Calculates seasonal Taylor statistics (sdev, crmsd, ccoef) for a given variable.
    The statistics are calculated by pooling all station-day values for each season.
    Args:
        station_df (pd.DataFrame): Observed station data.
        model_df (pd.DataFrame): Modeled data.
        model_name (str): Name of the model (for context, not used in calculation directly).
        var_name (str): Name of the variable.
    Returns:
        tuple: (list of (sdev, crmsd, ccoef) tuples for model, list of reference_sdevs)
               Each list has 4 elements, one for each season (DJF, MAM, JJA, SON).
               NaNs are used if a season has insufficient data.
    """
    if station_df is None or model_df is None or station_df.empty or model_df.empty:
        print(f"Cannot calculate Taylor stats for {var_name} ({model_name}): station_df or model_df is empty/None.")
        return ([(np.nan, np.nan, np.nan)] * 4, [np.nan] * 4)

    all_seasons_taylor_stats = [] 
    all_seasons_ref_sdevs = []

    for season_quarter in range(1, 5):
        if season_quarter == 1: season_months = [12, 1, 2]
        elif season_quarter == 2: season_months = [3, 4, 5]
        elif season_quarter == 3: season_months = [6, 7, 8]
        else: season_months = [9, 10, 11]

        stat_seasonal_all = station_df[station_df.index.month.isin(season_months)]
        model_seasonal_all = model_df[model_df.index.month.isin(season_months)]

        if stat_seasonal_all.empty or model_seasonal_all.empty:
            # print(f"  Skipping Taylor stats for {var_name} ({model_name}), season {season_quarter}: no data.")
            all_seasons_taylor_stats.append((np.nan, np.nan, np.nan))
            all_seasons_ref_sdevs.append(np.nan)
            continue
            
        # Align dataframes by common indices (dates) and common columns (stations)
        common_idx = stat_seasonal_all.index.intersection(model_seasonal_all.index)
        common_cols = stat_seasonal_all.columns.intersection(model_seasonal_all.columns)

        if common_idx.empty or not common_cols.any():
            # print(f"  Skipping Taylor stats for {var_name} ({model_name}), season {season_quarter}: no common data points after initial alignment.")
            all_seasons_taylor_stats.append((np.nan, np.nan, np.nan))
            all_seasons_ref_sdevs.append(np.nan)
            continue

        # Select common data and stack to get a single series of all station-days
        stat_aligned_stacked = stat_seasonal_all.loc[common_idx, common_cols].stack(dropna=True) # dropna=True removes NaNs from stacking
        model_aligned_stacked = model_seasonal_all.loc[common_idx, common_cols].stack(dropna=True)
        
        # Final alignment of the stacked series (they should now have a (date, station) MultiIndex)
        final_common_multi_idx = stat_aligned_stacked.index.intersection(model_aligned_stacked.index)
        
        stat_final_series = stat_aligned_stacked.loc[final_common_multi_idx]
        model_final_series = model_aligned_stacked.loc[final_common_multi_idx]

        if len(stat_final_series) < 2: # Need at least 2 points for stats
            # print(f"  Skipping Taylor stats for {var_name} ({model_name}), season {season_quarter}: not enough common points after full alignment ({len(stat_final_series)}).")
            all_seasons_taylor_stats.append((np.nan, np.nan, np.nan))
            all_seasons_ref_sdevs.append(np.nan)
            continue
            
        try:
            # Calculate Taylor statistics (normalized by reference standard deviation if norm=True in sm.taylor_statistics)
            # The function sm.taylor_statistics by default normalizes sdev and crmsd by the reference sdev.
            # sdev returned is [sdev_ref/sdev_ref, sdev_mod/sdev_ref]
            # crmsd returned is [0.0, crmsd_mod/sdev_ref]
            # ccoef returned is [1.0, ccoef_mod_vs_ref]
            taylor_stats_results = sm.taylor_statistics(model_final_series.values, stat_final_series.values, field='data') # 'data' is a dummy field name
            
            model_sdev_normalized = taylor_stats_results['sdev'][1]    # Normalized std dev of the model
            model_crmsd_normalized = taylor_stats_results['crmsd'][1]  # Normalized centered RMS dev of the model
            model_ccoef = taylor_stats_results['ccoef'][1]             # Correlation coeff of model vs reference
            
            all_seasons_taylor_stats.append((model_sdev_normalized, model_crmsd_normalized, model_ccoef))
            all_seasons_ref_sdevs.append(taylor_stats_results['sdev'][0]) # This should be 1.0 if normalized, or actual ref sdev if not.
                                                                        # skill_metrics returns actual sdevs, then normalizes for plotting if requested.
                                                                        # Here, we are using the direct output.
        except Exception as e_taylor:
            print(f"  Error calculating Taylor stats for {var_name} ({model_name}), season {season_quarter}: {e_taylor}")
            all_seasons_taylor_stats.append((np.nan, np.nan, np.nan))
            all_seasons_ref_sdevs.append(np.nan)
            
    return all_seasons_taylor_stats, all_seasons_ref_sdevs
