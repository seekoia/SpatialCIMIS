# cimis_analysis_package/plotting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os
import geopandas as gpd
import skill_metrics as sm # Ensure skillmetrics is installed
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# Import configuration constants for defaults if needed
from . import config


def plot_variable_scatter_comparison(station_df, model_df, model_name, 
                                     var_name, var_unit, output_plot_dir):
    """
    Plots a scatter comparison between station observations and model data for a given variable.
    Args:
        station_df (pd.DataFrame): DataFrame of observed station data.
        model_df (pd.DataFrame): DataFrame of modeled data.
        model_name (str): Name of the model (e.g., "Spatial CIMIS", "GridMET").
        var_name (str): Short name of the variable (e.g., "Rad", "ETo").
        var_unit (str): Unit of the variable (e.g., "W/m²", "mm/day").
        output_plot_dir (str): Directory to save the plot.
    """
    if station_df is None or model_df is None or station_df.empty or model_df.empty:
        print(f"Cannot plot scatter for {model_name} ({var_name}): Empty or None dataframes provided.")
        return

    # Align dataframes by common dates (index) and common stations (columns)
    common_dates = station_df.index.intersection(model_df.index)
    common_stations = station_df.columns.intersection(model_df.columns)

    if common_dates.empty or not common_stations.any():
        print(f"No common dates or stations found between station_df and {model_name}_df for {var_name} scatter plot.")
        return

    stat_aligned = station_df.loc[common_dates, common_stations]
    model_aligned = model_df.loc[common_dates, common_stations]
    
    # For some variables (like radiation), zero might be a valid observation or a fill value.
    # Original script replaced 0 with NaN. This might need to be variable-specific.
    # For now, let's keep it, but consider parameterizing if 0 is valid for other vars.
    stat_aligned = stat_aligned.replace(0, np.nan) 
    model_aligned = model_aligned.replace(0, np.nan)

    # Melt DataFrames to get paired values
    stat_melt = pd.melt(stat_aligned.reset_index(), id_vars='date', var_name='station_id', value_name='station_value')
    model_melt = pd.melt(model_aligned.reset_index(), id_vars='date', var_name='station_id', value_name='model_value')
    
    # Merge the melted dataframes to ensure alignment
    merged_data = pd.merge(stat_melt, model_melt, on=['date', 'station_id']).dropna(subset=['station_value', 'model_value'])

    if merged_data.empty:
        print(f"No valid overlapping data points for scatter plot: {model_name} vs Station ({var_name}) after cleaning.")
        return

    observed_values = merged_data['station_value']
    modeled_values = merged_data['model_value']

    if len(observed_values) < 2: # Need at least 2 points for linregress
        print(f"Not enough data points ({len(observed_values)}) for linear regression: {model_name} vs Station ({var_name}).")
        return

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(observed_values, modeled_values)
        r_squared = r_value**2
    except ValueError as e:
        print(f"Error during linear regression for {model_name} vs Station ({var_name}): {e}")
        return

    plt.figure(figsize=(7, 7)) # Slightly larger for better readability
    plt.scatter(observed_values, modeled_values, marker='.', alpha=0.2, label=f'{model_name} (R²={r_squared:.2f})', s=10) # Smaller points, more transparent
    
    # Plot regression line using a range of observed values
    x_fit = np.array([observed_values.min(), observed_values.max()])
    y_fit = intercept + slope * x_fit
    plt.plot(x_fit, y_fit, color='red', linestyle='-', linewidth=2, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
    
    # Plot 1:1 line
    min_val = min(observed_values.min(), modeled_values.min())
    max_val = max(observed_values.max(), modeled_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1.5, label='1:1 Line')
    
    plt.xlabel(f"Station Observed {var_name} ({var_unit})", fontsize=12)
    plt.ylabel(f"{model_name} Modeled {var_name} ({var_unit})", fontsize=12)
    plt.title(f"Station Observations vs. {model_name} - Daily {var_name}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Ensure output directory exists
    config.ensure_dir_exists(output_plot_dir)
    plot_filename = os.path.join(output_plot_dir, f"scatter_station_vs_{model_name.lower().replace(' ', '_')}_{var_name}.pdf")
    try:
        plt.savefig(plot_filename, bbox_inches="tight")
        plt.close() # Close the figure to free memory
        print(f"Saved scatter plot to {plot_filename}")
        print(f"  Stats for {model_name} vs Station ({var_name}): R={r_value:.3f}, Slope={slope:.3f}, Intercept={intercept:.3f}")
    except Exception as e_save:
        print(f"Error saving scatter plot {plot_filename}: {e_save}")


def plot_seasonal_stats_maps(seasonal_stats_data, model_name, var_name, var_unit, 
                             shapefile_path, output_plot_dir):
    """
    Plots seasonal statistics (Correlation, Bias, MAE) on maps for a given variable.
    Args:
        seasonal_stats_data (gpd.GeoDataFrame or pd.DataFrame): (Geo)DataFrame with seasonal stats.
                                                               Must include 'Season', 'Correlation', 'Bias', 'MAE'.
                                                               If GeoDataFrame, must have 'geometry'.
        model_name (str): Name of the model.
        var_name (str): Short name of the variable.
        var_unit (str): Unit of the variable (used in title if relevant for bias/MAE).
        shapefile_path (str): Path to the background map shapefile (e.g., California).
        output_plot_dir (str): Directory to save the plots.
    """
    if seasonal_stats_data is None or seasonal_stats_data.empty:
        print(f"No seasonal stats data to plot for {model_name} ({var_name}).")
        return

    california_boundary = None
    if os.path.exists(shapefile_path):
        california_boundary = gpd.read_file(shapefile_path)
    else:
        print(f"Warning: Shapefile {shapefile_path} not found. Maps will be plotted without boundary.")

    is_geodataframe = isinstance(seasonal_stats_data, gpd.GeoDataFrame) and 'geometry' in seasonal_stats_data.columns

    seasons = seasonal_stats_data['Season'].unique()
    metrics_to_plot = ['Correlation', 'Bias', 'MAE']
    
    # Define colormap and potentially value ranges for each metric
    metric_plot_params = {
        'Correlation': {'cmap': 'cividis', 'vmin': 0, 'vmax': 1, 'legend_label': 'Correlation Coefficient'},
        'Bias': {'cmap': 'RdBu_r', 'legend_label': f'Bias ({var_unit})'}, # vmin/vmax will be dynamic
        'MAE': {'cmap': 'viridis', 'legend_label': f'MAE ({var_unit})'}    # vmin/vmax will be dynamic
    }

    for season in seasons:
        season_data_for_plot = seasonal_stats_data[seasonal_stats_data['Season'] == season]
        if season_data_for_plot.empty:
            continue

        fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot), figsize=(20, 7), constrained_layout=True) # Wider figure
        fig.suptitle(f"{model_name} - {var_name} - Season: {season}", fontsize=16, y=1.02) # Adjust title position

        for i, metric_key in enumerate(metrics_to_plot):
            ax = axes[i]
            plot_params = metric_plot_params.get(metric_key, {}).copy() # Get params, make a copy to modify
            
            # Dynamically set vmin/vmax for Bias and MAE if not predefined
            if metric_key == 'Bias' and ('vmin' not in plot_params or 'vmax' not in plot_params):
                abs_max_bias = season_data_for_plot[metric_key].abs().max()
                if pd.notna(abs_max_bias) and abs_max_bias > 0:
                    plot_params['vmin'], plot_params['vmax'] = -abs_max_bias, abs_max_bias
            elif metric_key == 'MAE' and ('vmin' not in plot_params or 'vmax' not in plot_params):
                 min_val, max_val = season_data_for_plot[metric_key].min(), season_data_for_plot[metric_key].max()
                 if pd.notna(min_val) and pd.notna(max_val) and min_val != max_val : #Ensure range exists
                    plot_params['vmin'], plot_params['vmax'] = min_val, max_val
            
            legend_label = plot_params.pop('legend_label', metric_key) # Use specific label or metric key

            if is_geodataframe and not season_data_for_plot['geometry'].isnull().all():
                # Plot California boundary if available
                if california_boundary is not None:
                    california_plot_boundary = california_boundary.to_crs(season_data_for_plot.crs) if california_boundary.crs != season_data_for_plot.crs else california_boundary
                    california_plot_boundary.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.7, zorder=1)
                
                if metric_key in season_data_for_plot.columns and not season_data_for_plot[metric_key].isnull().all():
                    season_data_for_plot.plot(column=metric_key, ax=ax, legend=True, 
                                              marker='s', markersize=80, zorder=2,
                                              legend_kwds={'label': legend_label, 'orientation': "horizontal", 'pad': 0.08, 'shrink':0.7},
                                              **plot_params) # Pass cmap, vmin, vmax
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_axis_off()
            else: # If not geodataframe or no valid geometries
                ax.text(0.5, 0.5, 'Station Data\n(No Geometry)', ha='center', va='center', transform=ax.transAxes, fontsize=10)
                # Optionally, plot a simple bar chart or table of the metric if no geometry
                if metric_key in season_data_for_plot.columns and not season_data_for_plot[metric_key].isnull().all():
                    season_data_for_plot.set_index('StationNbr')[metric_key].plot(kind='bar', ax=ax, color=plot_params.get('cmap','blue'))
                    ax.tick_params(axis='x', rotation=45, labelsize=8)


            ax.set_title(metric_key, fontsize=14)

        config.ensure_dir_exists(output_plot_dir)
        plot_filename = os.path.join(output_plot_dir, f"map_stats_{model_name.lower().replace(' ', '_')}_{var_name}_{season}.pdf")
        try:
            plt.savefig(plot_filename, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved seasonal stats map to {plot_filename}")
        except Exception as e_save:
            print(f"Error saving seasonal stats map {plot_filename}: {e_save}")


def aggregate_and_plot_station_timeseries(station_df, spatial_df, gridmet_df, taylor_list_gdf, 
                                          station_ids_to_plot, var_name, var_unit, output_plot_dir):
    """
    Aggregates data monthly/yearly and plots comparisons for selected stations for a given variable.
    Args:
        station_df, spatial_df, gridmet_df (pd.DataFrame): DataFrames for observed, Spatial CIMIS, GridMET.
        taylor_list_gdf (gpd.GeoDataFrame): Station metadata including 'Name'.
        station_ids_to_plot (list): List of station ID strings to plot.
        var_name (str): Short name of the variable.
        var_unit (str): Unit of the variable.
        output_plot_dir (str): Directory to save the plots.
    """
    if any(df is None or df.empty for df in [station_df, spatial_df, gridmet_df]) or \
       taylor_list_gdf is None or taylor_list_gdf.empty or not station_ids_to_plot:
        print(f"Missing data or station IDs for {var_name} timeseries aggregation and plotting. Skipping.")
        return

    # Ensure all station ID columns are strings for consistent indexing/lookup
    station_df.columns = station_df.columns.astype(str)
    spatial_df.columns = spatial_df.columns.astype(str)
    gridmet_df.columns = gridmet_df.columns.astype(str)
    # Ensure taylor_list_gdf index (if StationNbr) is also string if using .loc with string IDs
    if taylor_list_gdf.index.name == 'StationNbr': # Assuming StationNbr is the index
        taylor_list_gdf.index = taylor_list_gdf.index.astype(str)


    # --- Monthly Aggregation ---
    monthly_means = {
        'Station Obs.': station_df.groupby(station_df.index.month).mean(numeric_only=True),
        'Spatial CIMIS': spatial_df.groupby(spatial_df.index.month).mean(numeric_only=True),
        'GridMET': gridmet_df.groupby(gridmet_df.index.month).mean(numeric_only=True)
    }
    # --- Yearly Aggregation ---
    yearly_means = {
        'Station Obs.': station_df.groupby(station_df.index.year).mean(numeric_only=True),
        'Spatial CIMIS': spatial_df.groupby(spatial_df.index.year).mean(numeric_only=True),
        'GridMET': gridmet_df.groupby(gridmet_df.index.year).mean(numeric_only=True)
    }
    
    plot_colors = config.DEFAULT_PLOT_COLORS # Use colors from config

    for station_id_str in map(str, station_ids_to_plot): # Ensure IDs are strings
        # Check if station_id_str is a column in all relevant DataFrames
        if not all(station_id_str in df.columns for df_dict in [monthly_means, yearly_means] for df in df_dict.values()):
            print(f"Station {station_id_str} not found in all aggregated datasets for {var_name}. Skipping its timeseries plot.")
            continue
        
        station_name_info = station_id_str # Default
        try:
            # Try to get station name from taylor_list_gdf (assuming its index is StationNbr as string)
            if station_id_str in taylor_list_gdf.index:
                 station_name_info = taylor_list_gdf.loc[station_id_str, 'Name'] + f" (ID: {station_id_str})"
            elif 'StationNbr' in taylor_list_gdf.columns and station_id_str in taylor_list_gdf['StationNbr'].astype(str).values:
                 station_name_info = taylor_list_gdf[taylor_list_gdf['StationNbr'].astype(str) == station_id_str]['Name'].iloc[0] + f" (ID: {station_id_str})"

        except (KeyError, IndexError):
            print(f"  Note: Station name for ID {station_id_str} not found in taylor_list_gdf.")
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), constrained_layout=True) # Slightly larger
        fig.suptitle(f"{var_name} Time Series Comparison: {station_name_info}", fontsize=15, y=1.03)

        # --- Monthly Plot ---
        ax_monthly = axes[0]
        for key, df_mean in monthly_means.items():
            if station_id_str in df_mean.columns: # Check if station exists in this specific aggregated df
                ax_monthly.plot(df_mean.index, df_mean[station_id_str], marker='o', markersize=5, linestyle='-', 
                                color=plot_colors.get(key.split(' ')[0].lower(), 'grey'), label=key)
        ax_monthly.set_xlabel("Month", fontsize=12)
        ax_monthly.set_ylabel(f"Mean Daily {var_name} ({var_unit})", fontsize=12)
        ax_monthly.set_title("Average Monthly Cycle", fontsize=13)
        ax_monthly.legend(fontsize=10)
        ax_monthly.grid(True, linestyle=':', alpha=0.7)
        ax_monthly.xaxis.set_major_locator(ticker.MultipleLocator(1)) # Tick for every month
        ax_monthly.tick_params(axis='both', which='major', labelsize=10)

        # --- Yearly Plot ---
        ax_yearly = axes[1]
        for key, df_mean in yearly_means.items():
            if station_id_str in df_mean.columns:
                 ax_yearly.plot(df_mean.index, df_mean[station_id_str], marker='o', markersize=5, linestyle='-', 
                                color=plot_colors.get(key.split(' ')[0].lower(), 'grey'), label=key if ax_monthly.get_legend() is None else "_nolegend_") # Avoid duplicate legend items
        ax_yearly.set_xlabel("Year", fontsize=12)
        ax_yearly.set_ylabel(f"Mean Daily {var_name} ({var_unit})", fontsize=12)
        ax_yearly.set_title("Average Annual Time Series", fontsize=13)
        # ax_yearly.legend() # Optional: if monthly legend is sufficient
        ax_yearly.grid(True, linestyle=':', alpha=0.7)
        ax_yearly.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins='auto')) # Auto number of year ticks
        ax_yearly.tick_params(axis='both', which='major', labelsize=10)
        plt.setp(ax_yearly.get_xticklabels(), rotation=45, ha="right")


        config.ensure_dir_exists(output_plot_dir)
        plot_filename = os.path.join(output_plot_dir, f"timeseries_{var_name}_station_{station_id_str}.pdf")
        try:
            plt.savefig(plot_filename, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {var_name} timeseries plot for station {station_id_str} to {plot_filename}")
        except Exception as e_save:
            print(f"Error saving {var_name} timeseries plot for station {station_id_str}: {e_save}")


def plot_taylor_diagram_seasonal(spatial_stats_list, gridmet_stats_list, 
                                 var_name, var_unit, output_plot_dir):
    """
    Plots a seasonal Taylor diagram comparing Spatial CIMIS and GridMET for a given variable.
    Args:
        spatial_stats_list (list): List of (sdev, crmsd, ccoef) tuples for Spatial CIMIS, one per season.
        gridmet_stats_list (list): List of (sdev, crmsd, ccoef) tuples for GridMET, one per season.
        var_name (str): Short name of the variable.
        var_unit (str): Unit of the variable.
        output_plot_dir (str): Directory to save the plot.
    """
    if not spatial_stats_list and not gridmet_stats_list: # Check if both are empty or None
        print(f"Missing all stats for {var_name} Taylor diagram. Skipping plot.")
        return

    # Reference point (normalized: sdev=1, crmsd=0, ccoef=1)
    sdev_ref, crmsd_ref, ccoef_ref = np.array([1.0]), np.array([0.0]), np.array([1.0])
    
    season_markers = ['*', '^', 'o', 'P'] # For DJF, MAM, JJA, SON respectively
    season_labels = ['DJF', 'MAM', 'JJA', 'SON']
    
    # Prepare data for plotting, handling potential NaNs if a season had no data
    # sdev_sp, crmsd_sp, ccoef_sp will be lists of arrays, one array per model, containing seasonal values
    
    sdev_plot_spatial, crmsd_plot_spatial, ccoef_plot_spatial = [], [], []
    if spatial_stats_list:
        sdev_plot_spatial  = np.array([s[0] for s in spatial_stats_list])
        crmsd_plot_spatial = np.array([s[1] for s in spatial_stats_list])
        ccoef_plot_spatial = np.array([s[2] for s in spatial_stats_list])

    sdev_plot_gridmet, crmsd_plot_gridmet, ccoef_plot_gridmet = [], [], []
    if gridmet_stats_list:
        sdev_plot_gridmet  = np.array([s[0] for s in gridmet_stats_list])
        crmsd_plot_gridmet = np.array([s[1] for s in gridmet_stats_list])
        ccoef_plot_gridmet = np.array([s[2] for s in gridmet_stats_list])

    # Determine overall axis maximum for sdev and crmsd to ensure all points fit
    all_sdevs = np.concatenate( ([d for d in [sdev_plot_spatial, sdev_plot_gridmet] if len(d)>0] + [sdev_ref, [1.5]]) )
    all_crmsds = np.concatenate( ([d for d in [crmsd_plot_spatial, crmsd_plot_gridmet] if len(d)>0] + [crmsd_ref]) )
    
    axismax_sdev = np.nanmax(all_sdevs[np.isfinite(all_sdevs)]) if np.any(np.isfinite(all_sdevs)) else 1.5
    axismax_sdev = max(axismax_sdev, 1.0) # Ensure it's at least 1.0 for reference point

    # Max RMSD for tick range (can be different from sdev axis max)
    max_rmsd_tick = np.nanmax(all_crmsds[np.isfinite(all_crmsds)]) if np.any(np.isfinite(all_crmsds)) else 1.0
    max_rmsd_tick = max(max_rmsd_tick, 0.2) # Ensure at least one tick

    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
    fig = plt.figure(figsize=(8, 8)) # Slightly larger figure
    
    # Initialize Taylor Diagram with reference point
    # sm.taylor_diagram can be complex. The key is to plot reference first, then overlay models.
    # The sdevs, crmsds, ccoefs arguments in the first call are for the reference point.
    sm.taylor_diagram(sdev_ref, crmsd_ref, ccoef_ref, 
                      markerLabel='Reference', markerColor='black', markerSymbol='o', markerSize=10,
                      titleOBS='Normalized Reference', styleOBS='-', colOBS='black',
                      colRMS='darkgray', styleRMS='--', widthRMS=1.0, titleRMS='on', tickRMSangle=135.0,
                      colSTD='darkgray', styleSTD=':', widthSTD=1.0, titleSTD='on',
                      colCOR='darkgray', styleCOR='-.', widthCOR=1.0, titleCOR='on',
                      tickRMS=np.arange(0, max_rmsd_tick + 0.2, 0.2 if max_rmsd_tick > 0.5 else 0.1), 
                      tickSTD=np.arange(0, axismax_sdev + 0.2, 0.2 if axismax_sdev > 1 else 0.25),
                      tickCOR=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0]),
                      showlabelsRMS='on', showlabelsSTD='on', showlabelsCOR='on',
                      fig=fig, rect=111, axismax=axismax_sdev, extendaxis=False)

    # Plot points for Spatial CIMIS (Blue)
    if len(sdev_plot_spatial) > 0: # Check if there's data to plot
        for i in range(len(sdev_plot_spatial)): # Should be 4 seasons
            if not np.isnan(sdev_plot_spatial[i]): # Only plot if not NaN
                 sm.taylor_diagram(np.array([sdev_plot_spatial[i]]), 
                                   np.array([crmsd_plot_spatial[i]]), 
                                   np.array([ccoef_plot_spatial[i]]),
                                   markerSymbol=season_markers[i % len(season_markers)], 
                                   markerColor=config.DEFAULT_PLOT_COLORS.get('spatial','blue'), 
                                   markerSize=12,  
                                   overlay='on', fig=fig, alpha=0.8)

    # Plot points for GridMET (Red)
    if len(sdev_plot_gridmet) > 0:
        for i in range(len(sdev_plot_gridmet)):
            if not np.isnan(sdev_plot_gridmet[i]):
                sm.taylor_diagram(np.array([sdev_plot_gridmet[i]]), 
                                  np.array([crmsd_plot_gridmet[i]]), 
                                  np.array([ccoef_plot_gridmet[i]]),
                                  markerSymbol=season_markers[i % len(season_markers)], 
                                  markerColor=config.DEFAULT_PLOT_COLORS.get('gridmet','red'), 
                                  markerSize=12,  
                                  overlay='on', fig=fig, alpha=0.8)

    # Create a more robust legend
    legend_handles = [
        Line2D([0], [0], marker='s', color='w', label='Spatial CIMIS', 
               markerfacecolor=config.DEFAULT_PLOT_COLORS.get('spatial','blue'), markersize=10),
        Line2D([0], [0], marker='s', color='w', label='GridMET', 
               markerfacecolor=config.DEFAULT_PLOT_COLORS.get('gridmet','red'), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Reference',
               markerfacecolor='black', markeredgecolor='black', markersize=8) # Reference point
    ]
    # Add seasonal markers to legend
    for i, label in enumerate(season_labels):
        legend_handles.append(Line2D([0], [0], marker=season_markers[i], color='w', label=label, 
                                     markerfacecolor='dimgray', markersize=9))
    
    # Position legend outside the plot to avoid overlap
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.18, 0.95), fontsize='small', numpoints=1, title="Legend", title_fontsize="small")
    
    plt.suptitle(f"Seasonal Taylor Diagram (Normalized {var_name})", fontsize=14, y=0.98) # Main title
    
    config.ensure_dir_exists(output_plot_dir)
    plot_filename = os.path.join(output_plot_dir, f"taylor_diagram_seasonal_{var_name}.pdf")
    try:
        plt.savefig(plot_filename, bbox_inches="tight") # Use bbox_inches="tight"
        plt.close(fig)
        print(f"Saved {var_name} Taylor diagram to {plot_filename}")
    except Exception as e_save:
        print(f"Error saving {var_name} Taylor diagram {plot_filename}: {e_save}")

