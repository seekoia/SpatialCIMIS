#!/usr/bin/env python3
"""
Plot station time series comparisons for Spatial CIMIS and GridMET.

Creates multi-panel plots showing:
- Left column: Monthly climatology (mean across all years)
- Right column: Yearly time series

Each row shows one station with three lines:
- Spatial CIMIS (blue)
- GridMET (orange)
- Station observations (gray)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import sys

# Set plotting style
plt.style.use('default')
sns.set_context("paper")

# Define colors
COLORS = ['#006ba4', '#ff800e', '#ababab']  # blue, orange, gray


def load_data(station_file, spatial_file, gridmet_file):
    """Load all three datasets."""
    print("Loading data files...")
    
    # Load station observations
    station_data = pd.read_csv(station_file, index_col='date', parse_dates=True)
    # Keep columns as strings for consistency
    station_data.columns = station_data.columns.astype(str)
    print(f"  Station data: {station_data.shape}")
    
    # Load Spatial CIMIS predictions
    spatial_data = pd.read_csv(spatial_file, index_col='date', parse_dates=True)
    spatial_data.columns = spatial_data.columns.astype(str)
    print(f"  Spatial CIMIS: {spatial_data.shape}")
    
    # Load GridMET predictions
    gridmet_data = pd.read_csv(gridmet_file, index_col='date', parse_dates=True)
    gridmet_data.columns = gridmet_data.columns.astype(str)
    print(f"  GridMET: {gridmet_data.shape}")
    
    return station_data, spatial_data, gridmet_data


def load_station_metadata(station_list_file='CIMIS_Stations.csv'):
    """Load station names for plot titles."""
    print("Loading station metadata...")
    station_list = pd.read_csv(station_list_file)
    
    # Clean station names
    station_list['Name'] = station_list['Name'].str.replace(" ", "")
    station_list['Name'] = station_list['Name'].str.replace("/", "")
    station_list['Name'] = station_list['Name'].str.replace(".", "")
    station_list['Name'] = station_list['Name'].str.replace("-", "")
    
    return station_list


def compute_monthly_climatology(station_data, spatial_data, gridmet_data):
    """Compute monthly climatology (mean across all years)."""
    print("Computing monthly climatology...")
    
    station_monthly = station_data.groupby(station_data.index.month).mean(numeric_only=True)
    spatial_monthly = spatial_data.groupby(spatial_data.index.month).mean(numeric_only=True)
    gridmet_monthly = gridmet_data.groupby(gridmet_data.index.month).mean(numeric_only=True)
    
    station_monthly_std = station_data.groupby(station_data.index.month).std(numeric_only=True)
    spatial_monthly_std = spatial_data.groupby(spatial_data.index.month).std(numeric_only=True)
    gridmet_monthly_std = gridmet_data.groupby(gridmet_data.index.month).std(numeric_only=True)
    
    print(f"  Monthly climatology computed: {len(station_monthly.columns)} stations")
    
    return (station_monthly, spatial_monthly, gridmet_monthly,
            station_monthly_std, spatial_monthly_std, gridmet_monthly_std)


def compute_yearly_timeseries(station_data, spatial_data, gridmet_data, 
                              min_valid_days=60):
    """
    Compute yearly time series.
    Set years with too many missing days to NaN.
    Align all datasets to common years.
    """
    print("Computing yearly time series...")
    
    # Calculate yearly means
    yearly_means = station_data.groupby(station_data.index.year).mean(numeric_only=True)
    station_yearly = yearly_means.copy()
    
    # Mask years with too many NaN values
    if not yearly_means.empty:
        nan_counts = station_data.isnull().groupby(station_data.index.year).sum()
        relevant_nan_counts = nan_counts.reindex(columns=yearly_means.columns, fill_value=0)
        mask = relevant_nan_counts > min_valid_days
        station_yearly = station_yearly.mask(mask)
    
    spatial_yearly = spatial_data.groupby(spatial_data.index.year).mean(numeric_only=True)
    gridmet_yearly = gridmet_data.groupby(gridmet_data.index.year).mean(numeric_only=True)
    
    # Align to common years (intersection of all three)
    common_years = station_yearly.index.intersection(spatial_yearly.index).intersection(gridmet_yearly.index)
    station_yearly = station_yearly.loc[common_years]
    spatial_yearly = spatial_yearly.loc[common_years]
    gridmet_yearly = gridmet_yearly.loc[common_years]
    
    print(f"  Yearly data aligned to {len(common_years)} common years ({common_years.min()}-{common_years.max()})")
    
    return station_yearly, spatial_yearly, gridmet_yearly


def calculate_ylimits(station_list, station_monthly, spatial_monthly, gridmet_monthly,
                     station_yearly, spatial_yearly, gridmet_yearly):
    """Calculate consistent y-axis limits across all stations."""
    print("Calculating y-axis limits...")
    
    # For left plots (monthly): global min/max
    global_y_min_left = np.inf
    global_y_max_left = -np.inf
    
    # For right plots (yearly): max observed range
    max_observed_range_right = 0
    
    for sn in station_list:
        sn_str = str(sn)
        
        # Left plot data (monthly) - all columns are strings
        if sn_str in station_monthly.columns and sn_str in spatial_monthly.columns:
            current_min = np.nanmin([
                np.nanmin(spatial_monthly[sn_str].values),
                np.nanmin(gridmet_monthly[sn_str].values),
                np.nanmin(station_monthly[sn_str].values)
            ])
            current_max = np.nanmax([
                np.nanmax(spatial_monthly[sn_str].values),
                np.nanmax(gridmet_monthly[sn_str].values),
                np.nanmax(station_monthly[sn_str].values)
            ])
            global_y_min_left = min(global_y_min_left, current_min)
            global_y_max_left = max(global_y_max_left, current_max)
        
        # Right plot data (yearly) - all columns are strings
        if sn_str in station_yearly.columns and sn_str in spatial_yearly.columns:
            current_min = np.nanmin([
                np.nanmin(spatial_yearly[sn_str].values),
                np.nanmin(gridmet_yearly[sn_str].values),
                np.nanmin(station_yearly[sn_str].values)
            ])
            current_max = np.nanmax([
                np.nanmax(spatial_yearly[sn_str].values),
                np.nanmax(gridmet_yearly[sn_str].values),
                np.nanmax(station_yearly[sn_str].values)
            ])
            if np.isfinite(current_min) and np.isfinite(current_max):
                current_range = current_max - current_min
                max_observed_range_right = max(max_observed_range_right, current_range)
    
    # Add padding
    padding_left = (global_y_max_left - global_y_min_left) * 0.05
    if padding_left == 0:
        padding_left = 0.5
    final_y_min_left = global_y_min_left - padding_left
    final_y_max_left = global_y_max_left + padding_left
    
    if max_observed_range_right == 0:
        final_target_range_right = 1.0
    else:
        final_target_range_right = max_observed_range_right * 1.10
    
    return final_y_min_left, final_y_max_left, final_target_range_right


def plot_station_comparison(station_list, station_names,
                           station_monthly, spatial_monthly, gridmet_monthly,
                           station_yearly, spatial_yearly, gridmet_yearly,
                           output_file=None):
    """Create the multi-panel time series plot."""
    print("\nCreating plot...")
    
    num_rows = len(station_list)
    
    # Calculate y-axis limits
    final_y_min_left, final_y_max_left, final_target_range_right = calculate_ylimits(
        station_list, station_monthly, spatial_monthly, gridmet_monthly,
        station_yearly, spatial_yearly, gridmet_yearly
    )
    
    # Create figure
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(9, 9)
    fig.set_constrained_layout_pads(h_pad=1/100, hspace=0)
    
    raw_subfigs = fig.subfigures(nrows=num_rows, ncols=1)
    
    # Handle single vs multiple subfigures
    if num_rows == 1 and not isinstance(raw_subfigs, (list, np.ndarray)):
        subfigs_iterable = [raw_subfigs]
    else:
        subfigs_iterable = raw_subfigs
    
    # Plot each station
    for i, sn in enumerate(station_list):
        sn_str = str(sn)
        sn_int = int(sn)
        current_subfig = subfigs_iterable[i]
        
        ax = current_subfig.subplots(nrows=1, ncols=2)
        
        # Add title
        plot_letter = chr(ord('a') + i)
        title_content = f"{station_names.get(sn_int, f'Station {sn}')} ({sn})"
        full_title = f"{plot_letter}) {title_content}"
        ax[0].text(0, 1.02, full_title, transform=ax[0].transAxes,
                  ha='left', va='bottom', fontsize=12, fontweight='bold')
        
        # Check if station exists in data
        if sn_str not in station_monthly.columns:
            print(f"  Warning: Station {sn} not found in station data")
        if sn_str not in spatial_monthly.columns:
            print(f"  Warning: Station {sn} not found in spatial data")
        
        # --- Left plot: Monthly climatology ---
        # All columns are strings, so use sn_str for all datasets
        if sn_str in station_monthly.columns and sn_str in spatial_monthly.columns:
            ax[0].plot(gridmet_monthly.index, gridmet_monthly[sn_str].values,
                      color=COLORS[1], label='GridMET', linestyle='-', 
                      marker='o', zorder=2)
            ax[0].plot(station_monthly.index, station_monthly[sn_str].values,
                      color=COLORS[2], label='Station', linestyle='-',
                      marker='o', zorder=3)
            ax[0].plot(spatial_monthly.index, spatial_monthly[sn_str].values,
                      color=COLORS[0], label='Spatial CIMIS', linestyle='-',
                      marker='o', zorder=4)
            ax[0].set_ylabel('Mean Daily ETo (mm/day)', fontsize=10)
            ax[0].legend(fontsize=8)
            ax[0].set_ylim(final_y_min_left, final_y_max_left)
        
        # --- Right plot: Yearly time series ---
        # All columns are strings, so use sn_str for all datasets
        if sn_str in station_yearly.columns and sn_str in spatial_yearly.columns:
            ax[1].plot(station_yearly.index.values, spatial_yearly[sn_str].values,
                      '-o', color=COLORS[0], label='Spatial CIMIS')
            ax[1].plot(station_yearly.index.values, gridmet_yearly[sn_str].values,
                      '-o', color=COLORS[1], label='GridMET')
            ax[1].plot(station_yearly.index.values, station_yearly[sn_str].values,
                      '-o', color=COLORS[2], label='Station')
            ax[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            # Apply consistent range
            local_min = np.nanmin([
                np.nanmin(spatial_yearly[sn_str].values),
                np.nanmin(gridmet_yearly[sn_str].values),
                np.nanmin(station_yearly[sn_str].values)
            ])
            local_max = np.nanmax([
                np.nanmax(spatial_yearly[sn_str].values),
                np.nanmax(gridmet_yearly[sn_str].values),
                np.nanmax(station_yearly[sn_str].values)
            ])
            
            if np.isfinite(local_min) and np.isfinite(local_max):
                local_midpoint = (local_min + local_max) / 2
                plot_y_min = local_midpoint - (final_target_range_right / 2)
                plot_y_max = local_midpoint + (final_target_range_right / 2)
                ax[1].set_ylim(plot_y_min, plot_y_max)
            else:
                ax[1].set_ylim(0, final_target_range_right)
        
        # --- X-axis labels ---
        if i == num_rows - 1:  # Last row
            # Custom month labels for left plot
            if len(spatial_monthly.index) == 12:
                tick_positions = [spatial_monthly.index[idx] for idx in [1, 3, 5, 7, 9, 11]]
                tick_labels = ['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec']
                ax[0].set_xticks(tick_positions)
                ax[0].set_xticklabels(tick_labels)
        else:
            ax[0].set_xlabel(' ')
            ax[1].set_xlabel(' ')
            
            # Phantom ticks for alignment
            if len(spatial_monthly.index) == 12:
                phantom_ticks = [spatial_monthly.index[idx] for idx in [1, 3, 5, 7, 9, 11]]
                ax[0].set_xticks(phantom_ticks)
                ax[0].set_xticklabels([''] * len(phantom_ticks))
            else:
                ax[0].set_xticklabels([])
            
            ax[1].set_xticklabels([])
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function."""
    # Parse command line arguments
    station_list_arg = None
    output_file = 'station_timeseries.pdf'
    
    if len(sys.argv) > 1:
        # Check if first arg is station list or output file
        if sys.argv[1].endswith('.pdf') or sys.argv[1].endswith('.png'):
            output_file = sys.argv[1]
        else:
            station_list_arg = [int(s) for s in sys.argv[1].split(',')]
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("="*70)
    print("Station Time Series Comparison Plot")
    print("="*70)
    
    # File paths
    station_file = 'output/station_rs_data.csv'
    spatial_file = 'output/spatial_cimis_station_rs.csv'
    gridmet_file = 'output/gridmet_station_rs.csv'
    station_list_file = 'CIMIS_Stations.csv'
    
    # Load data
    station_data, spatial_data, gridmet_data = load_data(
        station_file, spatial_file, gridmet_file
    )
    
    # Load station names
    station_info = load_station_metadata(station_list_file)
    station_names = dict(zip(station_info['StationNbr'], station_info['Name']))
    
    # Default station list (from notebook)
    if station_list_arg is None:
        station_list = [158, 84, 117, 15, 136]
    else:
        station_list = station_list_arg
    
    print(f"\nPlotting stations: {station_list}")
    
    # Compute climatologies
    (station_monthly, spatial_monthly, gridmet_monthly,
     station_monthly_std, spatial_monthly_std, gridmet_monthly_std) = compute_monthly_climatology(
        station_data, spatial_data, gridmet_data
    )
    
    # Compute yearly time series
    station_yearly, spatial_yearly, gridmet_yearly = compute_yearly_timeseries(
        station_data, spatial_data, gridmet_data
    )
    
    # Create plot
    plot_station_comparison(
        station_list, station_names,
        station_monthly, spatial_monthly, gridmet_monthly,
        station_yearly, spatial_yearly, gridmet_yearly,
        output_file
    )
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

