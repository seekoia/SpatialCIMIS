#!/usr/bin/env python3
"""
Compare 2-week period in 2017 between spatial_cimis_station_eto.csv and 
spatial_cimis_station_eto_gridmet.csv for the 5 stations specified in analysis_config.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

def load_config():
    """Load station IDs from analysis_config.txt"""
    config_path = '/home/salba/SpatialCIMIS/analysis_config.txt'
    stations = []
    
    with open(config_path, 'r') as f:
        for line in f:
            if line.startswith('plot_stations'):
                # Extract station IDs from line like: plot_stations = 158,152,117,15,136
                station_line = line.split('=')[1].strip()
                stations = [int(x.strip()) for x in station_line.split(',')]
                break
    
    return stations

def load_data():
    """Load both CSV files"""
    spatial_path = '/home/salba/SpatialCIMIS/output/test/spatial_cimis_station_eto.csv'
    gridmet_path = '/home/salba/SpatialCIMIS/output/test/spatial_cimis_station_eto_gridmet.csv'
    
    print("Loading spatial CIMIS data...")
    spatial_df = pd.read_csv(spatial_path)
    spatial_df['date'] = pd.to_datetime(spatial_df['date'])
    
    print("Loading GridMET data...")
    gridmet_df = pd.read_csv(gridmet_path)
    gridmet_df['date'] = pd.to_datetime(gridmet_df['date'])
    
    return spatial_df, gridmet_df

def filter_2017_data(df, stations):
    """Filter data for 2017 and the 5 specified stations"""
    # Filter for 2017
    df_2017 = df[df['date'].dt.year == 2017].copy()
    
    # Select only the specified stations
    station_cols = ['date'] + [str(station) for station in stations]
    df_filtered = df_2017[station_cols].copy()
    
    return df_filtered

def select_2week_period(df, start_date='2017-07-01'):
    """Select a 2-week period starting from the specified date"""
    start = pd.to_datetime(start_date)
    end = start + timedelta(days=13)  # 14 days total (0-13)
    
    mask = (df['date'] >= start) & (df['date'] <= end)
    return df[mask].copy()

def compare_datasets(spatial_df, gridmet_df, stations):
    """Compare the two datasets for the specified stations"""
    results = {}
    
    for station in stations:
        station_str = str(station)
        
        if station_str in spatial_df.columns and station_str in gridmet_df.columns:
            # Get data for this station
            spatial_data = spatial_df[station_str].dropna()
            gridmet_data = gridmet_df[station_str].dropna()
            
            # Calculate statistics
            results[station] = {
                'spatial_mean': spatial_data.mean(),
                'spatial_std': spatial_data.std(),
                'gridmet_mean': gridmet_data.mean(),
                'gridmet_std': gridmet_data.std(),
                'correlation': np.corrcoef(spatial_data, gridmet_data)[0, 1] if len(spatial_data) == len(gridmet_data) else np.nan,
                'rmse': np.sqrt(np.mean((spatial_data - gridmet_data) ** 2)) if len(spatial_data) == len(gridmet_data) else np.nan,
                'bias': (spatial_data.mean() - gridmet_data.mean()) if len(spatial_data) > 0 and len(gridmet_data) > 0 else np.nan,
                'spatial_data': spatial_data,
                'gridmet_data': gridmet_data
            }
        else:
            print(f"Warning: Station {station} not found in one or both datasets")
    
    return results

def create_visualizations(spatial_df, gridmet_df, results, stations):
    """Create comparison visualizations"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('2-Week Period Comparison (July 1-14, 2017): Spatial CIMIS vs GridMET', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot 1: Time series comparison for each station
    ax1 = axes_flat[0]
    for i, station in enumerate(stations):
        station_str = str(station)
        if station_str in spatial_df.columns and station_str in gridmet_df.columns:
            ax1.plot(spatial_df['date'], spatial_df[station_str], 
                    label=f'CIMIS Station {station}', linewidth=2, alpha=0.8)
            ax1.plot(gridmet_df['date'], gridmet_df[station_str], 
                    label=f'GridMET Station {station}', linestyle='--', linewidth=2, alpha=0.8)
    
    ax1.set_title('Time Series Comparison')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('ETo (mm/day)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Scatter plot - CIMIS vs GridMET
    ax2 = axes_flat[1]
    for station in stations:
        station_str = str(station)
        if station_str in spatial_df.columns and station_str in gridmet_df.columns:
            ax2.scatter(spatial_df[station_str], gridmet_df[station_str], 
                       label=f'Station {station}', alpha=0.7, s=50)
    
    # Add 1:1 line
    min_val = min(spatial_df[[str(s) for s in stations]].min().min(), 
                  gridmet_df[[str(s) for s in stations]].min().min())
    max_val = max(spatial_df[[str(s) for s in stations]].max().max(), 
                  gridmet_df[[str(s) for s in stations]].max().max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1 Line')
    
    ax2.set_title('CIMIS vs GridMET Scatter Plot')
    ax2.set_xlabel('Spatial CIMIS ETo (mm/day)')
    ax2.set_ylabel('GridMET ETo (mm/day)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot comparison
    ax3 = axes_flat[2]
    data_for_box = []
    labels_for_box = []
    
    for station in stations:
        station_str = str(station)
        if station_str in spatial_df.columns and station_str in gridmet_df.columns:
            data_for_box.extend([spatial_df[station_str].dropna().values, 
                               gridmet_df[station_str].dropna().values])
            labels_for_box.extend([f'CIMIS {station}', f'GridMET {station}'])
    
    ax3.boxplot(data_for_box, labels=labels_for_box)
    ax3.set_title('Distribution Comparison')
    ax3.set_ylabel('ETo (mm/day)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics comparison
    ax4 = axes_flat[3]
    station_names = [f'Station {s}' for s in stations]
    cimis_means = [results[s]['spatial_mean'] for s in stations if s in results]
    gridmet_means = [results[s]['gridmet_mean'] for s in stations if s in results]
    
    x = np.arange(len(station_names))
    width = 0.35
    
    ax4.bar(x - width/2, cimis_means, width, label='Spatial CIMIS', alpha=0.8)
    ax4.bar(x + width/2, gridmet_means, width, label='GridMET', alpha=0.8)
    
    ax4.set_title('Mean ETo Comparison by Station')
    ax4.set_ylabel('Mean ETo (mm/day)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(station_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Correlation by station
    ax5 = axes_flat[4]
    correlations = [results[s]['correlation'] for s in stations if s in results and not np.isnan(results[s]['correlation'])]
    station_labels = [f'Station {s}' for s in stations if s in results and not np.isnan(results[s]['correlation'])]
    
    bars = ax5.bar(station_labels, correlations, alpha=0.8, color='skyblue')
    ax5.set_title('Correlation by Station')
    ax5.set_ylabel('Correlation Coefficient')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{corr:.3f}', ha='center', va='bottom')
    
    # Plot 6: RMSE and Bias comparison
    ax6 = axes_flat[5]
    rmse_values = [results[s]['rmse'] for s in stations if s in results and not np.isnan(results[s]['rmse'])]
    bias_values = [results[s]['bias'] for s in stations if s in results and not np.isnan(results[s]['bias'])]
    
    x = np.arange(len(station_labels))
    ax6_twin = ax6.twinx()
    
    bars1 = ax6.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8, color='lightcoral')
    bars2 = ax6_twin.bar(x + width/2, bias_values, width, label='Bias', alpha=0.8, color='lightgreen')
    
    ax6.set_title('RMSE and Bias by Station')
    ax6.set_ylabel('RMSE (mm/day)', color='lightcoral')
    ax6_twin.set_ylabel('Bias (mm/day)', color='lightgreen')
    ax6.set_xticks(x)
    ax6.set_xticklabels(station_labels)
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Add legends
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    return fig

def print_summary_statistics(results, stations):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS FOR 2-WEEK PERIOD (July 1-14, 2017)")
    print("="*80)
    
    print(f"{'Station':<10} {'CIMIS Mean':<12} {'GridMET Mean':<12} {'Correlation':<12} {'RMSE':<10} {'Bias':<10}")
    print("-"*80)
    
    for station in stations:
        if station in results:
            r = results[station]
            print(f"{station:<10} {r['spatial_mean']:<12.3f} {r['gridmet_mean']:<12.3f} "
                  f"{r['correlation']:<12.3f} {r['rmse']:<10.3f} {r['bias']:<10.3f}")
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    # Calculate overall statistics
    all_cimis = []
    all_gridmet = []
    
    for station in stations:
        if station in results:
            all_cimis.extend(results[station]['spatial_data'].values)
            all_gridmet.extend(results[station]['gridmet_data'].values)
    
    if all_cimis and all_gridmet:
        overall_corr = np.corrcoef(all_cimis, all_gridmet)[0, 1]
        overall_rmse = np.sqrt(np.mean((np.array(all_cimis) - np.array(all_gridmet)) ** 2))
        overall_bias = np.mean(all_cimis) - np.mean(all_gridmet)
        
        print(f"Overall Correlation: {overall_corr:.3f}")
        print(f"Overall RMSE: {overall_rmse:.3f} mm/day")
        print(f"Overall Bias: {overall_bias:.3f} mm/day")
        print(f"Mean CIMIS ETo: {np.mean(all_cimis):.3f} mm/day")
        print(f"Mean GridMET ETo: {np.mean(all_gridmet):.3f} mm/day")

def main():
    """Main function"""
    print("Starting comparison of 2-week period in 2017...")
    
    # Load configuration
    stations = load_config()
    print(f"Stations to analyze: {stations}")
    
    # Load data
    spatial_df, gridmet_df = load_data()
    
    # Filter for 2017 and selected stations
    spatial_2017 = filter_2017_data(spatial_df, stations)
    gridmet_2017 = filter_2017_data(gridmet_df, stations)
    
    print(f"Spatial CIMIS 2017 data shape: {spatial_2017.shape}")
    print(f"GridMET 2017 data shape: {gridmet_2017.shape}")
    
    # Select 2-week period (July 1-14, 2017)
    spatial_2week = select_2week_period(spatial_2017, '2017-07-01')
    gridmet_2week = select_2week_period(gridmet_2017, '2017-07-01')
    
    print(f"2-week period data shape - CIMIS: {spatial_2week.shape}, GridMET: {gridmet_2week.shape}")
    print(f"Date range: {spatial_2week['date'].min()} to {spatial_2week['date'].max()}")
    
    # Compare datasets
    results = compare_datasets(spatial_2week, gridmet_2week, stations)
    
    # Print summary statistics
    print_summary_statistics(results, stations)
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = create_visualizations(spatial_2week, gridmet_2week, results, stations)
    
    # Save the plot
    output_path = '/home/salba/SpatialCIMIS/output/test/2017_2week_comparison.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'Station': stations,
        'CIMIS_Mean': [results[s]['spatial_mean'] if s in results else np.nan for s in stations],
        'GridMET_Mean': [results[s]['gridmet_mean'] if s in results else np.nan for s in stations],
        'Correlation': [results[s]['correlation'] if s in results else np.nan for s in stations],
        'RMSE': [results[s]['rmse'] if s in results else np.nan for s in stations],
        'Bias': [results[s]['bias'] if s in results else np.nan for s in stations]
    })
    
    csv_output_path = '/home/salba/SpatialCIMIS/output/test/2017_2week_comparison_stats.csv'
    results_df.to_csv(csv_output_path, index=False)
    print(f"Detailed statistics saved to: {csv_output_path}")
    
    print("\nComparison completed successfully!")

if __name__ == "__main__":
    main()

