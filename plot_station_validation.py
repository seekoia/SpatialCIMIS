#!/usr/bin/env python3
"""
Plot station validation metrics for Spatial CIMIS and GridMET.

Creates a 2x3 plot showing:
- Row 1: Spatial CIMIS (Correlation, Bias, MAE)
- Row 2: GridMET (Correlation, Bias, MAE)

Each metric is plotted on a California map with stations colored by their values.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from shapely.errors import ShapelyDeprecationWarning
import sys

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def load_station_metadata(station_list_file='CIMIS_Stations.csv'):
    """Load station metadata with coordinates."""
    print("Loading station metadata...")
    station_list = pd.read_csv(station_list_file)
    
    # Parse lat/lon if needed
    if station_list['HmsLatitude'].dtype == 'object':
        station_list['HmsLatitude'] = station_list['HmsLatitude'].str.split('/ ').str[-1]
        station_list['HmsLongitude'] = station_list['HmsLongitude'].str.split('/ ').str[-1]
    
    station_list['HmsLatitude'] = pd.to_numeric(station_list['HmsLatitude'], errors='coerce')
    station_list['HmsLongitude'] = pd.to_numeric(station_list['HmsLongitude'], errors='coerce')
    
    station_list = station_list.rename(columns={
        'HmsLatitude': 'Latitude',
        'HmsLongitude': 'Longitude'
    })
    
    print(f"  Loaded {len(station_list)} stations")
    return station_list


def load_data(station_file, spatial_file, gridmet_file):
    """Load all three datasets."""
    print("\nLoading data files...")
    
    # Load station observations
    station_data = pd.read_csv(station_file, index_col='date', parse_dates=True)
    print(f"  Station data: {station_data.shape}")
    
    # Load Spatial CIMIS predictions
    spatial_data = pd.read_csv(spatial_file, index_col='date', parse_dates=True)
    print(f"  Spatial CIMIS: {spatial_data.shape}")
    
    # Load GridMET predictions
    gridmet_data = pd.read_csv(gridmet_file, index_col='date', parse_dates=True)
    print(f"  GridMET: {gridmet_data.shape}")
    
    return station_data, spatial_data, gridmet_data


def calculate_statistics(station_data, spatial_data, gridmet_data):
    """Calculate correlation, bias, MAE, and RMSE for all stations."""
    print("\nCalculating statistics...")
    
    # Convert station columns to int for consistent indexing
    station_data.columns = station_data.columns.astype(np.int64)
    
    # Get common stations
    common_stations = list(set(station_data.columns) & 
                          set(spatial_data.columns.astype(int)) & 
                          set(gridmet_data.columns.astype(int)))
    
    print(f"  Found {len(common_stations)} common stations")
    
    # Initialize result series
    spatial_cor = pd.Series(index=common_stations, dtype=float)
    gridmet_cor = pd.Series(index=common_stations, dtype=float)
    spatial_bias = pd.Series(index=common_stations, dtype=float)
    gridmet_bias = pd.Series(index=common_stations, dtype=float)
    spatial_mae = pd.Series(index=common_stations, dtype=float)
    gridmet_mae = pd.Series(index=common_stations, dtype=float)
    spatial_rmse = pd.Series(index=common_stations, dtype=float)
    gridmet_rmse = pd.Series(index=common_stations, dtype=float)
    
    # Calculate for each station
    for station in common_stations:
        station_str = str(station)
        
        # Spatial CIMIS statistics
        valid_mask = station_data[station].notna() & spatial_data[station_str].notna()
        if valid_mask.sum() > 0:
            spatial_cor[station] = station_data[station][valid_mask].corr(
                spatial_data[station_str][valid_mask], method='pearson'
            )
            spatial_bias[station] = (spatial_data[station_str][valid_mask] - 
                                    station_data[station][valid_mask]).mean()
            spatial_mae[station] = mean_absolute_error(
                station_data[station][valid_mask],
                spatial_data[station_str][valid_mask]
            )
            spatial_rmse[station] = mean_squared_error(
                station_data[station][valid_mask],
                spatial_data[station_str][valid_mask],
                squared=False
            )
        
        # GridMET statistics
        valid_mask = station_data[station].notna() & gridmet_data[station_str].notna()
        if valid_mask.sum() > 0:
            gridmet_cor[station] = station_data[station][valid_mask].corr(
                gridmet_data[station_str][valid_mask], method='pearson'
            )
            gridmet_bias[station] = (gridmet_data[station_str][valid_mask] - 
                                    station_data[station][valid_mask]).mean()
            gridmet_mae[station] = mean_absolute_error(
                station_data[station][valid_mask],
                gridmet_data[station_str][valid_mask]
            )
            gridmet_rmse[station] = mean_squared_error(
                station_data[station][valid_mask],
                gridmet_data[station_str][valid_mask],
                squared=False
            )
    
    stats = {
        'spatCor': spatial_cor,
        'gridCor': gridmet_cor,
        'spatBias': spatial_bias,
        'gridBias': gridmet_bias,
        'spatMae': spatial_mae,
        'gridMae': gridmet_mae,
        'spatRmse': spatial_rmse,
        'gridRmse': gridmet_rmse
    }
    
    return stats, common_stations


def create_geodataframe(station_list, stats, common_stations):
    """Create GeoDataFrame with station locations and statistics."""
    print("\nCreating GeoDataFrame...")
    
    # Filter to common stations
    stations_subset = station_list[station_list['StationNbr'].isin(common_stations)].copy()
    
    # Create GeoDataFrame
    stations_gdf = gpd.GeoDataFrame(
        stations_subset,
        geometry=gpd.points_from_xy(stations_subset.Longitude, stations_subset.Latitude),
        crs="EPSG:4326"
    )
    
    # Add statistics
    for stat_name, stat_series in stats.items():
        stations_gdf[stat_name] = stations_gdf['StationNbr'].map(stat_series)
    
    print(f"  Created GeoDataFrame with {len(stations_gdf)} stations")
    
    return stations_gdf


def plot_validation(stations_gdf, shapefile_path='CA_State.shp', output_file=None):
    """Create the validation plot."""
    print("\nCreating plot...")
    
    # Load California boundary
    california = gpd.read_file(shapefile_path)
    california = california.to_crs("epsg:4326")
    
    # Create figure
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(15, 10)
    
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle('Spatial CIMIS', fontsize=16)
    
    # Spatial CIMIS plots
    (ax1, ax2, ax3) = subfigs[0].subplots(nrows=1, ncols=3)
    
    # Plot California boundaries
    california.plot(ax=ax1, color="white", edgecolor="black", linewidth=1)
    california.plot(ax=ax2, color="white", edgecolor="black", linewidth=1)
    california.plot(ax=ax3, color="white", edgecolor="black", linewidth=1)
    
    # Correlation
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax1, cmap='cividis', legend=True, cax=cax, 
                     column='spatCor', marker='s', markersize=70, 
                     vmin=0.5, vmax=1.0)
    ax1.set_title('Correlation', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    
    # Bias
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax2, cmap='RdBu_r', column='spatBias', cax=cax, 
                     legend=True, marker='s', markersize=70, 
                     vmin=-1.6, vmax=1.6)
    ax2.set_title('Bias (mm/day)', fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    # MAE
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax3, cmap='binary', column='spatMae', cax=cax, 
                     legend=True, marker='s', markersize=70, 
                     vmin=0.2, vmax=3.0)
    ax3.set_title('MAE (mm/day)', fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    # GridMET plots
    subfigs[1].suptitle('GridMET', fontsize=16)
    (ax1, ax2, ax3) = subfigs[1].subplots(nrows=1, ncols=3)
    
    # Plot California boundaries
    california.plot(ax=ax1, color="white", edgecolor="black", linewidth=1)
    california.plot(ax=ax2, color="white", edgecolor="black", linewidth=1)
    california.plot(ax=ax3, color="white", edgecolor="black", linewidth=1)
    
    # Correlation
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax1, cmap='cividis', legend=True, cax=cax, 
                     column='gridCor', marker='s', markersize=70, 
                     vmin=0.5, vmax=1.0)
    ax1.set_title('Correlation', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    
    # Bias
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax2, cmap='RdBu_r', column='gridBias', cax=cax, 
                     legend=True, marker='s', markersize=70, 
                     vmin=-1.6, vmax=1.6)
    ax2.set_title('Bias (mm/day)', fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    # MAE
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    stations_gdf.plot(ax=ax3, cmap='binary', column='gridMae', cax=cax, 
                     legend=True, marker='s', markersize=70, 
                     vmin=0.2, vmax=3.0)
    ax3.set_title('MAE (mm/day)', fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def print_summary_statistics(stats):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    
    print("\nGridMET Bias:")
    print(f"  Median: {stats['gridBias'].median():.3f} mm/day")
    print(f"  Max: {stats['gridBias'].max():.3f} mm/day")
    print(f"  Min: {stats['gridBias'].min():.3f} mm/day")
    
    print("\nSpatial CIMIS Bias:")
    print(f"  Median: {stats['spatBias'].median():.3f} mm/day")
    print(f"  Max: {stats['spatBias'].max():.3f} mm/day")
    print(f"  Min: {stats['spatBias'].min():.3f} mm/day")
    
    print("\nSpatial CIMIS Correlation:")
    print(f"  Mean: {stats['spatCor'].mean():.3f}")
    print(f"  Median: {stats['spatCor'].median():.3f}")
    
    print("\nGridMET Correlation:")
    print(f"  Mean: {stats['gridCor'].mean():.3f}")
    print(f"  Median: {stats['gridCor'].median():.3f}")
    
    print("\nSpatial CIMIS MAE:")
    print(f"  Mean: {stats['spatMae'].mean():.3f} mm/day")
    print(f"  Median: {stats['spatMae'].median():.3f} mm/day")
    
    print("\nGridMET MAE:")
    print(f"  Mean: {stats['gridMae'].mean():.3f} mm/day")
    print(f"  Median: {stats['gridMae'].median():.3f} mm/day")


def main():
    """Main function."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = 'station_validation.pdf'
    
    print("="*70)
    print("Station Validation Plot")
    print("="*70)
    
    # File paths
    station_file = 'output/station_eto_data.csv'
    spatial_file = 'output/spatial_cimis_station_eto.csv'
    gridmet_file = 'output/gridmet_station_eto.csv'
    station_list_file = 'CIMIS_Stations.csv'
    shapefile_path = 'CA_State.shp'
    
    # Load station metadata
    station_list = load_station_metadata(station_list_file)
    
    # Load data
    station_data, spatial_data, gridmet_data = load_data(
        station_file, spatial_file, gridmet_file
    )
    
    # Calculate statistics
    stats, common_stations = calculate_statistics(
        station_data, spatial_data, gridmet_data
    )
    
    # Create GeoDataFrame
    stations_gdf = create_geodataframe(station_list, stats, common_stations)
    
    # Print summary
    print_summary_statistics(stats)
    
    # Create plot
    plot_validation(stations_gdf, shapefile_path, output_file)
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

