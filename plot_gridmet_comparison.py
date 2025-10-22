#!/usr/bin/env python3
"""
Create comparison figure: Spatial CIMIS vs GridMET with bias

This script creates a 3-panel figure showing:
1. Spatial CIMIS climatology
2. GridMET climatology
3. Bias (GridMET - Spatial CIMIS)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import sys

try:
    import rioxarray as rio
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False
    print("Warning: rioxarray not available. Some features may be limited.")


def load_config(config_file='analysis_config.txt'):
    """Load configuration from a text file."""
    config = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Warning: Config file {config_file} not found.")
    return config


def main():
    """Main plotting function."""
    
    # Parse command line arguments - config file is required
    if len(sys.argv) < 2:
        print("Error: Config file argument is required.")
        print("Usage: python plot_gridmet_comparison.py <config_file>")
        print("Example: python plot_gridmet_comparison.py analysis_config_rs.txt")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    print("="*70)
    print("Spatial CIMIS vs GridMET Comparison Plot")
    print("="*70)
    
    # Load configuration
    config = load_config(config_file)
    
    # Variable is required in config
    if 'variable' not in config:
        print("Error: 'variable' not specified in config file")
        print("Add 'variable = Rs' (or ETo, Tx, Tn, etc.) to your config file")
        sys.exit(1)
    
    variable = config['variable']
    output_path = config.get('output_path', '/home/salba/SpatialCIMIS/output/')
    shapefile_path = config.get('shapefile_path', 'CA_State.shp')
    
    print(f"\nVariable: {variable}")
    print(f"Output path: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load Spatial CIMIS climatology
    print("\n1. Loading Spatial CIMIS climatology...")
    spatial_file = output_path + f'spatial_mean_{variable.lower()}.nc'
    
    if not os.path.exists(spatial_file):
        print(f"  Error: Spatial CIMIS file not found: {spatial_file}")
        print(f"  Run analysis script first!")
        sys.exit(1)
    
    spatial_ds = xr.open_dataset(spatial_file)
    
    # Get the data (handle different storage formats)
    if variable in spatial_ds.data_vars:
        spatial_data = spatial_ds[variable]
    else:
        # Get first non-coordinate variable
        data_vars = [v for v in spatial_ds.data_vars if v not in ['lat', 'lon', 'x', 'y', 'mask']]
        if data_vars:
            spatial_data = spatial_ds[data_vars[0]]
        else:
            spatial_data = spatial_ds.to_array().squeeze()
    
    if HAS_RIOXARRAY:
        # Set spatial dimensions before setting CRS
        spatial_data.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
        spatial_data.rio.write_crs("EPSG:3310", inplace=True)
        spatial_data.rio.write_nodata(np.nan, inplace=True)
    
    # Convert Rs from MJ/m²/day to W/m² to match GridMET units
    if variable == 'Rs':
        print(f"  Converting Rs from MJ/m²/day to W/m² (× 11.57)...")
        spatial_data = spatial_data * 11.57
    
    print(f"  Spatial CIMIS loaded: {spatial_data.shape}")
    
    # Load GridMET climatology
    print("\n2. Loading GridMET climatology...")
    gridmet_file = output_path + f'gridmet_mean_{variable.lower()}.nc'
    
    if not os.path.exists(gridmet_file):
        print(f"  Warning: GridMET file not found: {gridmet_file}")
        print(f"  Will skip GridMET comparison")
        gridmet_data = None
    else:
        gridmet_ds = xr.open_dataset(gridmet_file)
        
        # Get the data variable (skip metadata variables)
        if variable in gridmet_ds.data_vars:
            gridmet_data = gridmet_ds[variable]
        else:
            # Find the actual data variable (not spatial_ref, crs, etc.)
            data_vars = [v for v in gridmet_ds.data_vars 
                        if v not in ['lat', 'lon', 'x', 'y', 'spatial_ref', 'crs']]
            if data_vars:
                gridmet_data = gridmet_ds[data_vars[0]]
                print(f"  Using variable: {data_vars[0]}")
            else:
                gridmet_data = gridmet_ds.to_array().squeeze()
        
        if HAS_RIOXARRAY:
            # GridMET uses 'lon' and 'lat' as dimension names, not 'x' and 'y'
            # Check what dimensions exist
            dims = list(gridmet_data.dims)
            print(f"  GridMET dimensions: {dims}")
            
            # Set spatial dims based on what's available
            if 'lon' in dims and 'lat' in dims:
                gridmet_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
            elif 'x' in dims and 'y' in dims:
                gridmet_data.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
            
            gridmet_data.rio.write_crs("EPSG:4326", inplace=True)
            gridmet_data.rio.write_nodata(np.nan, inplace=True)
        
        print(f"  GridMET loaded: {gridmet_data.shape}")
    
    # Load California shapefile for clipping
    print("\n3. Loading California boundary...")
    if os.path.exists(shapefile_path):
        california = gpd.read_file(shapefile_path)
        california = california.to_crs("epsg:4326")
        print(f"  California boundary loaded")
    else:
        print(f"  Warning: Shapefile not found: {shapefile_path}")
        california = None
    
    # Reproject and match if both datasets available
    if gridmet_data is not None and HAS_RIOXARRAY:
        print("\n4. Reprojecting Spatial CIMIS to match GridMET...")
        
        from rasterio.enums import Resampling
        
        try:
            spatial_match = spatial_data.rio.reproject_match(
                gridmet_data,
                nodata=np.nan,
                resampling=Resampling.bilinear
            )
            
            print(f"  Reprojected to: {spatial_match.shape}")
            
            # Calculate bias
            print("\n5. Computing bias...")
            bias = gridmet_data - spatial_match
            bias.rio.write_crs("EPSG:4326", inplace=True)
            
            print(f"  Bias computed: range [{np.nanmin(bias.values):.2f}, {np.nanmax(bias.values):.2f}]")
            
        except Exception as e:
            print(f"  Warning: Could not reproject: {e}")
            spatial_match = spatial_data
            bias = None
    else:
        spatial_match = spatial_data
        bias = None
    
    # Clip to California shapefile boundary
    if california is not None and HAS_RIOXARRAY and gridmet_data is not None:
        print("\n6. Clipping to California boundary...")
        
        try:
            import gc
            
            # Clip gridmet first (it's in EPSG:4326, same as shapefile)
            gridmet_clipped = gridmet_data.rio.clip(california.geometry, california.crs, drop=False)
            print(f"  GridMET clipped")
            
            # Convert GridMET to numpy immediately and cleanup
            gridmet_values = gridmet_clipped.values.copy()
            del gridmet_clipped
            gc.collect()
            
            # Clip spatial match (reprojected to EPSG:4326)
            spatial_clipped = spatial_match.rio.clip(california.geometry, california.crs, drop=False)
            print(f"  Spatial CIMIS clipped")
            
            # Convert Spatial CIMIS to numpy immediately and cleanup
            spatial_values = spatial_clipped.values.copy()
            del spatial_clipped
            gc.collect()
            
            # Compute bias from numpy arrays (much more memory efficient)
            print(f"  Computing bias from clipped data...")
            if spatial_values is not None and gridmet_values is not None:
                bias_values = gridmet_values - spatial_values
            else:
                bias_values = None
            
        except Exception as e:
            print(f"  Warning: Clipping failed ({e}), using bounding box data...")
            spatial_values = spatial_match.values if spatial_match is not None else None
            gridmet_values = gridmet_data.values if gridmet_data is not None else None
            bias_values = bias.values if bias is not None else None
    else:
        print("\n6. Using bounding box data (shapefile or rioxarray not available)...")
        spatial_values = spatial_match.values if spatial_match is not None else None
        gridmet_values = gridmet_data.values if gridmet_data is not None else None
        bias_values = bias.values if bias is not None else None
    
    # Close datasets to free memory
    spatial_ds.close()
    if gridmet_data is not None:
        gridmet_ds.close()
    
    # Create the 3-panel figure
    print("\n7. Creating comparison plot...")
    
    fig, axes = plt.subplots(ncols=3, figsize=(16, 5), constrained_layout=True)
    
    # Variable-specific parameters
    var_params = {
        'ETo': {'vmin': 0, 'vmax': 7, 'cmap': 'afmhot', 'label': 'ETo (mm/day)'},
        'Tx': {'vmin': 0, 'vmax': 40, 'cmap': 'afmhot', 'label': 'Tx (°C)'},
        'Tn': {'vmin': -10, 'vmax': 25, 'cmap': 'afmhot', 'label': 'Tn (°C)'},
        'Rs': {'vmin': 150, 'vmax': 280, 'cmap': 'afmhot', 'label': 'Rs (W/m²)'}
    }
    
    params = var_params.get(variable, {'vmin': None, 'vmax': None, 'cmap': 'viridis', 'label': variable})
    
    # Panel 1: Spatial CIMIS
    if spatial_values is not None:
        im0 = axes[0].imshow(spatial_values, cmap=params['cmap'], 
                           vmin=params['vmin'], vmax=params['vmax'],
                           origin='upper', aspect='auto', interpolation='nearest')
        plt.colorbar(im0, ax=axes[0], label=params['label'], shrink=0.8)
    
    axes[0].set_title('Spatial CIMIS', fontsize=12, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    
    # Panel 2: GridMET
    if gridmet_values is not None:
        im1 = axes[1].imshow(gridmet_values, cmap=params['cmap'],
                           vmin=params['vmin'], vmax=params['vmax'],
                           origin='upper', aspect='auto', interpolation='nearest')
        plt.colorbar(im1, ax=axes[1], label=params['label'], shrink=0.8)
        axes[1].set_title('GridMET', fontsize=12, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'GridMET\nData Not Available', 
                    transform=axes[1].transAxes, ha='center', va='center',
                    fontsize=14, fontweight='bold')
        axes[1].set_title('GridMET', fontsize=12, fontweight='bold')
    
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    
    # Panel 3: Bias
    if bias_values is not None:
        # Center colormap at zero
        vmax_bias = max(abs(np.nanmin(bias_values)), abs(np.nanmax(bias_values)))
        im2 = axes[2].imshow(bias_values, cmap='RdBu_r',
                           vmin=-vmax_bias, vmax=vmax_bias,
                           origin='upper', aspect='auto', interpolation='nearest')
        plt.colorbar(im2, ax=axes[2], label=f'Bias ({params["label"]})', shrink=0.8)
        axes[2].set_title('GridMET - Spatial CIMIS', fontsize=12, fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'Bias\nNot Available', 
                    transform=axes[2].transAxes, ha='center', va='center',
                    fontsize=14, fontweight='bold')
        axes[2].set_title('GridMET - Spatial CIMIS', fontsize=12, fontweight='bold')
    
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].get_xaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)
    
    # Save figure as PNG only
    plot_file = output_path + f'{variable.lower()}_gridmet_spatial_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  PNG saved to: {plot_file}")
    
    # Close figure
    plt.close()
    
    print(f"\n{'='*70}")
    print("✓ Comparison plot complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

