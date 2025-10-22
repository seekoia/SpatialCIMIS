#!/usr/bin/env python3
"""
Compare two ASCII raster files to check if they have the same spatial footprint.

This script compares:
- Number of columns (ncols)
- Number of rows (nrows)
- Lower-left corner x coordinate (xllcorner)
- Lower-left corner y coordinate (yllcorner)
- Cell size (cellsize)
- NODATA value (optional check)

Usage:
    python compare_ascii_footprints.py <file1.asc> <file2.asc>
"""

import sys
import rasterio
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def read_ascii_metadata(filepath):
    """Read ASCII raster file metadata using rasterio."""
    print(f"\nReading: {filepath}")
    
    with rasterio.open(filepath) as src:
        metadata = {
            'ncols': src.width,
            'nrows': src.height,
            'xllcorner': src.bounds.left,
            'yllcorner': src.bounds.bottom,
            'xurcorner': src.bounds.right,
            'yurcorner': src.bounds.top,
            'cellsize_x': src.res[0],
            'cellsize_y': src.res[1],
            'nodata': src.nodata,
            'crs': src.crs,
            'bounds': src.bounds,
            'transform': src.transform
        }
    
    return metadata


def read_ascii_data(filepath):
    """Read ASCII raster data using rasterio."""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        # Replace nodata with NaN
        if src.nodata is not None:
            data = data.astype(float)
            data[data == src.nodata] = np.nan
        return data


def print_metadata(metadata, label):
    """Print metadata in a readable format."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"  Dimensions:")
    print(f"    ncols: {metadata['ncols']}")
    print(f"    nrows: {metadata['nrows']}")
    print(f"  Extent:")
    print(f"    xllcorner: {metadata['xllcorner']:.10f}")
    print(f"    yllcorner: {metadata['yllcorner']:.10f}")
    print(f"    xurcorner: {metadata['xurcorner']:.10f}")
    print(f"    yurcorner: {metadata['yurcorner']:.10f}")
    print(f"  Resolution:")
    print(f"    cellsize_x: {metadata['cellsize_x']:.10f}")
    print(f"    cellsize_y: {metadata['cellsize_y']:.10f}")
    print(f"  Other:")
    print(f"    nodata: {metadata['nodata']}")
    print(f"    crs: {metadata['crs']}")


def compare_footprints(meta1, meta2, tolerance=1e-6):
    """Compare two footprints and return comparison results."""
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    
    matches = True
    details = []
    
    # Compare dimensions
    if meta1['ncols'] != meta2['ncols']:
        matches = False
        details.append(f"  ❌ ncols: {meta1['ncols']} vs {meta2['ncols']}")
    else:
        details.append(f"  ✓ ncols: {meta1['ncols']} (match)")
    
    if meta1['nrows'] != meta2['nrows']:
        matches = False
        details.append(f"  ❌ nrows: {meta1['nrows']} vs {meta2['nrows']}")
    else:
        details.append(f"  ✓ nrows: {meta1['nrows']} (match)")
    
    # Compare extent (with tolerance for floating point)
    xll_diff = abs(meta1['xllcorner'] - meta2['xllcorner'])
    if xll_diff > tolerance:
        matches = False
        details.append(f"  ❌ xllcorner: {meta1['xllcorner']:.10f} vs {meta2['xllcorner']:.10f} (diff: {xll_diff:.10e})")
    else:
        details.append(f"  ✓ xllcorner: {meta1['xllcorner']:.10f} (match)")
    
    yll_diff = abs(meta1['yllcorner'] - meta2['yllcorner'])
    if yll_diff > tolerance:
        matches = False
        details.append(f"  ❌ yllcorner: {meta1['yllcorner']:.10f} vs {meta2['yllcorner']:.10f} (diff: {yll_diff:.10e})")
    else:
        details.append(f"  ✓ yllcorner: {meta1['yllcorner']:.10f} (match)")
    
    xur_diff = abs(meta1['xurcorner'] - meta2['xurcorner'])
    if xur_diff > tolerance:
        matches = False
        details.append(f"  ❌ xurcorner: {meta1['xurcorner']:.10f} vs {meta2['xurcorner']:.10f} (diff: {xur_diff:.10e})")
    else:
        details.append(f"  ✓ xurcorner: {meta1['xurcorner']:.10f} (match)")
    
    yur_diff = abs(meta1['yurcorner'] - meta2['yurcorner'])
    if yur_diff > tolerance:
        matches = False
        details.append(f"  ❌ yurcorner: {meta1['yurcorner']:.10f} vs {meta2['yurcorner']:.10f} (diff: {yur_diff:.10e})")
    else:
        details.append(f"  ✓ yurcorner: {meta1['yurcorner']:.10f} (match)")
    
    # Compare cell size
    cellsize_x_diff = abs(meta1['cellsize_x'] - meta2['cellsize_x'])
    if cellsize_x_diff > tolerance:
        matches = False
        details.append(f"  ❌ cellsize_x: {meta1['cellsize_x']:.10f} vs {meta2['cellsize_x']:.10f} (diff: {cellsize_x_diff:.10e})")
    else:
        details.append(f"  ✓ cellsize_x: {meta1['cellsize_x']:.10f} (match)")
    
    cellsize_y_diff = abs(meta1['cellsize_y'] - meta2['cellsize_y'])
    if cellsize_y_diff > tolerance:
        matches = False
        details.append(f"  ❌ cellsize_y: {meta1['cellsize_y']:.10f} vs {meta2['cellsize_y']:.10f} (diff: {cellsize_y_diff:.10e})")
    else:
        details.append(f"  ✓ cellsize_y: {meta1['cellsize_y']:.10f} (match)")
    
    # Compare NODATA (optional - might be different)
    if meta1['nodata'] != meta2['nodata']:
        details.append(f"  ⚠ nodata: {meta1['nodata']} vs {meta2['nodata']} (different, but not critical)")
    else:
        details.append(f"  ✓ nodata: {meta1['nodata']} (match)")
    
    # Compare CRS
    if str(meta1['crs']) != str(meta2['crs']):
        matches = False
        details.append(f"  ❌ crs: {meta1['crs']} vs {meta2['crs']}")
    else:
        details.append(f"  ✓ crs: {meta1['crs']} (match)")
    
    # Print results
    for detail in details:
        print(detail)
    
    print(f"\n{'='*70}")
    if matches:
        print("✅ RESULT: Footprints MATCH!")
        print("The two ASCII files have the same spatial footprint.")
    else:
        print("❌ RESULT: Footprints DO NOT MATCH!")
        print("The two ASCII files have different spatial footprints.")
    print(f"{'='*70}\n")
    
    return matches


def plot_overlay_comparison(file1, file2, data1, data2, output_file='ascii_comparison.png'):
    """
    Create an overlay plot of both ASCII files with 50% transparency.
    
    Parameters:
    -----------
    file1, file2 : Path
        File paths
    data1, data2 : ndarray
        Raster data arrays
    output_file : str
        Output filename for the plot
    """
    print(f"\n{'='*70}")
    print("CREATING OVERLAY PLOT")
    print(f"{'='*70}")
    
    # Calculate statistics for color scaling
    valid1 = data1[~np.isnan(data1)]
    valid2 = data2[~np.isnan(data2)]
    
    if len(valid1) == 0 or len(valid2) == 0:
        print("Warning: One or both files contain no valid data. Skipping plot.")
        return
    
    vmin = min(np.nanmin(data1), np.nanmin(data2))
    vmax = max(np.nanmax(data1), np.nanmax(data2))
    
    print(f"\nOriginal data ranges:")
    print(f"  File 1: {np.nanmin(data1):.4f} to {np.nanmax(data1):.4f}")
    print(f"  File 2: {np.nanmin(data2):.4f} to {np.nanmax(data2):.4f}")
    print(f"  Combined range: {vmin:.4f} to {vmax:.4f}")
    
    # Normalize each dataset to 0-1 independently
    print(f"\nNormalizing each dataset to 0-1...")
    data1_norm = (data1 - np.nanmin(data1)) / (np.nanmax(data1) - np.nanmin(data1))
    data2_norm = (data2 - np.nanmin(data2)) / (np.nanmax(data2) - np.nanmin(data2))
    
    print(f"  File 1 normalized: {np.nanmin(data1_norm):.4f} to {np.nanmax(data1_norm):.4f}")
    print(f"  File 2 normalized: {np.nanmin(data2_norm):.4f} to {np.nanmax(data2_norm):.4f}")
    
    # Create figure with single overlay plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot Overlay with transparency using normalized data
    # Use different colormaps to distinguish the files
    # Plot File 2 first (Blues), then File 1 (Reds) on top
    im3b = ax.imshow(data2_norm, cmap='Blues', alpha=0.5, vmin=0, vmax=1,
                     origin='upper', interpolation='nearest')
    im3a = ax.imshow(data1_norm, cmap='Reds', alpha=0.5, vmin=0, vmax=1,
                     origin='upper', interpolation='nearest')
    ax.set_title('Overlay (Blue=File2, Red=File1)\n(Both Normalized 0-1)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Add colorbar for normalized scale
    cbar = plt.colorbar(im3a, ax=ax, label='Normalized Value (0-1)', shrink=0.8)
    cbar.set_alpha(1.0)
    cbar.draw_all()
    
    # Add a custom legend for the overlay
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, label=f'File2: {file2.name}'),
        Patch(facecolor='red', alpha=0.5, label=f'File1: {file1.name}'),
        Patch(facecolor='purple', alpha=0.7, label='Overlap')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    plt.close()
    
    # Also create a difference plot
    print("\nCreating difference plot...")
    
    # Check if shapes match
    if data1.shape == data2.shape:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calculate difference using normalized data
        diff = data2_norm - data1_norm
        
        # Plot difference
        diff_vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
        im_diff = axes[0].imshow(diff, cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax,
                                origin='upper', interpolation='nearest')
        axes[0].set_title('Difference (File2 - File1)\n(After Normalization)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Column', fontsize=11)
        axes[0].set_ylabel('Row', fontsize=11)
        plt.colorbar(im_diff, ax=axes[0], label='Normalized Difference', shrink=0.8)
        
        # Plot histogram of differences
        valid_diff = diff[~np.isnan(diff)]
        if len(valid_diff) > 0:
            axes[1].hist(valid_diff, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
            axes[1].axvline(np.nanmean(diff), color='orange', linestyle='--', linewidth=2, 
                           label=f'Mean: {np.nanmean(diff):.4f}')
            axes[1].set_xlabel('Normalized Difference Value', fontsize=11)
            axes[1].set_ylabel('Frequency', fontsize=11)
            axes[1].set_title('Distribution of Differences\n(Normalized Data)', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            print(f"  Difference statistics (normalized data):")
            print(f"    Mean: {np.nanmean(diff):.6f}")
            print(f"    Std Dev: {np.nanstd(diff):.6f}")
            print(f"    Min: {np.nanmin(diff):.6f}")
            print(f"    Max: {np.nanmax(diff):.6f}")
            print(f"    RMSE: {np.sqrt(np.nanmean(diff**2)):.6f}")
        
        plt.tight_layout()
        
        diff_output = output_file.replace('.png', '_difference.png')
        plt.savefig(diff_output, dpi=300, bbox_inches='tight')
        print(f"  Difference plot saved to: {diff_output}")
        
        plt.close()
    else:
        print("  Skipping difference plot (different dimensions)")


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python compare_ascii_footprints.py <file1.asc> <file2.asc>")
        sys.exit(1)
    
    file1 = Path(sys.argv[1])
    file2 = Path(sys.argv[2])
    
    # Check if files exist
    if not file1.exists():
        print(f"Error: File not found: {file1}")
        sys.exit(1)
    
    if not file2.exists():
        print(f"Error: File not found: {file2}")
        sys.exit(1)
    
    print("="*70)
    print("ASCII RASTER FOOTPRINT COMPARISON")
    print("="*70)
    print(f"\nFile 1: {file1}")
    print(f"File 2: {file2}")
    
    # Read metadata
    try:
        meta1 = read_ascii_metadata(file1)
        meta2 = read_ascii_metadata(file2)
    except Exception as e:
        print(f"\nError reading files: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print metadata
    print_metadata(meta1, "FILE 1 METADATA")
    print_metadata(meta2, "FILE 2 METADATA")
    
    # Compare footprints
    matches = compare_footprints(meta1, meta2)
    
    # Read data and create overlay plots
    try:
        print("\nReading raster data for plotting...")
        data1 = read_ascii_data(file1)
        data2 = read_ascii_data(file2)
        
        # Create plots
        output_file = 'ascii_comparison.png'
        plot_overlay_comparison(file1, file2, data1, data2, output_file)
        
    except Exception as e:
        print(f"\nWarning: Could not create plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Exit with appropriate code
    sys.exit(0 if matches else 1)


if __name__ == "__main__":
    main()

