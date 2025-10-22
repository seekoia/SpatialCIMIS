# Spatial CIMIS Processing

A comprehensive Python package for processing Spatial CIMIS data with grid standardization to handle varying ASCII file formats.

## Overview

This repository contains scripts for processing Spatial CIMIS ASCII files into standardized NetCDF format, handling different resolutions, extents, and grid sizes that occur throughout the dataset.

## Key Features

- **Grid Standardization**: Uses `rasterio.warp.reproject()` to standardize all data to a consistent 500×552 grid at 2km resolution
- **Variable Processing**: Supports Tx, ETo, Tn, Rs, and other CIMIS variables
- **GridMET Integration**: Compare Spatial CIMIS data with GridMET gridded meteorological data
- **Station Analysis**: Extract time series at CIMIS station locations
- **SLURM Support**: Batch processing scripts for HPC environments

## Main Scripts

### Core Processing
- `spatial_cimis_processing.py` - Main processing script with grid standardization
- `spatial_cimis_unified.py` - Unified analysis combining Spatial CIMIS and GridMET
- `spatial_cimis_analysis.py` - Analysis and comparison tools

### SLURM Job Scripts
- `run_processing_job.sh` - Process variables with grid standardization
- `run_unified_analysis.sh` - Run unified analysis with both datasets
- `run_rs_reprocessing.sh` - Regenerate Rs files with coordinate fixes

### Plotting and Visualization
- `plot_comparison.py` - Spatial and temporal comparisons
- `plot_grid_change.py` - Visualize grid changes over time
- `plot_station_timeseries.py` - Station time series plots

### Testing and Validation
- `test_grid_standardization.py` - Test grid standardization logic
- `quick_test_rs2010.py` - Quick test for Rs 2010 processing

## Configuration Files

- `scp_config.txt` - Spatial CIMIS processing configuration
- `analysis_config_rs.txt` - Analysis configuration for Rs variable
- `plot_config_rs.txt` - Plotting configuration

## Key Improvements

### Grid Standardization (2025-10-21)
- Handles varying ASCII file formats (500m vs 2km resolution)
- Uses bilinear interpolation for smooth transitions
- Preserves 74% more valid data compared to simple array slicing
- Future-proof for any grid changes

### Coordinate Fix (2025-10-22)
- Fixed Y-coordinate inversion bug affecting spatial operations
- Corrected coordinate generation to match rasterio convention
- Required regeneration of all NetCDF files

### GridMET Integration (2025-10-22)
- Added support for GridMET comparison data
- Handles variable name mapping (`surface_downwelling_shortwave_flux_in_air`)
- Enables comprehensive dataset comparison

## Usage

### Quick Start
```bash
# Test grid standardization
python quick_test_rs2010.py

# Process single year
python spatial_cimis_processing.py --config scp_config.txt --start-year 2010 --end-year 2010

# Run unified analysis
sbatch run_unified_analysis.sh analysis_config_rs.txt
```

### Configuration
Edit configuration files to set:
- Variable to process (Tx, ETo, Tn, Rs, etc.)
- Data paths and output locations
- Processing years and options
- Analysis parameters

## Requirements

- Python 3.7+
- xarray
- rasterio
- geopandas
- rioxarray
- netCDF4
- matplotlib
- pandas
- numpy

## Performance

| Task | Time | Memory | Output Size |
|------|------|--------|-------------|
| Grid test | <1 min | <1 GB | N/A |
| Single year | 3-5 min | ~2 GB | 773 MB |
| Unified analysis | 6-8 min | 19.5 GB | ~20 MB |
| Full dataset (19 years) | 1-2h | ~4 GB | ~15 GB |

## Documentation

- `GRID_STANDARDIZATION_README.md` - Technical details of grid standardization
- `SOLUTION_SUMMARY.md` - High-level problem and solution overview
- `WORKING_NOTES.md` - Daily development log and notes
- `VSCODE_USAGE.md` - Usage instructions for VS Code server

## License

This project is part of the Spatial CIMIS research initiative.

## Contact

For questions or issues, please refer to the working notes or contact the development team.
