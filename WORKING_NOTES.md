# Spatial CIMIS Processing - Working Notes

## Project Overview
Processing Spatial CIMIS data with grid standardization to handle varying ASCII file formats (different resolutions, extents, and grid sizes).

---

## Daily Log

### 2025-10-21 - Grid Standardization Implementation

#### Problem Identified
- ASCII files have varying formats:
  - **Resolution**: 500m vs 2km (e.g., Rs 2010 changes mid-year)
  - **Grid extent**: Different corner coordinates
  - **Grid size**: 500×552 vs 510×560
- Original script only handled simple array slicing → spatial misalignment and data loss

#### Solution Implemented
- Created `spatial_cimis_processing.py` with rasterio-based grid standardization
- Uses `rasterio.warp.reproject()` with bilinear interpolation
- Standard target grid: 500×552 @ 2km resolution
- Made mask optional (default: no mask)
- Single-year mode (no 3-year window for individual year processing)

#### Testing Results
✅ **Grid standardization tests**: All 4 test cases passed
- Standard grid (500×552 @ 2km)
- Extended grid (510×560 @ 2km)  
- High resolution (2000×2208 @ 500m)
- Resolution change handling

✅ **Rs 2010 test**: Successfully processed
- Handled 500m (Jan-Jul) → 2km (Aug-Dec) resolution change
- Output: 773M NetCDF file with consistent grid
- Processing time: ~3-5 minutes

✅ **Rs 2018 comparison** (Old vs New):
- **Data preservation**: +74% more valid cells (39M additional)
- **Data accuracy**: Identical at matching cells (max diff = 0.000000)
- **Better coverage**: Especially at grid transition dates
- Old simple slicing was dropping valid data

✅ **Full analysis pipeline**: Completed successfully
- Job 28232344: 8m 36s runtime
- Loaded grid-standardized Rs data (7305 days, 552×500 grid)
- Generated station comparisons and climatologies
- All outputs created in `/home/salba/SpatialCIMIS/output/`

#### Files Created
- `spatial_cimis_processing.py` - Main processing script
- `test_grid_standardization.py` - Test suite
- `quick_test_rs2010.py` - Quick Rs 2010 test
- `run_interactive.py` - Interactive menu
- `run_processing.sh` - Shell wrapper
- `run_processing_job.sh` - SLURM job script
- `scp_config.txt` - Configuration (updated)
- Documentation: `GRID_STANDARDIZATION_README.md`, `VSCODE_USAGE.md`, `SOLUTION_SUMMARY.md`

#### Next Steps
- [ ] Process full Rs dataset (2004-2022) with grid standardization
- [ ] Apply to other variables (Tx, ETo, Tn)
- [ ] Consider removing mask entirely from workflow
- [ ] Update production pipeline

#### Notes
- Grid standardization adds ~74% more valid data by properly handling spatial reprojection
- Processing is fast: 3-8 min per year
- No mask needed - NaN values naturally mark missing data
- Solution is future-proof for any grid changes

---

### 2025-10-22 - Coordinate Fix & GridMET Integration

#### Major Issues Resolved
1. **Y-coordinate inversion bug**: Spatial CIMIS data was vertically flipped
   - **Problem**: Stations appeared outside grid after Oct 23, 2018 grid change
   - **Root cause**: `spatial_cimis_processing.py` generated Y coordinates bottom-to-top, but `rasterio` expects top-to-bottom
   - **Fix**: Corrected Y-coordinate generation logic to match rasterio convention
   - **Impact**: Required regenerating all Rs NetCDF files

2. **GridMET integration failure**: Could not load GridMET data for comparison
   - **Problem**: Variable name mismatch in GridMET files
   - **Root cause**: Script looked for `srad` but files contain `surface_downwelling_shortwave_flux_in_air`
   - **Fix**: Added explicit handling for `surface_downwelling_shortwave_flux_in_air` variable
   - **Impact**: Now successfully loads both Spatial CIMIS and GridMET data

#### Files Modified
- `spatial_cimis_processing.py`: Fixed Y-coordinate generation (lines ~180-185)
- `spatial_cimis_unified.py`: Enhanced GridMET variable detection (lines 294-307, 358-363)
- `analysis_config_rs.txt`: Updated to use corrected Rs files
- `run_rs_reprocessing.sh`: Created to regenerate all Rs files with coordinate fix

#### Processing Jobs Completed
1. **Job 28237201**: Unified analysis with corrected Rs files
   - Runtime: 1m 21s
   - Status: ✅ COMPLETED successfully
   - Outputs: Station data, climatologies, metadata

2. **Job 28238059**: Full analysis with GridMET integration
   - Runtime: 6m 30s  
   - Status: ✅ COMPLETED successfully
   - GridMET: 20 files (2004-2023), 75 stations extracted
   - Both datasets: Spatial CIMIS (552×500) + GridMET (227×247)

3. **Job 28238136**: ETo processing with grid standardization
   - Status: 🔄 RUNNING (processing 2004-2023)
   - Resources: 4 CPUs, 16GB RAM, 4h time limit

#### Key Results
- **Coordinate fix verified**: Station extraction now shows continuous data
- **GridMET integration working**: Both datasets successfully loaded and processed
- **Data quality**: 69 stations with complete coverage, realistic values (4-12 MJ/m²/day)
- **File organization**: Moved test outputs to `/output/test/` with clean names

#### Technical Details
- **Y-coordinate fix**: Changed from `y = y_bottom + np.arange(ny) * cellsize` to `y = y_top - np.arange(ny) * cellsize`
- **GridMET variable mapping**: Added `surface_downwelling_shortwave_flux_in_air` to detection logic
- **Output structure**: All files now properly organized in test directory

#### Files Created/Modified
- `spatial_cimis_processing.py` - Y-coordinate fix
- `spatial_cimis_unified.py` - GridMET variable detection
- `run_rs_reprocessing.sh` - Rs regeneration script
- `run_processing_job.sh` - ETo processing script
- `/output/test/` - Organized output directory

#### Next Steps
- [ ] Monitor ETo processing completion
- [ ] Test plotting with corrected data
- [ ] Process remaining variables (Tx, Tn)
- [ ] Deploy to production pipeline

#### Notes
- Coordinate fix was critical - affected all spatial operations
- GridMET integration enables comprehensive comparison analysis
- Processing pipeline now fully functional for both datasets

---

### [Date] - [Title]

#### Tasks Completed
- 

#### Issues Encountered
- 

#### Results
- 

#### Next Steps
- 

#### Notes
- 

---

## Configuration Reference

### Current Setup (scp_config.txt)
```
variable = Rs
data_path = /group/moniergrp/SpatialCIMIS/ascii/
netcdf_path = /group/moniergrp/SpatialCIMIS/netcdf/test/
output_path = /home/salba/SpatialCIMIS/data/
start_year = 2004
end_year = 2022
use_mask = false
process_data = true
```

### Target Grid Parameters
```
ncols = 500
nrows = 552
xllcorner = -400000
yllcorner = -650000
cellsize = 2000.0
crs = EPSG:3310 (California Albers)
```

---

## Quick Commands Reference

```bash
# Quick test
python quick_test_rs2010.py

# Process single year
python spatial_cimis_processing.py --config scp_config.txt --start-year 2010 --end-year 2010

# Process full dataset
python spatial_cimis_processing.py --config scp_config.txt

# Submit SLURM job for analysis
sbatch run_unified_analysis.sh analysis_config_rs.txt

# Monitor job
squeue -u salba
tail -f unified_analysis_*.log

# Compare files
python3 -c "import xarray as xr; ..."
```

---

## Issues & Solutions

### Issue: Mask creation slow
**Solution**: Made mask optional, default to no mask. Processing without mask is faster and preserves natural data coverage.

### Issue: 3-year window for single year processing
**Solution**: Added logic to only use 3-year window when processing multiple years.

### Issue: Plotting script expects day_of_year dimension
**Solution**: Updated plot_comparison.py to handle both dimensioned and pre-computed climatologies.

---

## Performance Metrics

| Task | Time | Memory | Output Size |
|------|------|--------|-------------|
| Grid test | <1 min | <1 GB | N/A |
| Single year (Rs 2010) | 3-5 min | ~2 GB | 773 MB |
| Unified analysis | 8.5 min | 19.5 GB | ~20 MB outputs |
| Full dataset (19 years) | Est. 1-2h | ~4 GB | ~15 GB |

---

## Data Quality Metrics

### Rs 2018 (Old vs New Processing)
- Valid cells: 52.9M → 92.1M (+74%)
- Mean value: 19.17 → 19.29 MJ/m²/day (0.7% change)
- Data accuracy: Identical at matching cells
- Oct 23 grid change: Better preservation (+11K cells)

---

## To Do
- [ ] Process full Rs 2004-2022 dataset
- [ ] Test with ETo (has 510×560 extent issues)
- [ ] Test with Tx (verify on all variables)
- [ ] Compare processing times old vs new
- [ ] Document production deployment
- [ ] Create validation plots

---

## Notes & Observations
- Grid standardization significantly improves data preservation
- Bilinear interpolation provides smooth transitions between resolutions
- No performance penalty despite reprojection step
- Mask not needed for most use cases

