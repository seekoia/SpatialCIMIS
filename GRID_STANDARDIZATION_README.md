# Spatial CIMIS Grid Standardization

## Problem Identified

The Spatial CIMIS ASCII files have **varying grid dimensions, resolutions, and spatial extents** across different time periods:

### Examples Found:

1. **Different Resolutions:**
   - `Rs.2010-01-15.asc`: 2000×2208 grid at **500m** resolution
   - `Rs.2010-08-15.asc`: 500×552 grid at **2000m** resolution

2. **Different Extents:**
   - Most files: `xllcorner=-400000, yllcorner=-650000`
   - Some files: `xllcorner=-410000, yllcorner=-660000`

3. **Different Grid Sizes (at same resolution):**
   - Standard: 500×552
   - Variant: 510×560

## Previous Approach

The original code (`spatial_cimis_processing_reference.py`) attempted to handle this with simple array slicing:

```python
if tmp.shape[0] == ny:
    data[t, :, :] = tmp
else:
    # Trimming for 560×510 grids
    data[t, :, :] = tmp[3:555, 5:505]
```

**Limitations:**
- Only handles one specific case (560×510 → 552×500)
- Doesn't account for different spatial extents
- Fails completely for different resolutions (e.g., 500m vs 2000m)
- No proper spatial alignment

## New Solution

The new script (`spatial_cimis_processing.py`) implements **proper grid standardization** using rasterio's reprojection capabilities:

### Key Features:

1. **Defines a Standard Target Grid:**
   ```python
   TARGET_GRID = {
       'ncols': 500,
       'nrows': 552,
       'xllcorner': -400000,
       'yllcorner': -650000,
       'cellsize': 2000.0,
       'crs': 'EPSG:3310'
   }
   ```

2. **Automatic Detection:**
   - Reads grid parameters from each ASCII file header
   - Detects when regridding is needed

3. **Proper Spatial Regridding:**
   - Uses `rasterio.warp.reproject()` with bilinear interpolation
   - Maintains spatial accuracy across different resolutions
   - Handles coordinate transformations properly

### Core Function:

```python
def read_and_reproject_asc(filename, target_transform, target_shape):
    """
    Read ASCII file and reproject to target grid if needed.
    - Handles different resolutions (500m, 2000m, etc.)
    - Handles different extents
    - Uses bilinear resampling for smooth transitions
    """
```

## Usage

### Using Configuration File:

```bash
python spatial_cimis_processing.py --config scp_config.txt --process-all
```

### Command Line Options:

```bash
# Create mask only
python spatial_cimis_processing.py --variable Tx --process-mask

# Process data for specific years
python spatial_cimis_processing.py --variable Rs \
    --start-year 2010 --end-year 2012 \
    --process-data

# Full processing pipeline
python spatial_cimis_processing.py --variable ETo --process-all
```

## Benefits

1. **Consistency:** All output NetCDF files have identical grid dimensions
2. **Accuracy:** Proper spatial interpolation preserves data quality
3. **Flexibility:** Handles any grid variation automatically
4. **Future-proof:** Will work with any new grid changes

## Testing Recommendations

Before running on full dataset, test with a problem period:

```bash
# Test with Rs 2010 (has both 500m and 2000m resolution)
python spatial_cimis_processing.py \
    --variable Rs \
    --start-year 2010 \
    --end-year 2010 \
    --process-mask \
    --process-data
```

## Comparison with Reference Script

| Feature | Reference Script | New Script |
|---------|-----------------|------------|
| Handle 560×510 grids | ✓ (simple slice) | ✓ (proper reproject) |
| Handle different extents | ✗ | ✓ |
| Handle 500m resolution | ✗ | ✓ |
| Spatial accuracy | Limited | Full |
| Future-proof | ✗ | ✓ |

## Configuration File

The script uses the same configuration file format as the reference script:

```
variable = Rs
data_path = /group/moniergrp/SpatialCIMIS/ascii/
netcdf_path = /group/moniergrp/SpatialCIMIS/netcdf/
output_path = /home/salba/SpatialCIMIS/data/
start_year = 2004
end_year = 2023
mask_years = 2004-2024
process_mask = true
process_data = true
```

## Notes

- The target grid (500×552 at 2km) was chosen as it represents the most common format
- Bilinear interpolation is used for smooth resampling between resolutions
- The script preserves all original metadata and adds processing history
- NoData values (-9999) are properly handled throughout the reprojection process

