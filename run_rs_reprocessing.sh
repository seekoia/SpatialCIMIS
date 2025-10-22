#!/bin/bash
#SBATCH --partition bmh
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=16G
#SBATCH --time 4:00:00
#SBATCH --job-name rs_reprocess_corrected
#SBATCH --output rs_reprocess_%j.log
#SBATCH --error rs_reprocess_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=salba@ucdavis.edu

# Load required modules
module load gdal
module load netcdf-c

# Use local Python environment
source /home/salba/miniforge3/bin/activate base

# Set working directory
cd /home/salba/SpatialCIMIS

# Print job information
echo "=========================================="
echo "Rs Reprocessing with Corrected Y Coordinates"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo ""

# Backup existing files
echo "1. Backing up existing Rs files..."
mkdir -p /group/moniergrp/SpatialCIMIS/netcdf/test_backup_$(date +%Y%m%d)
cp /group/moniergrp/SpatialCIMIS/netcdf/test/spatial_cimis_rs_2*.nc \
   /group/moniergrp/SpatialCIMIS/netcdf/test_backup_$(date +%Y%m%d)/ 2>/dev/null || echo "  No existing files to backup"
echo "  Backed up to: test_backup_$(date +%Y%m%d)/"
echo ""

# Remove old files with inverted Y coordinates
echo "2. Removing files with inverted Y coordinates..."
rm -f /group/moniergrp/SpatialCIMIS/netcdf/test/spatial_cimis_rs_2*.nc
echo "  Removed old Rs files"
echo ""

# Regenerate with corrected Y coordinates
echo "3. Regenerating Rs files with corrected Y coordinates..."
echo "  Config: scp_config.txt"
echo "  Years: 2004-2022"
echo "  Grid standardization: ENABLED"
echo "  Y coordinate fix: APPLIED"
echo ""

python -u spatial_cimis_processing.py --config scp_config.txt

# Check if processing completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Reprocessing completed successfully!"
    echo "=========================================="
    
    # List created files
    echo ""
    echo "Created files:"
    ls -lh /group/moniergrp/SpatialCIMIS/netcdf/test/spatial_cimis_rs_2*.nc 2>/dev/null | wc -l
    echo " NetCDF files created"
    
    # Verify a sample file has correct Y coordinates
    echo ""
    echo "Verifying Y coordinate fix..."
    python3 << 'VERIFY'
import xarray as xr
ds = xr.open_dataset('/group/moniergrp/SpatialCIMIS/netcdf/test/spatial_cimis_rs_2010.nc')
y = ds.y.values
print(f"  Y[0:5]:  {y[:5]}")
print(f"  Y[-5:]:  {y[-5:]}")
print(f"  Y direction: {'✓ High to low (CORRECT)' if y[0] > y[-1] else '✗ Low to high (WRONG)'}")
ds.close()
VERIFY
    
else
    echo ""
    echo "=========================================="
    echo "✗ Reprocessing failed!"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Job finished at: $(date)"
echo ""
echo "Next steps:"
echo "  1. Rerun unified analysis with corrected files"
echo "  2. Verify station extraction is correct"
echo "  3. Check time series plots show continuous data"

