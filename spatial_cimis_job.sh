#!/bin/bash
#SBATCH --partition bmh 
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8 
#SBATCH --mem=32G
#SBATCH --time 24:00:00
#SBATCH --job-name spatial-cimis-processing
#SBATCH --output spatial_cimis_%j.log
#SBATCH --error spatial_cimis_%j.err
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
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"

# Check if required directories exist
if [ ! -d "/group/moniergrp/SpatialCIMIS/ascii/" ]; then
    echo "Warning: ASCII data directory not found"
fi

if [ ! -d "/group/moniergrp/SpatialCIMIS/netcdf/" ]; then
    echo "Creating netcdf output directory..."
    mkdir -p /group/moniergrp/SpatialCIMIS/netcdf/
fi

if [ ! -d "/home/salba/SpatialCIMIS/data/" ]; then
    echo "Creating local data output directory..."
    mkdir -p /home/salba/SpatialCIMIS/data/
fi

# Get variable from command line argument (required)
if [ $# -eq 0 ]; then
    echo "Error: Variable argument is required."
    echo "Usage: sbatch spatial_cimis_job.sh <variable>"
    echo "Supported variables: Tx, ETo, Tn, Tdew, U2"
    exit 1
fi

VARIABLE=$1
echo "Starting Spatial CIMIS data processing for variable: $VARIABLE"
python spatial_cimis_data_processing.py $VARIABLE

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Spatial CIMIS processing completed successfully at: $(date)"
else
    echo "Spatial CIMIS processing failed with exit code: $?"
fi

echo "Job finished at: $(date)"
