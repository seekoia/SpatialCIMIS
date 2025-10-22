#!/bin/bash
#SBATCH --partition bmh 
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8 
#SBATCH --mem=32G
#SBATCH --time 24:00:00
#SBATCH --job-name spatial-cimis-ref
#SBATCH --output spatial_cimis_ref_%j.log
#SBATCH --error spatial_cimis_ref_%j.err
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

# Get config file from command line or use default
if [ $# -eq 0 ]; then
    CONFIG_FILE="scp_config.txt"
else
    CONFIG_FILE=$1
fi

echo "Using configuration file: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Check if required directories exist
echo "Checking directories..."

# Read data_path from config
DATA_PATH=$(grep "^data_path" $CONFIG_FILE | cut -d'=' -f2 | tr -d ' ')
NETCDF_PATH=$(grep "^netcdf_path" $CONFIG_FILE | cut -d'=' -f2 | tr -d ' ')
OUTPUT_PATH=$(grep "^output_path" $CONFIG_FILE | cut -d'=' -f2 | tr -d ' ')

if [ ! -d "$DATA_PATH" ]; then
    echo "Warning: ASCII data directory not found: $DATA_PATH"
fi

if [ ! -d "$NETCDF_PATH" ]; then
    echo "Creating netcdf output directory: $NETCDF_PATH"
    mkdir -p "$NETCDF_PATH"
fi

if [ ! -d "$OUTPUT_PATH" ]; then
    echo "Creating output directory: $OUTPUT_PATH"
    mkdir -p "$OUTPUT_PATH"
fi

echo ""
echo "Starting Spatial CIMIS processing with reference script..."
echo "Configuration file: $CONFIG_FILE"
echo "Data path: $DATA_PATH"
echo "NetCDF path: $NETCDF_PATH"
echo "Output path: $OUTPUT_PATH"
echo ""

# Run the reference processing script with config file (unbuffered output)
python -u spatial_cimis_processing_reference.py --config "$CONFIG_FILE"

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Spatial CIMIS processing completed successfully at: $(date)"
else
    echo ""
    echo "Spatial CIMIS processing failed with exit code: $?"
    exit 1
fi

echo "Job finished at: $(date)"



