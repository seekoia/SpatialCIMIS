#!/bin/bash
#SBATCH --partition bmh 
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8 
#SBATCH --mem=32G
#SBATCH --time 4:00:00
#SBATCH --job-name extract-station-pixels
#SBATCH --output extract_pixels_%j.log
#SBATCH --error extract_pixels_%j.err
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
echo ""

# Get config file from command line or use default
if [ $# -eq 0 ]; then
    CONFIG_FILE="analysis_config.txt"
else
    CONFIG_FILE=$1
fi

echo "Using configuration file: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Read variable from config
VARIABLE=$(grep "^variable" $CONFIG_FILE | cut -d'=' -f2 | tr -d ' ')
echo "Variable: $VARIABLE"
echo ""

# Run the extraction script with unbuffered output
echo "Starting pixel extraction for CIMIS stations..."
python -u extract_station_pixels.py "$CONFIG_FILE"

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Extraction completed successfully at: $(date)"
else
    echo "✗ Extraction failed with exit code: $?"
    exit 1
fi

echo ""
echo "Job finished at: $(date)"


