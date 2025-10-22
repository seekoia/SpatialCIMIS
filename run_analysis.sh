#!/bin/bash
#SBATCH --partition bmh 
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8 
#SBATCH --mem=32G
#SBATCH --time 12:00:00
#SBATCH --job-name spatial-cimis-analysis
#SBATCH --output spatial_analysis_%j.log
#SBATCH --error spatial_analysis_%j.err
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

# Run the analysis script with unbuffered output
echo "Starting Spatial CIMIS analysis..."
python -u spatial_cimis_analysis.py "$CONFIG_FILE"

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Analysis completed successfully at: $(date)"
    
    # Optionally run plotting script if analysis succeeded
    echo ""
    echo "Starting plotting..."
    python -u spatial_cimis_plotting.py "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✓ Plotting completed successfully"
    else
        echo "✗ Plotting failed with exit code: $?"
    fi
else
    echo "✗ Analysis failed with exit code: $?"
    exit 1
fi

echo ""
echo "Job finished at: $(date)"

