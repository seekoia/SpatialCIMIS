# Running Spatial CIMIS Processing in VS Code Server

## Quick Start

### Option 1: Interactive Mode (Recommended for beginners)

Open a terminal in VS Code and run:

```bash
cd /home/salba/SpatialCIMIS
python run_interactive.py
```

This will give you a menu-driven interface where you can:
1. Run tests to verify everything works
2. Process specific years or variables
3. Choose which processing steps to run

### Option 2: Command Line (Quick tests)

```bash
# Test grid standardization
./run_processing.sh test

# Test with a problem year (Rs 2010 has resolution changes)
./run_processing.sh process-year Rs 2010

# Create mask only
./run_processing.sh mask-only Tx

# Process full pipeline for a variable
./run_processing.sh process ETo
```

### Option 3: Direct Python Script

```bash
# Using config file
python spatial_cimis_processing.py --config scp_config.txt --process-all

# Custom options
python spatial_cimis_processing.py \
    --variable Rs \
    --start-year 2010 \
    --end-year 2012 \
    --process-mask \
    --process-data
```

## Recommended Testing Workflow

### Step 1: Verify Grid Standardization Works

```bash
python test_grid_standardization.py
```

This tests files with different:
- Resolutions (500m vs 2000m)
- Extents (different corner coordinates)
- Grid sizes (500×552 vs 510×560)

Expected output:
```
✓ All tests passed! Grid standardization is working correctly.
```

### Step 2: Test with Problematic Period (Rs 2010)

Rs in 2010 changes from 500m resolution to 2km resolution mid-year:

```bash
python run_interactive.py test-year Rs 2010
```

Or:

```bash
./run_processing.sh process-year Rs 2010
```

### Step 3: Process Full Variable

Once tests pass, process a full variable:

```bash
python run_interactive.py process Tx
```

## Running in Background

For long-running processes, use `screen` or `tmux`:

### Using screen:

```bash
# Start a screen session
screen -S cimis_processing

# Run your processing
python run_interactive.py

# Detach: Press Ctrl+A then D

# Reattach later
screen -r cimis_processing

# List sessions
screen -ls
```

### Using tmux:

```bash
# Start tmux session
tmux new -s cimis_processing

# Run your processing
python run_interactive.py

# Detach: Press Ctrl+B then D

# Reattach later
tmux attach -t cimis_processing

# List sessions
tmux ls
```

## Monitoring Progress

### View logs in real-time:

```bash
# Logs are saved in logs/ directory
tail -f logs/process_*.log
```

### Check most recent log:

```bash
ls -lt logs/ | head -5
tail -100 logs/process_Rs_*.log
```

## Configuration

Edit `scp_config.txt` to set default parameters:

```
variable = Tx
data_path = /group/moniergrp/SpatialCIMIS/ascii/
netcdf_path = /group/moniergrp/SpatialCIMIS/netcdf/
output_path = /home/salba/SpatialCIMIS/data/
start_year = 2004
end_year = 2023
mask_years = 2004-2024
process_mask = false
process_data = false
```

## VS Code Terminal Setup

### Open Terminal in VS Code:
1. Press `` Ctrl+` `` or go to `Terminal > New Terminal`
2. Navigate to project: `cd /home/salba/SpatialCIMIS`

### Split Terminal for Monitoring:
1. Click the split terminal icon or press `Ctrl+Shift+5`
2. In one terminal: Run processing
3. In another: Monitor logs with `tail -f logs/*.log`

### Using VS Code's Task Runner:

Create `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Test Grid Standardization",
            "type": "shell",
            "command": "python",
            "args": ["test_grid_standardization.py"],
            "problemMatcher": [],
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        },
        {
            "label": "Run Interactive",
            "type": "shell",
            "command": "python",
            "args": ["run_interactive.py"],
            "problemMatcher": [],
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        }
    ]
}
```

Then run with `Ctrl+Shift+P` → "Tasks: Run Task"

## Troubleshooting

### Import Errors

If you get module import errors:

```bash
# Check Python environment
which python
python --version

# Install requirements
pip install numpy netCDF4 xarray rasterio pyproj
```

### Permission Errors

```bash
# Make scripts executable
chmod +x *.py *.sh
```

### Memory Issues

For large datasets, you may need to process years individually:

```bash
for year in {2004..2023}; do
    python spatial_cimis_processing.py \
        --variable Rs \
        --start-year $year \
        --end-year $year \
        --process-data
done
```

## Output Structure

```
/home/salba/SpatialCIMIS/
├── spatial_cimis_processing.py    # Main script
├── run_interactive.py             # Interactive runner
├── run_processing.sh              # Shell wrapper
├── test_grid_standardization.py   # Test script
├── scp_config.txt                 # Configuration
└── logs/                          # Processing logs
    ├── test_YYYYMMDD_HHMMSS.log
    ├── process_Rs_YYYYMMDD_HHMMSS.log
    └── ...

/group/moniergrp/SpatialCIMIS/
├── ascii/                         # Input ASCII files
└── netcdf/                        # Output NetCDF files
    ├── spatial_cimis_rs_mask.nc
    ├── spatial_cimis_rs_2010.nc
    ├── spatial_cimis_rs_2011.nc
    └── ...
```

## Tips for VS Code Server

1. **Use the integrated terminal** - More stable than external SSH
2. **Save output** - All operations log to `logs/` directory
3. **Monitor system resources**: Open new terminal and run `htop`
4. **Use Git** - Commit configuration changes
5. **Remote development** - Install Python extension for better debugging

## Common Commands Reference

```bash
# Quick test
python test_grid_standardization.py

# Interactive mode
python run_interactive.py

# Test single year
python run_interactive.py test-year Rs 2010

# Direct processing
python spatial_cimis_processing.py --variable Tx --process-all

# Background processing
screen -dmS cimis bash -c "python spatial_cimis_processing.py --variable Rs --process-all"

# Check running processes
ps aux | grep spatial_cimis_processing

# View logs
ls -lth logs/ | head
tail -f logs/process_*.log
```

## Next Steps

1. ✓ Run `python test_grid_standardization.py`
2. ✓ Test with `Rs 2010` (has resolution changes)
3. ✓ Review logs in `logs/` directory
4. ✓ If tests pass, process full variable
5. ✓ Monitor progress with `tail -f logs/*.log`

