# cimis_analysis_package/config.py

import os

# --- Base Project Path (assuming this file is in cimis_analysis_package) ---
# This helps in creating absolute paths if needed, especially for data/output
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Goes up one level from cimis_analysis_package

# --- Input Data Paths ---
STATION_API_URL = 'http://et.water.ca.gov/api/station'
# Default path for CIMIS_Stations.csv, can be overridden by main script if downloaded elsewhere
DEFAULT_STATION_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "CIMIS_Stations.csv")
CIMIS_DATA_PATH = "/group/moniergrp/SpatialCIMIS/CIMIS/"  # MODIFY if your path is different
NETCDF_SPATIAL_CIMIS_PATH = '/group/moniergrp/SpatialCIMIS/netcdf/'  # MODIFY if your path is different
NETCDF_GRIDMET_PATH = "/group/moniergrp/gridMET"  # MODIFY if your path is different
DEFAULT_SHAPEFILE_PATH = os.path.join(PROJECT_ROOT, "data", "CA_State.shp") # Ensure this shapefile is available

# --- Output Base Directories ---
# The main script will create subdirectories for each variable within these.
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "output")
OUTPUT_NETCDF_PATH_BASE = os.path.join(OUTPUT_BASE_DIR, "output_netcdf")
OUTPUT_CSV_PATH_BASE = os.path.join(OUTPUT_BASE_DIR, "output_csv")
OUTPUT_PLOT_PATH_BASE = os.path.join(OUTPUT_BASE_DIR, "output_plots")


# --- Date Filters ---
START_DATE_FILTER = '2004-01-01'
END_DATE_FILTER = '2024-01-01'
CONNECT_DATE_FILTER = '2004-01-01'
DISCONNECT_DATE_FILTER = '2024-01-01'

# --- Variable Configuration ---
TARGET_VARIABLES_CONFIG = {
    "Rad": {
        "cimis_csv_column_name": "Sol Rad (W/sq.m)",
        "cimis_csv_output_name": "Rad",
        "spatial_cimis_netcdf_var": "Rs",
        "gridmet_netcdf_var": "srad", # This is the prefix for gridMET files, actual var name inside might differ
        "gridmet_internal_var_name_heuristic": "radiation", # Heuristic to find actual var name in gridMET file
        "unit": "W/m²",
        "spatial_cimis_scale_factor": 11.57,
        "gridmet_scale_factor": 0.1
    },
    "ETo": {
        "cimis_csv_column_name": "ETo (mm)",
        "cimis_csv_output_name": "ETo",
        "spatial_cimis_netcdf_var": "ETo",
        "gridmet_netcdf_var": "eto", # Prefix for gridMET files
        "gridmet_internal_var_name_heuristic": "evapotranspiration", # Heuristic
        "unit": "mm/day",
        "spatial_cimis_scale_factor": 1.0,
        "gridmet_scale_factor": 1.0
    },
    # Example for Temperature Minimum
    # "Tmin": {
    #     "cimis_csv_column_name": "Min Temp (°C)", # Verify actual column name in CIMIS CSVs
    #     "cimis_csv_output_name": "Tmin",
    #     "spatial_cimis_netcdf_var": "Tmin", # Or appropriate var if from a different spatial product
    #     "gridmet_netcdf_var": "tmmn",       # Prefix for gridMET files
    #     "gridmet_internal_var_name_heuristic": "minimum_temperature", # Heuristic
    #     "unit": "°C",
    #     "spatial_cimis_scale_factor": 1.0,
    #     "gridmet_scale_factor": 1.0 # GridMET temperatures are usually in Kelvin, may need conversion
    # },
}

# --- Variables to Process in the main script ---
VARIABLES_TO_PROCESS = ["Rad"]  # e.g., ["Rad", "ETo"]

# --- Plotting Constants ---
DEFAULT_STATION_CANDIDATES_FOR_TIMESERIES = ['6', '160']
DEFAULT_PLOT_COLORS = {'station': '#ababab', 'spatial': 'blue', 'gridmet': 'red'}

# --- Helper to create directories ---
def ensure_dir_exists(path):
    """Ensures a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)

# Ensure base output directories exist when this config is loaded
ensure_dir_exists(OUTPUT_NETCDF_PATH_BASE)
ensure_dir_exists(OUTPUT_CSV_PATH_BASE)
ensure_dir_exists(OUTPUT_PLOT_PATH_BASE)

print("Configuration loaded.")
