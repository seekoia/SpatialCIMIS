#!/bin/bash
# Examples of how to run the Spatial CIMIS data processing for different variables

echo "Spatial CIMIS Data Processing Examples"
echo "======================================"

echo ""
echo "1. Process Maximum Temperature (Tx):"
echo "   sbatch spatial_cimis_job.sh Tx"
echo "   or"
echo "   python spatial_cimis_data_processing.py Tx"

echo ""
echo "2. Process Reference Evapotranspiration (ETo):"
echo "   sbatch spatial_cimis_job.sh ETo"
echo "   or"
echo "   python spatial_cimis_data_processing.py ETo"

echo ""
echo "3. Process Minimum Temperature (Tn):"
echo "   sbatch spatial_cimis_job.sh Tn"
echo "   or"
echo "   python spatial_cimis_data_processing.py Tn"

echo ""
echo "4. Process Dew Point Temperature (Tdew):"
echo "   sbatch spatial_cimis_job.sh Tdew"
echo "   or"
echo "   python spatial_cimis_data_processing.py Tdew"

echo ""
echo "5. Process Wind Speed at 2m (U2):"
echo "   sbatch spatial_cimis_job.sh U2"
echo "   or"
echo "   python spatial_cimis_data_processing.py U2"

echo ""
echo "Supported variables: Tx, ETo, Tn, Tdew, U2"
echo "Note: Variable argument is required for all commands"
echo "Note: Variable names are case-insensitive"
