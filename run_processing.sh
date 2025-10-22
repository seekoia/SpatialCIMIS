#!/bin/bash
#
# Run Spatial CIMIS processing with grid standardization
# Usage: ./run_processing.sh [test|process]
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create log directory if it doesn't exist
mkdir -p logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

case "${1:-help}" in
    test)
        echo "Running grid standardization tests..."
        echo "Output will be saved to logs/test_${TIMESTAMP}.log"
        python test_grid_standardization.py 2>&1 | tee "logs/test_${TIMESTAMP}.log"
        ;;
    
    process)
        VARIABLE="${2:-Tx}"
        CONFIG="${3:-scp_config.txt}"
        
        echo "=========================================="
        echo "Starting Spatial CIMIS Processing"
        echo "=========================================="
        echo "Variable: $VARIABLE"
        echo "Config: $CONFIG"
        echo "Timestamp: $TIMESTAMP"
        echo "Log file: logs/process_${VARIABLE}_${TIMESTAMP}.log"
        echo "=========================================="
        echo ""
        
        python spatial_cimis_processing.py \
            --config "$CONFIG" \
            --variable "$VARIABLE" \
            --process-all \
            2>&1 | tee "logs/process_${VARIABLE}_${TIMESTAMP}.log"
        ;;
    
    process-year)
        VARIABLE="${2:-Tx}"
        YEAR="${3:-2024}"
        CONFIG="${4:-scp_config.txt}"
        
        echo "=========================================="
        echo "Processing Single Year"
        echo "=========================================="
        echo "Variable: $VARIABLE"
        echo "Year: $YEAR"
        echo "Config: $CONFIG"
        echo "Log file: logs/process_${VARIABLE}_${YEAR}_${TIMESTAMP}.log"
        echo "=========================================="
        echo ""
        
        python spatial_cimis_processing.py \
            --config "$CONFIG" \
            --variable "$VARIABLE" \
            --start-year "$YEAR" \
            --end-year "$YEAR" \
            --process-mask \
            --process-data \
            2>&1 | tee "logs/process_${VARIABLE}_${YEAR}_${TIMESTAMP}.log"
        ;;
    
    mask-only)
        VARIABLE="${2:-Tx}"
        CONFIG="${3:-scp_config.txt}"
        
        echo "Creating mask for $VARIABLE..."
        echo "Log file: logs/mask_${VARIABLE}_${TIMESTAMP}.log"
        
        python spatial_cimis_processing.py \
            --config "$CONFIG" \
            --variable "$VARIABLE" \
            --process-mask \
            2>&1 | tee "logs/mask_${VARIABLE}_${TIMESTAMP}.log"
        ;;
    
    help|*)
        echo "Spatial CIMIS Processing Runner"
        echo ""
        echo "Usage: ./run_processing.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  test                     - Run grid standardization tests"
        echo "  process <var> [config]   - Run full processing for variable"
        echo "  process-year <var> <yr>  - Process single year for testing"
        echo "  mask-only <var> [config] - Create mask only"
        echo "  help                     - Show this help"
        echo ""
        echo "Examples:"
        echo "  ./run_processing.sh test"
        echo "  ./run_processing.sh process Tx"
        echo "  ./run_processing.sh process Rs scp_config.txt"
        echo "  ./run_processing.sh process-year Rs 2010"
        echo "  ./run_processing.sh mask-only ETo"
        echo ""
        echo "Logs are saved to: $SCRIPT_DIR/logs/"
        ;;
esac

