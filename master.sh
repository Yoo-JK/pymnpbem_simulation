#!/bin/bash

# ============================================================================
# pyMNPBEM Automation Pipeline
# ============================================================================
# Main execution script for Python-based MNPBEM simulations and postprocessing
#
# Usage:
#   ./master.sh --str-conf <path> --sim-conf <path> [options]
#
# Options:
#   --reanalyze    Skip simulation, only run postprocessing
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

# ============================================================================
# Parse Arguments
# ============================================================================

STRUCTURE_FILE=""
SIMULATION_FILE=""
MNPBEM_PATH=""
VERBOSE=false
REANALYZE_MODE=false

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required:
    --str-conf PATH       Path to structure configuration file
    --sim-conf PATH      Path to simulation configuration file

Options:
    --mnpbem-path PATH     Path to mnpbem source directory (e.g. ~/workspace/MNPBEM)
    --reanalyze            Skip simulation, only reanalyze existing results
    --verbose              Enable verbose output
    -h, --help             Show this help message

Examples:
  # Run full simulation + postprocessing
  ./master.sh --str-conf ./config/structure/config_structure.py \\
              --sim-conf ./config/simulation/config_simulation.py \\
              --mnpbem-path ~/workspace/MNPBEM \\
              --verbose

  # Reanalyze existing results (skip simulation)
  ./master.sh --str-conf ./config/structure/config_structure.py \\
              --sim-conf ./config/simulation/config_simulation.py \\
              --reanalyze \\
              --verbose

Note:
  - In reanalyze mode, result files are automatically found from output_dir in config
  - Structure config is needed to match the original simulation

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --str-conf)
            STRUCTURE_FILE="$2"
            shift 2
            ;;
        --sim-conf)
            SIMULATION_FILE="$2"
            shift 2
            ;;
        --mnpbem-path)
            MNPBEM_PATH="$2"
            shift 2
            ;;
        --reanalyze)
            REANALYZE_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            print_msg "Unknown option: $1" "$RED"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Validate Arguments
# ============================================================================

if [ -z "$STRUCTURE_FILE" ] || [ -z "$SIMULATION_FILE" ]; then
    print_msg "Error: Both --str-conf and --sim-conf are required" "$RED"
    print_usage
    exit 1
fi

if [ ! -f "$STRUCTURE_FILE" ]; then
    print_msg "Error: Structure config file not found: $STRUCTURE_FILE" "$RED"
    exit 1
fi

if [ ! -f "$SIMULATION_FILE" ]; then
    print_msg "Error: Simulation config file not found: $SIMULATION_FILE" "$RED"
    exit 1
fi

# Extract output_dir from simulation config
OUTPUT_DIR=$(python3 -c "
import sys
sys.path.insert(0, '.')
try:
    exec(open('$SIMULATION_FILE').read())
    if 'args' in dir() and 'output_dir' in args:
        print(args['output_dir'])
    else:
        print('./results')
except Exception as e:
    print('./results')
" 2>/dev/null)

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./results"
fi

# ============================================================================
# Main Pipeline
# ============================================================================

if [ "$REANALYZE_MODE" = true ]; then
    # ========================================================================
    # REANALYZE MODE: Postprocess existing results only
    # ========================================================================

    print_msg "=== pyMNPBEM Reanalysis Mode ===" "$BLUE"
    echo ""
    print_msg "  Structure config:  $STRUCTURE_FILE" "$BLUE"
    print_msg "  Simulation config: $SIMULATION_FILE" "$BLUE"
    if [ -n "$MNPBEM_PATH" ]; then
        print_msg "  mnpbem path:       $MNPBEM_PATH" "$BLUE"
    fi
    print_msg "  Output directory:  $OUTPUT_DIR" "$BLUE"
    echo ""

    # Check if result files exist
    RESULT_FILE="$OUTPUT_DIR/simulation_results.npz"
    if [ ! -f "$RESULT_FILE" ]; then
        # Fallback: check for .mat format
        RESULT_FILE="$OUTPUT_DIR/simulation_results.mat"
        if [ ! -f "$RESULT_FILE" ]; then
            print_msg "Error: No result files found in $OUTPUT_DIR" "$RED"
            print_msg "  Run without --reanalyze to generate simulation results first" "$RED"
            exit 1
        fi
    fi

    print_msg "  Found existing results: $RESULT_FILE" "$GREEN"
    echo ""

    # Run postprocessing only
    print_msg "  Reanalyzing results..." "$YELLOW"

    if [ "$VERBOSE" = true ]; then
        python run_postprocess.py --str-conf "$STRUCTURE_FILE" --sim-conf "$SIMULATION_FILE" --verbose
    else
        python run_postprocess.py --str-conf "$STRUCTURE_FILE" --sim-conf "$SIMULATION_FILE"
    fi

    POSTPROCESS_EXIT_CODE=$?

    if [ $POSTPROCESS_EXIT_CODE -ne 0 ]; then
        print_msg "Error: Reanalysis failed (exit code: $POSTPROCESS_EXIT_CODE)" "$RED"
        exit 1
    fi

    print_msg "  Reanalysis completed successfully" "$GREEN"
    echo ""

    print_msg "=== Reanalysis Completed ===" "$GREEN"
    echo ""
    print_msg "  Updated results in: $OUTPUT_DIR/" "$BLUE"
    print_msg "    - Field analysis: field_analysis.json (updated)" "$BLUE"
    print_msg "    - Processed data: simulation_processed.* (updated)" "$BLUE"
    print_msg "    - Plots: *.png/pdf (regenerated)" "$BLUE"
    echo ""

else
    # ========================================================================
    # NORMAL MODE: Full simulation + postprocessing
    # ========================================================================

    print_msg "=== pyMNPBEM Automation Pipeline ===" "$BLUE"
    echo ""
    print_msg "  Structure config:  $STRUCTURE_FILE" "$BLUE"
    print_msg "  Simulation config: $SIMULATION_FILE" "$BLUE"
    if [ -n "$MNPBEM_PATH" ]; then
        print_msg "  mnpbem path:       $MNPBEM_PATH" "$BLUE"
    fi
    print_msg "  Output directory:  $OUTPUT_DIR" "$BLUE"
    echo ""

    # Build mnpbem-path argument
    MNPBEM_ARG=""
    if [ -n "$MNPBEM_PATH" ]; then
        MNPBEM_ARG="--mnpbem-path $MNPBEM_PATH"
    fi

    # Step 1: Run simulation (Python directly, no MATLAB)
    print_msg "  Step 1/2: Running BEM simulation..." "$YELLOW"

    TEMP_OUTPUT=$(mktemp)
    if [ "$VERBOSE" = true ]; then
        python -u run_simulation.py --str-conf "$STRUCTURE_FILE" --sim-conf "$SIMULATION_FILE" $MNPBEM_ARG --verbose 2>&1 | tee "$TEMP_OUTPUT"
    else
        python -u run_simulation.py --str-conf "$STRUCTURE_FILE" --sim-conf "$SIMULATION_FILE" $MNPBEM_ARG 2>&1 | tee "$TEMP_OUTPUT"
    fi

    PYTHON_EXIT_CODE=$?

    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        print_msg "Error: Simulation failed" "$RED"
        rm -f "$TEMP_OUTPUT"
        exit 1
    fi

    # Extract RUN_FOLDER from output
    RUN_FOLDER=$(grep "^RUN_FOLDER=" "$TEMP_OUTPUT" | cut -d'=' -f2)
    rm -f "$TEMP_OUTPUT"

    if [ -z "$RUN_FOLDER" ]; then
        print_msg "Error: Could not determine run folder" "$RED"
        exit 1
    fi

    print_msg "  Simulation completed successfully" "$GREEN"
    print_msg "    Run folder: $RUN_FOLDER" "$BLUE"
    echo ""

    # Step 2: Postprocess results
    print_msg "  Step 2/2: Processing and analyzing results..." "$YELLOW"

    # Create a temporary config file with updated paths
    # PostprocessManager calculates: output_dir + simulation_name
    # So we set output_dir to RUN_FOLDER and simulation_name to empty string
    TEMP_SIM_CONFIG=$(mktemp)
    cat "$SIMULATION_FILE" > "$TEMP_SIM_CONFIG"
    echo "" >> "$TEMP_SIM_CONFIG"
    echo "# Override for postprocessing to prevent path duplication" >> "$TEMP_SIM_CONFIG"
    echo "# PostprocessManager calculates: output_dir + simulation_name" >> "$TEMP_SIM_CONFIG"
    echo "# So we set output_dir to RUN_FOLDER and simulation_name to empty string" >> "$TEMP_SIM_CONFIG"
    echo "args['output_dir'] = '$RUN_FOLDER'" >> "$TEMP_SIM_CONFIG"
    echo "args['simulation_name'] = ''" >> "$TEMP_SIM_CONFIG"

    # Ensure log directory exists
    mkdir -p "$RUN_FOLDER/logs"

    if [ "$VERBOSE" = true ]; then
        python -u run_postprocess.py --str-conf "$STRUCTURE_FILE" --sim-conf "$TEMP_SIM_CONFIG" --verbose 2>&1 | tee -a "$RUN_FOLDER/logs/pipeline.log"
    else
        python run_postprocess.py --str-conf "$STRUCTURE_FILE" --sim-conf "$TEMP_SIM_CONFIG" >> "$RUN_FOLDER/logs/pipeline.log" 2>&1
    fi

    POSTPROCESS_EXIT_CODE=$?
    rm -f "$TEMP_SIM_CONFIG"

    if [ $POSTPROCESS_EXIT_CODE -ne 0 ]; then
        print_msg "Error: Failed to process results" "$RED"
        print_msg "  Check log file: $RUN_FOLDER/logs/pipeline.log" "$RED"
        exit 1
    fi
    print_msg "  Results processed successfully" "$GREEN"
    echo ""

    # Pipeline completed
    print_msg "=== Pipeline Completed Successfully ===" "$GREEN"
    echo ""
    print_msg "  All results saved in: $RUN_FOLDER/" "$BLUE"
    print_msg "    - Data files: simulation_results.npz, .txt, .csv, .json" "$BLUE"
    print_msg "    - Field analysis: field_analysis.json" "$BLUE"
    print_msg "    - Plots: simulation_spectrum.png, field_*.png" "$BLUE"
    print_msg "    - Logs: logs/pipeline.log" "$BLUE"
    echo ""
fi
