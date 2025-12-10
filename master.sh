#!/bin/bash
#
# Master script for pyMNPBEM simulation pipeline
#
# Usage:
#   ./master.sh --str-conf <structure_config> --sim-conf <simulation_config> [--reanalyze] [--verbose]
#
# Arguments:
#   --str-conf    Path to structure configuration file (required)
#   --sim-conf    Path to simulation configuration file (required)
#   --reanalyze   Skip simulation, only run postprocessing (optional)
#   --verbose     Enable verbose output (optional)
#

set -e  # Exit on error

# Default values
VERBOSE=""
REANALYZE=false
STR_CONF=""
SIM_CONF=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --str-conf)
            STR_CONF="$2"
            shift 2
            ;;
        --sim-conf)
            SIM_CONF="$2"
            shift 2
            ;;
        --reanalyze)
            REANALYZE=true
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --str-conf <structure_config> --sim-conf <simulation_config> [--reanalyze] [--verbose]"
            echo ""
            echo "Arguments:"
            echo "  --str-conf    Path to structure configuration file (required)"
            echo "  --sim-conf    Path to simulation configuration file (required)"
            echo "  --reanalyze   Skip simulation, only run postprocessing"
            echo "  --verbose     Enable verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$STR_CONF" ]; then
    echo "Error: --str-conf is required"
    exit 1
fi

if [ -z "$SIM_CONF" ]; then
    echo "Error: --sim-conf is required"
    exit 1
fi

# Check if config files exist
if [ ! -f "$STR_CONF" ]; then
    echo "Error: Structure config file not found: $STR_CONF"
    exit 1
fi

if [ ! -f "$SIM_CONF" ]; then
    echo "Error: Simulation config file not found: $SIM_CONF"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "pyMNPBEM Simulation Pipeline"
echo "============================================================"
echo "Structure config: $STR_CONF"
echo "Simulation config: $SIM_CONF"
echo "Reanalyze only: $REANALYZE"
echo "============================================================"

# Step 1: Run simulation (unless --reanalyze is specified)
if [ "$REANALYZE" = false ]; then
    echo ""
    echo "[Step 1/2] Running simulation..."
    echo "------------------------------------------------------------"

    # Run simulation and capture output
    SIMULATION_OUTPUT=$(python "$SCRIPT_DIR/run_simulation.py" \
        --str-conf "$STR_CONF" \
        --sim-conf "$SIM_CONF" \
        $VERBOSE)

    # Print output
    echo "$SIMULATION_OUTPUT"

    # Extract RUN_FOLDER from output
    RUN_FOLDER=$(echo "$SIMULATION_OUTPUT" | grep "^RUN_FOLDER=" | cut -d'=' -f2)

    if [ -z "$RUN_FOLDER" ]; then
        echo "Warning: Could not extract RUN_FOLDER from simulation output"
    else
        echo ""
        echo "Run folder: $RUN_FOLDER"
    fi
else
    echo ""
    echo "[Step 1/2] Skipping simulation (--reanalyze mode)"
fi

# Step 2: Run postprocessing
echo ""
echo "[Step 2/2] Running postprocessing..."
echo "------------------------------------------------------------"

python "$SCRIPT_DIR/run_postprocess.py" \
    --str-conf "$STR_CONF" \
    --sim-conf "$SIM_CONF" \
    $VERBOSE

echo ""
echo "============================================================"
echo "Pipeline completed successfully"
echo "============================================================"
