#!/bin/bash

# ============================================================================
# MNPBEM Automation Pipeline
# ============================================================================
# Main execution script for MNPBEM simulations and postprocessing
#
# Usage:
#   ./master.sh --str-conf <path> --sim-conf <path> [options]
#
# Options:
#   --reanalyze    Skip MATLAB simulation, only run postprocessing
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
VERBOSE=false
REANALYZE_MODE=false

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required:
    --str-conf PATH       Path to structure configuration file
    --sim-conf PATH      Path to simulation configuration file

Options:
    --reanalyze            Skip MATLAB simulation, only reanalyze existing results
    --verbose              Enable verbose output
    -h, --help             Show this help message

Examples:
  # Run full simulation + postprocessing
  ./master.sh --str-conf ./config/structures/sphere.py \\
              --sim-conf ./config/simulations/config.py \\
              --verbose

  # Reanalyze existing results (skip MATLAB)
  ./master.sh --str-conf ./config/structures/sphere.py \\
              --sim-conf ./config/simulations/config.py \\
              --reanalyze \\
              --verbose

Note:
  - In reanalyze mode, the .mat file is automatically found from output_dir in config
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
    # REANALYZE MODE: Postprocess existing .mat file only
    # ========================================================================
    
    print_msg "╔════════════════════════════════════════════════════════════╗" "$BLUE"
    print_msg "║         MNPBEM Reanalysis Mode                           ║" "$BLUE"
    print_msg "╚════════════════════════════════════════════════════════════╝" "$BLUE"
    echo ""
    print_msg "📄 Structure config:  $STRUCTURE_FILE" "$BLUE"
    print_msg "📄 Simulation config: $SIMULATION_FILE" "$BLUE"
    print_msg "📁 Output directory:  $OUTPUT_DIR" "$BLUE"
    echo ""
    
    # Check if .mat file exists
    MAT_FILE="$OUTPUT_DIR/simulation_results.mat"
    if [ ! -f "$MAT_FILE" ]; then
        print_msg "✗ Error: simulation_results.mat not found in $OUTPUT_DIR" "$RED"
        print_msg "   Run without --reanalyze to generate simulation results first" "$RED"
        exit 1
    fi
    
    print_msg "📊 Found existing results: $MAT_FILE" "$GREEN"
    echo ""
    
    # Run postprocessing only
    print_msg "📊 Reanalyzing results..." "$YELLOW"
    
    if [ "$VERBOSE" = true ]; then
        python run_postprocess.py --str-conf "$STRUCTURE_FILE" --sim-conf "$SIMULATION_FILE" --verbose
    else
        python run_postprocess.py --str-conf "$STRUCTURE_FILE" --sim-conf "$SIMULATION_FILE"
    fi
    
    POSTPROCESS_EXIT_CODE=$?
    
    if [ $POSTPROCESS_EXIT_CODE -ne 0 ]; then
        print_msg "✗ Error: Reanalysis failed (exit code: $POSTPROCESS_EXIT_CODE)" "$RED"
        exit 1
    fi
    
    print_msg "✓ Reanalysis completed successfully" "$GREEN"
    echo ""
    
    print_msg "╔════════════════════════════════════════════════════════════╗" "$GREEN"
    print_msg "║         Reanalysis Completed Successfully! 🎉            ║" "$GREEN"
    print_msg "╚════════════════════════════════════════════════════════════╝" "$GREEN"
    echo ""
    print_msg "📁 Updated results in: $OUTPUT_DIR/" "$BLUE"
    print_msg "   ├─ Field analysis: field_analysis.json (updated)" "$BLUE"
    print_msg "   ├─ Processed data: simulation_processed.* (updated)" "$BLUE"
    print_msg "   └─ Plots: *.png/pdf (regenerated)" "$BLUE"
    echo ""
    
else
    # ========================================================================
    # NORMAL MODE: Full simulation + postprocessing
    # ========================================================================
    
    print_msg "╔════════════════════════════════════════════════════════════╗" "$BLUE"
    print_msg "║         MNPBEM Automation Pipeline Started               ║" "$BLUE"
    print_msg "╚════════════════════════════════════════════════════════════╝" "$BLUE"
    echo ""
    print_msg "📄 Structure config:  $STRUCTURE_FILE" "$BLUE"
    print_msg "📄 Simulation config: $SIMULATION_FILE" "$BLUE"
    print_msg "📁 Output directory:  $OUTPUT_DIR" "$BLUE"
    echo ""
    
    # Step 1: Generate MATLAB simulation code
    print_msg "🔧 Step 1/3: Generating MATLAB simulation code..." "$YELLOW"
    
    TEMP_OUTPUT=$(mktemp)
    if [ "$VERBOSE" = true ]; then
        python run_simulation.py --str-conf "$STRUCTURE_FILE" --sim-conf "$SIMULATION_FILE" --verbose 2>&1 | tee "$TEMP_OUTPUT"
    else
        python run_simulation.py --str-conf "$STRUCTURE_FILE" --sim-conf "$SIMULATION_FILE" 2>&1 | tee "$TEMP_OUTPUT"
    fi
    
    PYTHON_EXIT_CODE=$?
    
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        print_msg "✗ Error: Failed to generate MATLAB code" "$RED"
        rm -f "$TEMP_OUTPUT"
        exit 1
    fi
    
    # Extract RUN_FOLDER from output
    RUN_FOLDER=$(grep "^RUN_FOLDER=" "$TEMP_OUTPUT" | cut -d'=' -f2)
    rm -f "$TEMP_OUTPUT"
    
    if [ -z "$RUN_FOLDER" ]; then
        print_msg "✗ Error: Could not determine run folder" "$RED"
        exit 1
    fi
    
    print_msg "✓ MATLAB code generated successfully" "$GREEN"
    print_msg "   Run folder: $RUN_FOLDER" "$BLUE"
    echo ""
    
    # Extract MNPBEM path from simulation config
    print_msg "📂 Reading MNPBEM path from configuration..." "$YELLOW"
    MNPBEM_PATH=$(python3 -c "
import sys
sys.path.insert(0, '.')
try:
    exec(open('$SIMULATION_FILE').read())
    if 'args' in dir() and 'mnpbem_path' in args:
        print(args['mnpbem_path'])
    else:
        print('/home/yoojk20/workspace/MNPBEM')
except Exception as e:
    print('/home/yoojk20/workspace/MNPBEM')
" 2>/dev/null)
    
    if [ -z "$MNPBEM_PATH" ]; then
        print_msg "✗ Error: MNPBEM path not found in config" "$RED"
        print_msg "   Please set 'mnpbem_path' in your simulation config file" "$RED"
        exit 1
    fi
    
    # Expand ~ to home directory if present
    MNPBEM_PATH="${MNPBEM_PATH/#\~/$HOME}"
    
    print_msg "   MNPBEM Path: $MNPBEM_PATH" "$BLUE"
    
    # Verify MNPBEM path exists
    if [ ! -d "$MNPBEM_PATH" ]; then
        print_msg "✗ Error: MNPBEM directory not found: $MNPBEM_PATH" "$RED"
        print_msg "   Please check 'mnpbem_path' in your simulation config" "$RED"
        exit 1
    fi
    print_msg "✓ MNPBEM path verified" "$GREEN"
    echo ""
    
    # Step 2: Run MATLAB simulation
    print_msg "⚡ Step 2/3: Running MATLAB simulation..." "$YELLOW"
    
    # Check if MATLAB is available
    if ! command -v matlab &> /dev/null; then
        print_msg "✗ Error: MATLAB not found in PATH" "$RED"
        exit 1
    fi
    
    # Check if simulation script exists in RUN_FOLDER
    if [ ! -f "$RUN_FOLDER/simulation_script.m" ]; then
        print_msg "✗ Error: simulation_script.m not found in $RUN_FOLDER" "$RED"
        exit 1
    fi
    
    # Run MATLAB with dynamic MNPBEM path
    cd "$RUN_FOLDER"
    
    # Start MATLAB in background to avoid tee pipe issues
    matlab -nodisplay -nodesktop -r "addpath(genpath('$MNPBEM_PATH')); run('simulation_script.m')" > "logs/matlab.log" 2>&1 &
    MATLAB_PID=$!
    
    if [ "$VERBOSE" = true ]; then
        # Verbose mode: Follow log file in real-time
        print_msg "  Following MATLAB output (PID: $MATLAB_PID)..." "$BLUE"
        echo ""
        
        # Use tail -f to follow log, will exit when MATLAB process ends
        tail -f "logs/matlab.log" --pid=$MATLAB_PID 2>/dev/null || true
        
        echo ""
        print_msg "  MATLAB process finished" "$BLUE"
    fi
    
    # Wait for MATLAB to complete
    wait $MATLAB_PID
    MATLAB_EXIT_CODE=$?
    
    cd - > /dev/null
    
    if [ $MATLAB_EXIT_CODE -ne 0 ]; then
        print_msg "✗ Error: MATLAB simulation failed (exit code: $MATLAB_EXIT_CODE)" "$RED"
        print_msg "Check log file: $RUN_FOLDER/logs/matlab.log" "$RED"
        exit 1
    fi
    print_msg "✓ MATLAB simulation completed successfully" "$GREEN"
    echo ""
    
    # Step 3: Postprocess results
    print_msg "📊 Step 3/3: Processing and analyzing results..." "$YELLOW"

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
    
    if [ "$VERBOSE" = true ]; then
        python run_postprocess.py --str-conf "$STRUCTURE_FILE" --sim-conf "$TEMP_SIM_CONFIG" --verbose 2>&1 | tee -a "$RUN_FOLDER/logs/pipeline.log"
    else
        python run_postprocess.py --str-conf "$STRUCTURE_FILE" --sim-conf "$TEMP_SIM_CONFIG" >> "$RUN_FOLDER/logs/pipeline.log" 2>&1
    fi
    
    POSTPROCESS_EXIT_CODE=$?
    rm -f "$TEMP_SIM_CONFIG"
    
    if [ $POSTPROCESS_EXIT_CODE -ne 0 ]; then
        print_msg "✗ Error: Failed to process results" "$RED"
        print_msg "Check log file: $RUN_FOLDER/logs/pipeline.log" "$RED"
        exit 1
    fi
    print_msg "✓ Results processed successfully" "$GREEN"
    echo ""
    
    # Pipeline completed
    print_msg "╔════════════════════════════════════════════════════════════╗" "$GREEN"
    print_msg "║         Pipeline Completed Successfully! 🎉              ║" "$GREEN"
    print_msg "╚════════════════════════════════════════════════════════════╝" "$GREEN"
    echo ""
    print_msg "📁 All results saved in: $RUN_FOLDER/" "$BLUE"
    print_msg "   ├─ Data files: simulation_results.mat, .txt, .csv, .json" "$BLUE"
    print_msg "   ├─ Field analysis: field_analysis.json" "$BLUE"
    print_msg "   ├─ Plots: simulation_spectrum.png, field_*.png" "$BLUE"
    print_msg "   └─ Logs: logs/matlab.log, pipeline.log" "$BLUE"
    echo ""
fi
