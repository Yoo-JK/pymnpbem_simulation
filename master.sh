#!/bin/bash
#
# master.sh — mnpbem_simulation-compatible pipeline entry point.
#
# Usage:
#   ./master.sh --str-conf <structure_config> --sim-conf <simulation_config> \
#               [--reanalyze] [--verbose] [-- <extra run_simulation.py flags>]
#
# Takes the same two configs as mnpbem_simulation, where the postprocess
# options live inline in the sim-conf. Runs, in order:
#   1. spectrum simulation
#   2. field pass (only when the sim-conf sets calculate_fields = True;
#      pymnpbem computes fields as a second pass over the sigma cache)
#   3. postprocess
#
# master.py is the native entry point and keeps its own semantics
# (--anal-conf decides whether analysis runs); this script is the
# mnpbem_simulation-shaped front end. (master.py 는 그대로 두고, 구버전 인터페이스만 여기서 제공.)
#

set -e

STR_CONF=""
SIM_CONF=""
REANALYZE=false
VERBOSE=""
EXTRA=()

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
        --)
            shift
            EXTRA=("$@")
            break
            ;;
        -h|--help)
            echo "Usage: $0 --str-conf <structure_config> --sim-conf <simulation_config> [--reanalyze] [--verbose] [-- <extra flags>]"
            echo ""
            echo "  --str-conf    Structure config .py (required)"
            echo "  --sim-conf    Simulation config .py, postprocess keys inline (required)"
            echo "  --reanalyze   Skip simulation, only run postprocess"
            echo "  --verbose     Verbose output"
            echo "  --            Everything after is forwarded to run_simulation.py"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$STR_CONF" ]; then
    echo "Error: --str-conf is required"
    exit 1
fi

if [ -z "$SIM_CONF" ]; then
    echo "Error: --sim-conf is required"
    exit 1
fi

if [ ! -f "$STR_CONF" ]; then
    echo "Error: Structure config file not found: $STR_CONF"
    exit 1
fi

if [ ! -f "$SIM_CONF" ]; then
    echo "Error: Simulation config file not found: $SIM_CONF"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

echo "============================================================"
echo "pyMNPBEM Simulation Pipeline"
echo "============================================================"
echo "Structure config:  $STR_CONF"
echo "Simulation config: $SIM_CONF"
echo "Reanalyze only:    $REANALYZE"
echo "============================================================"

WANT_FIELDS=$("$PY" - "$SCRIPT_DIR" "$SIM_CONF" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from pymnpbem_simulation.config import load_py_config
print('1' if load_py_config(sys.argv[2]).get('calculate_fields') is True else '0')
EOF
)

if [ "$REANALYZE" = false ]; then
    echo ""
    echo "[Step 1/3] Running simulation (spectrum) ..."
    echo "------------------------------------------------------------"
    "$PY" "$SCRIPT_DIR/run_simulation.py" \
        --str-conf "$STR_CONF" \
        --sim-conf "$SIM_CONF" \
        $VERBOSE "${EXTRA[@]}"

    if [ "$WANT_FIELDS" = "1" ]; then
        echo ""
        echo "[Step 2/3] Running field pass (calculate_fields = True) ..."
        echo "------------------------------------------------------------"
        "$PY" "$SCRIPT_DIR/run_simulation.py" \
            --str-conf "$STR_CONF" \
            --sim-conf "$SIM_CONF" \
            --fields \
            $VERBOSE "${EXTRA[@]}"
    else
        echo ""
        echo "[Step 2/3] Skipping field pass (calculate_fields is not True)"
    fi
else
    echo ""
    echo "[Step 1-2/3] Skipping simulation (--reanalyze mode)"
fi

echo ""
echo "[Step 3/3] Running postprocessing ..."
echo "------------------------------------------------------------"
"$PY" "$SCRIPT_DIR/run_postprocess.py" \
    --str-conf "$STR_CONF" \
    --sim-conf "$SIM_CONF"

echo ""
echo "============================================================"
echo "Pipeline completed successfully"
echo "============================================================"
