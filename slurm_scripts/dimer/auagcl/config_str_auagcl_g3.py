"""
Test Configuration: Advanced Dimer Cube
Au@Ag@AgCl Triple-Layer Dimer with constant AgCl refractive index
"""

import os
from pathlib import Path

args = {}

# ============================================================================
# STRUCTURE NAME
# ============================================================================
args['structure_name'] = 'test_au_ag_agcl_dimer'

# ============================================================================
# ADVANCED DIMER CUBE - Au@Ag@AgCl
# ============================================================================
args['structure'] = 'advanced_dimer_cube'

# --- Core and Shells (inner → outer) ---
args['core_size'] = 47  # Au core: 30 nm
args['shell_layers'] = [3, 1]  # [Ag: 5nm, AgCl: 3nm]

# --- Materials (inner → outer) ---
# AgCl will be defined as custom constant refractive index below
args['materials'] = ['gold', 'silver', 'agcl']

# --- Per-Layer Rounding (inner → outer) ---
args['roundings'] = [0.25, 0.2, 0.15]  # [Au: round, Ag: medium, AgCl: sharp]
# OR use single value:
# args['rounding'] = 0.2

# --- Mesh Density ---
args['mesh_density'] = 12

# --- Dimer Configuration ---
args['gap'] = 3  # 5nm gap for strong coupling
args['offset'] = [0, 0, 0]  # End-to-end (no offset)
args['tilt_angle'] = 0  # 15° tilt
args['tilt_axis'] = [1, 0, 0] 
args['rotation_angle'] = 0  # No additional rotation

# ============================================================================
# MATERIALS
# ============================================================================

# --- Medium ---
args['medium'] = 'water'

# --- Custom Refractive Index: AgCl as constant ---
# AgCl: n ≈ 2.0, k ≈ 0 (transparent in visible range)
# Calculate epsilon = (n + ik)^2 = n^2 (since k≈0)
# For n=2.0: epsilon = 4.0

args['refractive_index_paths'] = {
    # AgCl as constant refractive index
    'agcl': {'type': 'constant', 'epsilon': 2.02}  # n=2.0 → epsilon=4.0
}

# Optional: Use custom data files for Au and Ag
# args['refractive_index_paths'] = {
#     'gold': os.path.join(Path.home(), 'materials/gold_palik.dat'),
#     'silver': os.path.join(Path.home(), 'materials/silver_jc.dat'),
#     'agcl': {'type': 'constant', 'epsilon': 4.0}
# }

# ============================================================================
# NOTES
# ============================================================================
# 
# AgCl Refractive Index Options:
# 
# Option 1: Constant epsilon (current setup)
#   args['refractive_index_paths'] = {
#       'agcl': {'type': 'constant', 'epsilon': 4.0}
#   }
#   → epsilon = 4.0 for all wavelengths
#
# Option 2: Constant complex refractive index (with absorption)
#   # For n=2.0, k=0.1: epsilon = (2.0 + 0.1i)^2 = 3.99 + 0.4i
#   args['refractive_index_paths'] = {
#       'agcl': {'type': 'constant', 'epsilon': 3.99 + 0.4j}
#   }
#   → But MNPBEM uses epsilon, not n+ik directly
#
# Option 3: Wavelength-dependent from file
#   args['refractive_index_paths'] = {
#       'agcl': {'type': 'table', 'file': './materials/agcl.dat'}
#   }
#   → File format: [wavelength(nm), n, k] per line
#
# Option 4: No custom path (use built-in if available)
#   args['refractive_index_paths'] = {}
#   → Falls back to MNPBEM built-in 'agcl' if it exists
#
# Recommended: Option 1 for simple testing, Option 3 for accurate results
#
# ============================================================================

# ============================================================================
# SUBSTRATE (Optional)
# ============================================================================
args['use_substrate'] = False
# Uncomment to add glass substrate:
# args['use_substrate'] = True
# args['substrate'] = {
#     'material': 'glass',
#     'position': 0,
# }
