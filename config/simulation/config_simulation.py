"""
MNPBEM Simulation Configuration - Complete Recipe Book

This file contains all simulation parameters (excitation, wavelength,
numerics, compute, materials, output). It does NOT contain postprocess/
analysis parameters — put those in config/analysis/config_analysis.py.

Usage:
  python run_simulation.py \
      --str-conf config/structure/config_structure.py \
      --sim-conf config/simulation/config_simulation.py

The config file must define an 'args' dict.
"""

import os
from pathlib import Path

args = {}

# ============================================================================
# PARALLEL COMPUTING OPTIONS
# ============================================================================
# Enable parallel computing with multiple cores
args['use_parallel'] = True  # Set to False to disable parallel computing

# Number of workers (cores) to use
# Options:
#   - Integer (e.g., 10): Use exactly 10 workers
#   - 'auto': Automatically detect available cores
#   - 'env': Read from environment variable MNPBEM_NUM_WORKERS
# args['num_workers'] = 'env'  # Recommended for Slurm clusters

# Alternative: specify exact number
args['num_workers'] = 2
args['max_comp_threads'] = 64
args['wavelength_chunk_size'] = 10

# ============================================================================
# SIMULATION NAME (IDENTIFIER)
# ============================================================================
# Give your simulation a descriptive name
args['simulation_name'] = 'au_sphere_spectrum'

# ============================================================================
# SIMULATION TYPE
# ============================================================================
# Simulation method:
#   - 'ret'  : Retarded/full Maxwell equations (accurate, for larger particles)
#   - 'stat' : Quasistatic approximation (fast, suitable for small particles)

args['simulation_type'] = 'ret'

# Interpolation method:
#   - 'curv' : Curved boundary elements (more accurate, recommended)
#   - 'flat' : Flat boundary elements (faster, less accurate)

args['interp'] = 'curv'

# Wait bar (progress indicator):
#   - 0 : Off (recommended for batch jobs)
#   - 1 : On

args['waitbar'] = 0

# ============================================================================
# EXCITATION TYPE
# ============================================================================
# Type of excitation:
#   - 'planewave' : Plane wave illumination (most common)
#   - 'dipole'    : Point dipole excitation (for LDOS, decay rates)
#   - 'eels'      : Electron energy loss spectroscopy

args['excitation_type'] = 'planewave'

# --- Plane Wave Configuration ---
# Only used if excitation_type == 'planewave'

# Polarization direction(s) - can specify multiple
# Each polarization is a 3D vector [x, y, z]
#
# ============================================================================
# UNPOLARIZED LIGHT CALCULATION (FDTD-style, automatic detection)
# ============================================================================
# To calculate unpolarized light response, specify TWO ORTHOGONAL polarizations.
# The system will automatically detect orthogonal polarizations and calculate:
#   - Unpolarized spectrum: sigma_unpol = (sigma_pol1 + sigma_pol2) / 2
#   - Unpolarized field: I_unpol = (I_pol1 + I_pol2) / 2
#
# Example for unpolarized calculation (x and y polarizations are orthogonal):
#   args['polarizations'] = [[1, 0, 0], [0, 1, 0]]
#   args['propagation_dirs'] = [[0, 0, 1], [0, 0, 1]]
# ============================================================================

args['polarizations'] = [
    [1, 0, 0],  # x-polarization
    [0, 1, 0],  # y-polarization
]

# Propagation direction(s) - can specify multiple
# Each direction is a 3D unit vector [x, y, z]
args['propagation_dirs'] = [
    [0, 0, 1],  # Propagating in +z direction
    [0, 0, 1],  # Propagating in +z direction (for 2nd polarization)
]

# --- Dipole Configuration ---
# Only used if excitation_type == 'dipole'

# args['dipole_position'] = [0, 0, 15]  # Position in nm [x, y, z]
# args['dipole_moment'] = [0, 0, 1]     # Dipole moment direction [x, y, z]

# --- EELS Configuration ---
# Only used if excitation_type == 'eels'

# args['impact_parameter'] = [10, 0]  # Impact parameter in nm [x, y]
# args['beam_energy'] = 200e3         # Beam energy in eV
# args['beam_width'] = 0.2            # Beam width in nm

# ============================================================================
# WAVELENGTH RANGE
# ============================================================================
# Wavelength range for spectrum calculation
# Format: [start_nm, end_nm, num_points]

args['wavelength_range'] = [400, 800, 100]  # 400-800 nm, 100 points

# Examples:
# args['wavelength_range'] = [300, 1500, 240]  # Broad UV to NIR
# args['wavelength_range'] = [550, 550, 1]     # Single wavelength

# ============================================================================
# NUMERICAL ACCURACY PARAMETERS
# ============================================================================

# Refinement level for numerical integration
# Higher = more accurate but slower
# Typical values: 1-3
args['refine'] = 3

# Relative cutoff for interaction matrices
# Higher = more accurate but more memory
# Typical values: 2-3, default is 3 for sufficient accuracy
args['relcutoff'] = 3

# ============================================================================
# MATERIALS
# ============================================================================

# Medium (surrounding environment)
args['medium'] = 'water'
# Options: 'air', 'water', 'vacuum', 'glass'

# Custom refractive index paths (optional)
args['refractive_index_paths'] = {}
# Override built-in material data with custom files
# Example:
# args['refractive_index_paths'] = {
#     'gold': os.path.join(Path.home(), 'materials/gold_palik.dat'),
#     'agcl': {'type': 'constant', 'epsilon': 2.02},
# }

# ============================================================================
# SUBSTRATE (Optional)
# ============================================================================

args['use_substrate'] = False
# Uncomment to add substrate (half-space) below nanoparticle:
# args['use_substrate'] = True
# args['substrate'] = {
#     'material': 'glass',  # or 'silicon', custom dict
#     'gap': 0.001,  # distance from particle bottom to substrate surface (nm)
# }

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================

# Mirror symmetry (for reducing computation time)
# Options:
#   - False : Disabled (default)
#   - 'xy'  : x=0 and y=0 plane symmetry (1/4 mesh, ~16x faster)
#   - 'x'   : y=0 plane symmetry only (1/2 mesh, ~4x faster)
#   - 'y'   : x=0 plane symmetry only (1/2 mesh, ~4x faster)
args['use_mirror_symmetry'] = False

# Iterative solver (for very large structures with >10,000 elements)
# Uses less memory but may be slower
# Enable if you encounter out-of-memory errors
args['use_iterative_solver'] = False

# Nonlocal effects (advanced, for very small particles <5nm)
# Includes quantum effects at metal surfaces
# Requires additional setup
args['use_nonlocality'] = False

# ============================================================================
# GPU PRECISION
# ============================================================================
#   gpu_precision = 'fp64' : complex128 dense LU (accurate, default)
#   gpu_precision = 'fp32' : complex64 dense LU (~14x faster on RTX A6000)
#       Validation (vs FP64): Au dimer spectrum worst 4.5e-4 — within 1e-3 BEM tolerance.
#   (GPU execution is enabled via run_simulation.py --n-gpus-per-worker)
args['gpu_precision'] = 'fp64'

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Output directory for results
args['output_dir'] = os.path.join(Path.home(), 'research/pymnpbem/my_run')

# Data file save formats
# Available: 'npz', 'json', 'csv', 'txt'
args['output_formats'] = ['npz', 'json', 'png']

# ============================================================================
# ADDITIONAL SIMULATION EXAMPLES
# ============================================================================

# Example 1: Single wavelength field calculation
# args['wavelength_range'] = [550, 550, 1]  # Single wavelength
# args['calculate_fields'] = True
# args['field_region'] = {
#     'x_range': [-100, 100, 201],
#     'y_range': [-100, 100, 201],
#     'z_range': [0, 0, 1]
# }

# Example 2: Broadband spectrum (UV to NIR)
# args['wavelength_range'] = [300, 1500, 240]
# args['simulation_type'] = 'ret'  # Use retarded for broad range

# Example 3: Angle-resolved measurements
# args['polarizations'] = [[1, 0, 0]] * 37  # Same polarization
# # Generate angles from 0 to 90 degrees
# import numpy as np
# angles = np.linspace(0, 90, 37)
# args['propagation_dirs'] = [
#     [0, np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))]
#     for a in angles
# ]

# Example 4: Dipole emission study
# args['excitation_type'] = 'dipole'
# args['dipole_position'] = [0, 0, 10]
# args['dipole_moment'] = [0, 0, 1]
# args['wavelength_range'] = [400, 800, 80]

# Example 5: EELS line scan
# args['excitation_type'] = 'eels'
# args['beam_energy'] = 200e3
# args['beam_width'] = 0.2
# # For line scan, vary impact_parameter in a loop (not shown here)
# args['impact_parameter'] = [10, 0]

# Example 6: Mirror symmetry for speed
# args['use_mirror_symmetry'] = 'xy'
# args['simulation_type'] = 'ret'  # auto-promoted to ret_mirror

# Example 7: Iterative solver for large meshes
# args['use_iterative_solver'] = True
# args['simulation_type'] = 'ret'  # auto-promoted to ret_iter

# Example 8: With substrate
# args['use_substrate'] = True
# args['substrate'] = {'material': 'glass', 'gap': 0.001}
# args['simulation_type'] = 'ret'  # auto-promoted to ret_layer
