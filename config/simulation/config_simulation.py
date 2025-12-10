"""
pyMNPBEM Simulation Configuration

This file contains all simulation parameters for running
plasmonic BEM simulations using pyMNPBEM.
"""

import os
from pathlib import Path

args = {}

# ============================================================================
# pyMNPBEM PATH (REQUIRED)
# ============================================================================
# Path to your pyMNPBEM installation directory
# This should point to the directory containing the mnpbem package

args['pymnpbem_path'] = os.path.join(Path.home(), 'pyMNPBEM')

# Examples:
# args['pymnpbem_path'] = '/usr/local/pyMNPBEM'
# args['pymnpbem_path'] = Path.home() / 'pyMNPBEM'
# args['pymnpbem_path'] = '/opt/pyMNPBEM'

# ============================================================================
# SIMULATION NAME (IDENTIFIER)
# ============================================================================
# Give your simulation a descriptive name
args['simulation_name'] = 'au_sphere_spectrum'

# ============================================================================
# SIMULATION TYPE
# ============================================================================
# Simulation method:
#   - 'stat' : Quasistatic approximation (fast, suitable for small particles <50nm)
#   - 'ret'  : Retarded/full Maxwell equations (accurate, for larger particles >50nm)

args['simulation_type'] = 'stat'

# Interpolation method:
#   - 'curv' : Curved boundary elements (more accurate, recommended)
#   - 'flat' : Flat boundary elements (faster, less accurate)

args['interp'] = 'curv'

# Progress bar (waitbar):
#   - 0 : Off (recommended for batch jobs)
#   - 1 : On (shows progress)

args['waitbar'] = 1

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
# The system will automatically calculate:
#   - Unpolarized spectrum: sigma_unpol = (sigma_pol1 + sigma_pol2) / 2
#   - Unpolarized field: I_unpol = (I_pol1 + I_pol2) / 2
#                        enh_unpol = sqrt((enh1^2 + enh2^2) / 2)
#
# This follows the FDTD (Lumerical) convention for incoherent averaging.
# Reference: https://optics.ansys.com/hc/en-us/articles/1500006149562
#
# Example for unpolarized calculation (x and y polarizations are orthogonal):
#   args['polarizations'] = [[1, 0, 0], [0, 1, 0]]
#   args['propagation_dirs'] = [[0, 0, 1], [0, 0, 1]]
#
# For dipole excitation, use THREE ORTHOGONAL directions for unpolarized:
#   args['polarizations'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # x, y, z
#   -> sigma_unpol = (sigma_x + sigma_y + sigma_z) / 3
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
# CALCULATION OPTIONS
# ============================================================================

# Calculate optical cross sections (scattering, absorption, extinction)
args['calculate_cross_sections'] = True

# Calculate electric field distribution
args['calculate_fields'] = True

# Calculate surface charge distributions for mode analysis
args['calculate_surface_charges'] = True

# Field calculation region (only used if calculate_fields=True)
args['field_region'] = {
    'x_range': [-80, 80, 161],  # [min, max, num_points] in nm
    'y_range': [0, 0, 1],       # xz-plane at y=0
    'z_range': [-80, 80, 161]   # [min, max, num_points] in nm
}

# Field calculation options
args['field_mindist'] = 0.5     # Minimum distance from particle surface (nm)
args['field_wavelength_idx'] = 'peak'  # Which wavelength to calculate fields:
                                        # 'middle', 'peak', 'peak_ext', 'peak_sca',
                                        # 'peak_abs', or integer index

# ============================================================================
# FIELD DATA EXPORT OPTIONS
# ============================================================================

# Export field arrays to JSON (can create large files!)
# Set to True only if you need field data in JSON format
# Full resolution data is always available in numpy .npy files
args['export_field_arrays'] = False  # Exports downsampled field arrays to JSON

# Field analysis options
args['field_hotspot_count'] = 10  # Number of hotspots to identify
args['field_hotspot_min_distance'] = 3  # Minimum distance between hotspots (grid points)

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Output directory for results
args['output_dir'] = os.path.join(Path.home(), 'research/pymnpbem/results')

# Data file save formats (for postprocessing)
# Available: 'txt', 'csv', 'json'
args['output_formats'] = ['txt', 'csv', 'json']

# Generate plots
args['save_plots'] = True

# Plot formats
# Available: 'png', 'pdf', 'eps', 'svg'
args['plot_format'] = ['png', 'pdf']

# Plot DPI (resolution)
args['plot_dpi'] = 300

# Spectrum x-axis type
# Options: 'wavelength' (nm) or 'energy' (eV)
args['spectrum_xaxis'] = 'wavelength'

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================

# Mirror symmetry (for reducing computation time)
# Options: False, 'x', 'y', 'z', 'xy', 'xz', 'yz'
# Only use if your structure and excitation have the appropriate symmetry
args['use_mirror_symmetry'] = False

# Example: Use x-symmetry for symmetric dimer with x-polarization
# args['use_mirror_symmetry'] = 'x'

# Iterative solver (for very large structures with >10,000 elements)
# Uses less memory but may be slower
# Enable if you encounter out-of-memory errors
args['use_iterative_solver'] = False

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

# Example 5: High accuracy calculation with field
# args['refine'] = 3
# args['relcutoff'] = 3
# args['calculate_fields'] = True
# args['field_region'] = {
#     'x_range': [-50, 50, 201],
#     'y_range': [0, 0, 1],
#     'z_range': [-50, 50, 201]
# }
