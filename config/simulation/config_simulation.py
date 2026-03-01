import os
from pathlib import Path

args = {}


# ============================================================================
# PARALLEL COMPUTING OPTIONS
# ============================================================================
# Enable parallel computing with Python multiprocessing
args['use_parallel'] = True  # Set to False to disable parallel computing

# Number of workers (cores) to use for multiprocessing pool
# Options:
#   - Integer (e.g., 10): Use exactly 10 workers
#   - 'auto': Automatically detect available cores (os.cpu_count())
#   - 'env': Read from environment variable MNPBEM_NUM_WORKERS
# args['num_workers'] = 'env'  # Recommended for Slurm clusters

# Alternative: specify exact number
args['num_workers'] = 2
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
#   - 'stat' : Quasistatic approximation (fast, suitable for small particles <50nm)
#   - 'ret'  : Retarded/full Maxwell equations (accurate, for larger particles >50nm)

args['simulation_type'] = 'stat'

# Interpolation method:
#   - 'curv' : Curved boundary elements (more accurate, recommended)
#   - 'flat' : Flat boundary elements (faster, less accurate)

args['interp'] = 'curv'

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
# Set to False to skip spectrum calculation and only calculate fields
# NOTE: If False, field_wavelength_idx cannot use 'peak', 'peak_ext', 'peak_sca'
#       (these require spectrum data to find peak wavelength)
#       Use 'middle', integer index, or wavelength list instead
args['calculate_cross_sections'] = True

# Calculate electric field distribution
args['calculate_fields'] = True

# Field calculation region (only used if calculate_fields=True)
args['field_region'] = {
    'x_range': [-80, 80, 161],  # [min, max, num_points] in nm
    'y_range': [0, 0, 1],       # xz-plane at y=0
    'z_range': [-80, 80, 161]   # [min, max, num_points] in nm
}

# Field calculation options
args['field_mindist'] = 0.5     # Minimum distance from particle surface (nm)
args['field_nmax'] = 2000       # Work off calculation in portions (for large grids)
args['field_wavelength_idx'] = 'peak'  # Which wavelength(s) to calculate fields
# Options:
#   - 'middle'      : Middle wavelength of range
#   - 'peak'        : Absorption peak (finds max for EACH polarization)
#   - 'peak_ext'    : Extinction peak
#   - 'peak_sca'    : Scattering peak
#   - integer (50)  : Specific wavelength index
#   - list [400, 500, 600, ...]  : List of wavelengths (nm), mapped to nearest indices
#
# Example for multiple wavelengths (20 wavelengths from 400-1000nm):
# args['field_wavelength_idx'] = list(range(400, 1001, 30))  # [400, 430, 460, ..., 970, 1000]

# ============================================================================
# FIELD DATA EXPORT OPTIONS
# ============================================================================

# Export field arrays to npz
# Set to True only if you need field data exported separately
# Full resolution data is always available in the main result file
args['export_field_arrays'] = False

# Field analysis options
args['field_hotspot_count'] = 10  # Number of hotspots to identify
args['field_hotspot_min_distance'] = 3  # Minimum distance between hotspots (grid points)

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Output directory for results
args['output_dir'] = os.path.join(Path.home(), 'research/mnpbem/sphere_test')

# Data file save format for simulation results
# Available: 'npz', 'mat', 'hdf5'
# 'npz' is recommended for Python workflows (fast, compact, native numpy)
# 'mat' for MATLAB compatibility
# 'hdf5' for large datasets or cross-language access
args['save_format'] = 'npz'

# Data file save formats for postprocessing exports
# Available: 'txt', 'csv', 'json'
args['output_formats'] = ['txt', 'csv', 'json']

# Generate plots
args['save_plots'] = True

# Plot formats
# Available: 'png', 'pdf', 'eps', 'svg'
args['plot_format'] = ['png', 'pdf']

# Plot DPI (resolution)
args['plot_dpi'] = 300
args['spectrum_xaxis'] = 'energy'

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================

# ============================================================================
# MIRROR SYMMETRY (for reducing computation time)
# ============================================================================
# Options:
#   - False : Disabled (default)
#   - 'xy'  : x=0 and y=0 plane symmetry (1/4 mesh, ~16x faster)
#   - 'x'   : y=0 plane symmetry only (1/2 mesh, ~4x faster)
#   - 'y'   : x=0 plane symmetry only (1/2 mesh, ~4x faster)
#
# Requirements:
#   1. Structure must have the corresponding symmetry
#   2. Excitation must be compatible:
#      - Planewave: x/y polarization with z propagation only
#      - Dipole: Supported
#      - EELS: NOT supported
#
# Homodimer (identical particles):
#   - Any arrangement -> 'xy' available
#
# Heterodimer (different particles):
#   - x-axis arrangement -> 'x' only (y=0 symmetry)
#   - z-axis arrangement -> 'xy' available (both on x=0, y=0 planes)
#
# Note: 'z', 'xz', 'yz' are NOT supported by MNPBEM
args['use_mirror_symmetry'] = False

# Iterative solver (for very large structures with >10,000 elements)
# Uses less memory but may be slower
# Enable if you encounter out-of-memory errors
args['use_iterative_solver'] = False

# Iterative solver options (only used when use_iterative_solver=True)
# Uncomment and modify to override defaults
# args['iterative_solver_type'] = 'gmres'    # 'gmres', 'cgs', 'bicgstab'
# args['iterative_tol'] = 1e-4               # 수렴 tolerance
# args['iterative_maxit'] = 200              # 최대 반복 횟수
# args['iterative_restart'] = None           # GMRES restart (None=auto)
# args['iterative_precond'] = 'hmat'         # 'hmat' 또는 'full'
# args['iterative_output'] = 0              # 0=silent, 1=수렴 정보 출력

# Nonlocal effects (advanced, for very small particles <5nm)
# Includes quantum effects at metal surfaces
# Requires additional setup
args['use_nonlocality'] = False

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

# Example 6: High accuracy calculation with field
# args['refine'] = 3
# args['relcutoff'] = 3
# args['mesh_density'] = 288  # Double standard density (set in structure config)
# args['simulation_type'] = 'ret'
# args['calculate_fields'] = True
# args['export_field_arrays'] = True  # Export field arrays to npz
