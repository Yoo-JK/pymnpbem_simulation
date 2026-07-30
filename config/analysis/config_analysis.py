"""
MNPBEM Analysis Configuration - Complete Recipe Book

This file contains all postprocess/analysis parameters. Pass it via
--anal-conf to run_postprocess.py or master.py.

Usage:
  python master.py \
      --str-conf config/structure/config_structure.py \
      --sim-conf config/simulation/config_simulation.py \
      --anal-conf config/analysis/config_analysis.py

  # Analysis only (using existing simulation output):
  python run_postprocess.py \
      --result /path/to/results/my_sim/spectrum.npz \
      --anal-conf config/analysis/config_analysis.py

The config file must define an 'args' dict.
"""

import os

args = {}

# ============================================================================
# RESULT FILE  (required for run_postprocess.py standalone; ignored by master.py)
# ============================================================================
# args['result'] = '/path/to/results/my_simulation/spectrum.npz'

# ============================================================================
# ANALYZERS  (comma-separated or list)
# ============================================================================
# Available analyzers:
#   'spectrum'      : Cross-section analysis + spectrum plot (always recommended)
#   'fano'          : Fano resonance fit
#   'eigenmode'     : Quasistatic eigenmode decomposition (needs --config)
#   'fano-analysis' : Bright/dark mode + multi-Lorentzian Fano pipeline
#   'multipole'     : Multipole decomposition (needs --config)
args['analyzers'] = ['spectrum']

# Example: multiple analyzers
# args['analyzers'] = ['spectrum', 'fano', 'eigenmode']

# ============================================================================
# CALCULATE CROSS SECTIONS  (passed through from simulation)
# ============================================================================
# If True (default), spectrum data (ext/sca/abs) is available.
# If False, field-only mode — 'spectrum' analyzer cannot run.
args['calculate_cross_sections'] = True

# ============================================================================
# FIELD CALCULATION OPTIONS  (only used if calculate_fields=True in sim config)
# ============================================================================
# Calculate electric field distribution
args['calculate_fields'] = False

# Field calculation region (only used if calculate_fields=True)
# args['field_region'] = {
#     'x_range': [-80, 80, 161],  # [min, max, num_points] in nm
#     'y_range': [0, 0, 1],       # xz-plane at y=0
#     'z_range': [-80, 80, 161],  # [min, max, num_points] in nm
# }

# Minimum distance from particle surface (nm)
# args['field_mindist'] = 0.5

# Work off field calculation in portions (for large grids)
# args['field_nmax'] = 2000

# Which wavelength(s) to calculate fields at:
#   - 'middle'      : Middle wavelength of range
#   - 'peak'        : Absorption peak (finds max for EACH polarization)
#   - 'peak_ext'    : Extinction peak
#   - 'peak_sca'    : Scattering peak
#   - integer (50)  : Specific wavelength index
#   - list [400, 500, 600, ...] : List of wavelengths (nm), mapped to nearest indices
# args['field_wavelength_idx'] = 'peak'

# Export field arrays to JSON (can create large files!)
# Full resolution data is always available in field_data.npz
args['export_field_arrays'] = False

# Number of hotspots to identify
args['field_hotspot_count'] = 10

# Minimum distance between hotspots (nm)
args['field_hotspot_min_distance'] = 3

# ============================================================================
# SPECTRUM PLOT OPTIONS
# ============================================================================
# x-axis for spectrum plots
#   - 'wavelength' : x-axis in nm
#   - 'energy'     : x-axis in eV
args['spectrum_xaxis'] = 'energy'
# Examples:
# args['spectrum_xaxis'] = 'wavelength'
# args['spectrum_xaxis'] = 'energy'

# ============================================================================
# OUTPUT FILE FORMATS
# ============================================================================
# Available: 'npz', 'h5', 'csv', 'json', 'txt'
# args['export_formats'] = 'npz,csv,txt'

# Generate plots
args['save_plots'] = True

# Plot formats
# Available: 'png', 'pdf', 'eps', 'svg'
args['plot_format'] = ['png', 'pdf']

# Plot DPI (resolution)
args['plot_dpi'] = 300

# ============================================================================
# PLASMON EIGENMODE ANALYSIS  (run_eigenmode_analysis)
# ============================================================================
# QS eigendecomposition (plasmonmode) + SVD (data-driven) + Retarded
# eigendecomposition + Multipole character + Fano fit pipeline.
# Several surface_charge data wavelengths needed (field_wavelength_idx list).
# EIT vs asymmetric Fano feature distinguished via delta_phi(lambda) curve.
#
# Requires 'eigenmode' in analyzers and --config YAML for particle rebuild.

args['run_eigenmode_analysis'] = False

# --- Optional sub-keys (defaults in parentheses) ---

# Number of eigenmodes to compute
# args['eigenmode_n'] = 10

# Number of top modes to plot
# args['eigenmode_top_k'] = 5

# Wavelength for retarded eigenmode (nm); None = middle of wavelength list
# args['retarded_eigen_wavelength'] = 550

# Fano target wavelengths (eV) for delta_phi plot vertical markers
# args['fano_target_wavelengths'] = [1.43, 1.79, 1.91]

# SVD rank auto-determination error threshold
# args['svd_rank_threshold'] = 1.0e-3

# ============================================================================
# FANO FIT OPTIONS
# ============================================================================
# Number of Fano peaks (>=2 uses multi_fano_fit)
# args['fano_peaks'] = 1

# Fano dip energies in eV for fano-analysis analyzer
# args['fano_features'] = [1.43, 1.79, 1.91]

# Polarization index for fano-analysis sigma (default: 0)
# args['fano_pol'] = 0

# Path to quasistatic full-eig .npz cache (keys: ene, vr, dvec)
# Computed and saved here if absent (HEAVY computation)
# args['eig_cache'] = '/path/to/eig_cache.npz'

# ============================================================================
# MULTIPOLE DECOMPOSITION OPTIONS
# ============================================================================
# Multipole expansion order
# args['max_l'] = 4

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================
# Postprocess output directory (default: alongside result file)
# args['output'] = '/path/to/analysis_output'

# ============================================================================
# EXAMPLES
# ============================================================================

# Example 1: Spectrum + Fano fit
# args['analyzers'] = ['spectrum', 'fano']
# args['spectrum_xaxis'] = 'energy'
# args['fano_peaks'] = 1

# Example 2: Full eigenmode analysis
# args['analyzers'] = ['spectrum', 'eigenmode']
# args['run_eigenmode_analysis'] = True
# args['eigenmode_n'] = 10
# args['eigenmode_top_k'] = 5
# args['retarded_eigen_wavelength'] = 600
# args['fano_target_wavelengths'] = [1.43, 1.79]

# Example 3: Fano phase analysis (bright/dark mode decomposition)
# args['analyzers'] = ['spectrum', 'fano-analysis']
# args['fano_features'] = [1.43, 1.79, 1.91]
# args['fano_pol'] = 0
# args['eig_cache'] = '/path/to/eig.npz'

# Example 4: Full export (all formats + multipole)
# args['analyzers'] = ['spectrum', 'fano', 'multipole']
# args['export_formats'] = 'npz,csv,json,txt'
# args['max_l'] = 4
# args['save_plots'] = True
# args['plot_format'] = ['png', 'pdf']
# args['plot_dpi'] = 300
