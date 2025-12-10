import os
from pathlib import Path

args = {}

# ============================================================================
# PARALLEL COMPUTING
# ============================================================================
args['use_parallel'] = False
args['num_workers'] = 2  # Adjust: N=1-3: 8-16, N=4-5: 16-32, N=6-7: 32-64
args['wavelength_chunk_size'] = 10

# ============================================================================
# PATHS
# ============================================================================
args['mnpbem_path'] = os.path.join(Path.home(), 'workspace/MNPBEM')
args['output_dir'] = os.path.join(Path.home(), 'research/mnpbem/sphere_cluster')

# ============================================================================
# SIMULATION SETTINGS
# ============================================================================
args['simulation_name'] = 'w_sub/1_agg'
args['simulation_type'] = 'ret'  # 'ret' for 50nm spheres
args['interp'] = 'curv'
args['waitbar'] = 0

# ============================================================================
# EXCITATION
# ============================================================================
args['excitation_type'] = 'planewave'
args['polarizations'] = [
    [1, 0, 0],
    [0, 1, 0]
]
args['propagation_dirs'] = [
    [0, 0, 1],
    [0, 0, 1]
]

# ============================================================================
# WAVELENGTH
# ============================================================================
args['wavelength_range'] = [400, 800, 100]  # [start, end, n_points]

# ============================================================================
# NUMERICAL ACCURACY
# ============================================================================
args['refine'] = 3
args['relcutoff'] = 3

# ============================================================================
# CROSS SECTIONS
# ============================================================================
args['calculate_cross_sections'] = True

# ============================================================================
# FIELD CALCULATION
# ============================================================================
args['calculate_fields'] = True

args['field_region'] = {
    'x_range': [-150, 150, 151],  # Adjust: N≤2: ±80, N=3-4: ±100, N=5-6: ±120, N=7: ±150
    'y_range': [-100, 100, 101],
    'z_range': [0, 0, 1],  # 5nm above cluster
}

args['field_wavelength_idx'] = 'peak'  # Auto-detect absorption peak
args['field_mindist'] = 0.5
args['field_nmax'] = 2000

args['export_field_arrays'] = True
args['field_hotspot_count'] = 10
args['field_hotspot_min_distance'] = 3

# ============================================================================
# OUTPUT
# ============================================================================
args['output_formats'] = ['txt']
args['save_plots'] = True
args['plot_format'] = ['png']
args['plot_dpi'] = 300
args['spectrum_xaxis'] = 'eV'

# ============================================================================
# ADVANCED
# ============================================================================
args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = True
args['use_h2_compression'] = True
args['use_nonlocality'] = False

# ============================================================================
# MATLAB
# ============================================================================
args['matlab_executable'] = 'matlab'
args['matlab_options'] = '-nodisplay -nosplash -nodesktop'

