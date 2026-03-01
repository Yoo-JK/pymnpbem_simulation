import os
from pathlib import Path

args = {}

# ============================================================================
# Stat (Quasistatic) Simulation - 30nm Au Sphere on Glass Substrate
# ============================================================================

args['use_parallel'] = False
args['num_workers'] = 2
args['wavelength_chunk_size'] = 10

args['simulation_name'] = 'sphere_30nm_glass'
args['simulation_type'] = 'stat'
args['interp'] = 'curv'

args['excitation_type'] = 'planewave'
args['polarizations'] = [[1, 0, 0], [0, 1, 0]]
args['propagation_dirs'] = [[0, 0, 1], [0, 0, 1]]

args['wavelength_range'] = [400, 800, 100]

args['refine'] = 3
args['relcutoff'] = 3

args['calculate_cross_sections'] = True
args['calculate_fields'] = True
args['field_region'] = {
    'x_range': [-80, 80, 161],
    'y_range': [0, 0, 1],
    'z_range': [-80, 80, 161],
}
args['field_mindist'] = 0.5
args['field_nmax'] = 2000
args['field_wavelength_idx'] = 'peak'
args['export_field_arrays'] = False
args['field_hotspot_count'] = 10
args['field_hotspot_min_distance'] = 3

args['output_dir'] = os.path.join(Path.home(), 'scratch/mat2py/sphere/stat/sphere_30nm_glass')
args['save_format'] = 'npz'
args['output_formats'] = ['txt', 'csv', 'json']
args['save_plots'] = True
args['plot_format'] = ['png', 'pdf']
args['plot_dpi'] = 300
args['spectrum_xaxis'] = 'energy'

args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = False
args['use_nonlocality'] = False
