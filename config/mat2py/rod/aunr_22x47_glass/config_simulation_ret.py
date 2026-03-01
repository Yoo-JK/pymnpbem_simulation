import os
from pathlib import Path

args = {}

args['mnpbem_path'] = os.path.join(Path.home(), 'workspace/MNPBEM')

# ============================================================================
# Ret (Retarded) Simulation - 22x47 Au Nanorod on Glass Substrate
# ============================================================================

args['use_parallel'] = False
args['num_workers'] = 1
args['wavelength_chunk_size'] = 10

args['simulation_name'] = 'aunr_22x47_glass'
args['simulation_type'] = 'ret'
args['interp'] = 'curv'

args['excitation_type'] = 'planewave'
args['polarizations'] = [[1, 0, 0], [0, 1, 0]]
args['propagation_dirs'] = [[0, 0, 1], [0, 0, 1]]

args['wavelength_range'] = [400, 900, 200]

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
args['field_wavelength_idx'] = 'peak_sca'
args['export_field_arrays'] = False
args['field_hotspot_count'] = 10
args['field_hotspot_min_distance'] = 3

args['output_dir'] = os.path.join(Path.home(), 'research/mat2py/rod/ret/aunr_22x47_glass')
args['save_format'] = 'npz'
args['output_formats'] = ['txt']
args['save_plots'] = True
args['plot_format'] = ['png']
args['plot_dpi'] = 300
args['spectrum_xaxis'] = 'energy'

args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = False
args['use_nonlocality'] = False
