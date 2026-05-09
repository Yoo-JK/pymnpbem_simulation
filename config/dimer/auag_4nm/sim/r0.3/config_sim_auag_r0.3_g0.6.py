"""
Simulation config (pymnpbem). Migrated from MATLAB mnpbem_simulation.
Source: config_sim_auag_r0.3_g0.6.py
"""

import os
from pathlib import Path

args = {}

# Parallel computing
args['use_parallel'] = True
args['num_workers'] = 5
args['max_comp_threads'] = 1
args['wavelength_chunk_size'] = 10

# Simulation identity
args['simulation_name'] = 'auag_r0.3_g0.6'

# Solver method
args['simulation_type'] = 'ret'
args['interp'] = 'curv'

# Excitation
args['excitation_type'] = 'planewave'
args['polarizations'] = [
        [1, 0, 0],
        [0, 1, 0]]
args['propagation_dirs'] = [
        [0, 0, 1],
        [0, 0, 1]]

# Wavelength range [start_nm, end_nm, n_points]
args['wavelength_range'] = [300, 1000, 140]

# Numerical accuracy
args['refine'] = 3
args['relcutoff'] = 3

# Calculation toggles
args['calculate_cross_sections'] = True
args['calculate_fields'] = False

# Field calculation options
args['field_region'] = {
    'x_range': [-80, 80, 161],
    'y_range': [0, 0, 1],
    'z_range': [-80, 80, 161],
}
args['field_mindist'] = 0.5
args['field_nmax'] = 2000
args['field_wavelength_idx'] = [510, 555, 583, 596, 605, 619, 634, 644, 671, 681, 706, 716, 770, 786, 859, 869, 891]
args['export_field_arrays'] = False
args['field_hotspot_count'] = 10
args['field_hotspot_min_distance'] = 3

# Output settings
args['output_dir'] = os.path.join(Path.home(), 'research/pymnpbem/auag_dimer_4nm')
args['output_formats'] = ['txt', 'csv', 'json']
args['save_plots'] = True
args['plot_format'] = ['png', 'pdf']
args['plot_dpi'] = 300
args['spectrum_xaxis'] = 'energy'

# Advanced
args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = True
args['use_nonlocality'] = False
args['run_eigenmode_analysis'] = True
