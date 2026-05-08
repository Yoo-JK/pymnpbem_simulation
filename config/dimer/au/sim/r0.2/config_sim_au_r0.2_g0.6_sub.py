"""
Simulation config (pymnpbem). Migrated from MATLAB mnpbem_simulation.
Source: config_sim_au_r0.2_g0.6_sub.py
"""

import os
from pathlib import Path

args = {}

# Parallel computing
args['use_parallel'] = True
args['num_workers'] = 4
args['max_comp_threads'] = 1
args['wavelength_chunk_size'] = 10

# Simulation identity
args['simulation_name'] = 'au_r0.2_g0.6_sub'

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
args['wavelength_range'] = [500, 1000, 100]

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
args['field_wavelength_idx'] = [568, 666, 690, 708, 864, 888, 552, 673, 698, 877]
args['export_field_arrays'] = False
args['field_hotspot_count'] = 10
args['field_hotspot_min_distance'] = 3

# Output settings
args['output_dir'] = os.path.join(Path.home(), 'research/mnpbem/dimer')
args['output_formats'] = ['txt', 'csv', 'json']
args['save_plots'] = True
args['plot_format'] = ['png', 'pdf']
args['plot_dpi'] = 300
args['spectrum_xaxis'] = 'energy'

# Advanced
args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = False
args['use_nonlocality'] = False
args['run_eigenmode_analysis'] = True
