import os
from pathlib import Path

args = {}
args['mnpbem_path'] = os.path.join(Path.home(), 'workspace/MNPBEM')

args['simulation_name'] = 'johnson/Au99Cu01'
args['simulation_type'] = 'ret'
args['interp'] = 'curv'

args['waitbar'] = 1

args['excitation_type'] = 'planewave'
args['polarizations'] = [
    [1, 0, 0],
    [0, 1, 0],
]
args['propagation_dirs'] = [
    [0, 0, 1],
    [0, 0, 1],
]
args['wavelength_range'] = [400, 800, 100]
args['refine'] = 3
args['relcutoff'] = 3

args['calculate_cross_sections'] = True
args['output_dir'] = os.path.join(Path.home(), 'research/mnpbem/rod')

args['output_formats'] = ['txt']
args['save_plots'] = True
args['plot_format'] = ['png']
args['plot_dpi'] = 300
args['spectrum_xaxis'] = 'energy'
args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = False
args['use_nonlocality'] = False
args['matlab_executable'] = 'matlab'
args['matlab_options'] = '-nodisplay -nosplash -nodesktop'
