"""
Structure config (pymnpbem). Migrated from MATLAB mnpbem_simulation.
Source: config_str_auagcl_g3.py
"""

args = {}

# Structure identity
args['structure_name'] = 'test_au_ag_agcl_dimer'

# Structure type
args['structure'] = 'advanced_dimer_cube'

# Geometry
args['core_size'] = 47
args['shell_layers'] = [3, 1]
args['roundings'] = [0.25, 0.2, 0.15]
args['mesh_density'] = 12

# Placement / transform
args['gap'] = 3
args['offset'] = [0, 0, 0]
args['tilt_angle'] = 0
args['tilt_axis'] = [1, 0, 0]
args['rotation_angle'] = 0

# Medium
args['medium'] = 'water'

# Particle material list (inner -> outer)
args['materials'] = ['gold', 'silver', 'agcl']

# Custom refractive-index sources (e.g. constant epsilon, table file)
args['refractive_index_paths'] = {
    'agcl': {'type': 'constant', 'epsilon': 2.02},
}

# Substrate
args['use_substrate'] = False
