"""
Structure config (pymnpbem). Migrated from MATLAB mnpbem_simulation.
Source: config_str_au_r0.3_g0.0.py
"""

args = {}

# Structure identity
args['structure_name'] = 'au_dimer'

# Structure type
args['structure'] = 'connected_dimer_cube'

# Geometry
args['core_size'] = 47
args['rounding'] = 0.3
args['mesh_density'] = 2

# Placement / transform
args['gap'] = -0.1
args['offset'] = [0, 0, 0]
args['tilt_angle'] = 0
args['tilt_axis'] = [1, 0, 0]
args['rotation_angle'] = 0

# Medium
args['medium'] = 'water'

# Particle material list (inner -> outer)
args['materials'] = ['gold']

# Custom refractive-index sources (e.g. constant epsilon, table file)
args['refractive_index_paths'] = {}

# Substrate
args['use_substrate'] = False
