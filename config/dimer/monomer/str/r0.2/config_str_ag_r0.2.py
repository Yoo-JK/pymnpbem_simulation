"""
Structure config (pymnpbem). Migrated from MATLAB mnpbem_simulation.
Source: config_str_ag_r0.2.py
"""

args = {}

# Structure identity
args['structure_name'] = 'ag_monomer'

# Structure type
args['structure'] = 'advanced_monomer_cube'

# Geometry
args['core_size'] = 51
args['shell_layers'] = []
args['roundings'] = [0.2]
args['mesh_density'] = 2

# Medium
args['medium'] = 'water'

# Particle material list (inner -> outer)
args['materials'] = ['silver']

# Custom refractive-index sources (e.g. constant epsilon, table file)
args['refractive_index_paths'] = {
    'agcl': {'type': 'constant', 'epsilon': 2.02},
}

# Substrate
args['use_substrate'] = False
