import os
from pathlib import Path

args = {}

# ============================================================================
# 30nm Au Sphere in Air on Glass Substrate
# ============================================================================
# Sphere center at z=0, radius=15nm, bottom at z=-15
# Glass substrate at z=-16 (1nm gap between sphere and substrate)

args['structure_name'] = 'my_structure'

args['structure'] = 'sphere'
args['diameter'] = 30
args['mesh_density'] = 2
args['materials'] = ['gold']

args['medium'] = 'air'
args['refractive_index_paths'] = {}
args['use_substrate'] = True
args['substrate'] = {
    'material': 'glass',
    'position': -16
}
