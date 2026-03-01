import os
from pathlib import Path

args = {}

# ============================================================================
# 22x47 Au Nanorod in Water on Glass Substrate
# ============================================================================
# Rod oriented along x-axis, center at z=0
# Rod z-extent: diameter/2 = 11nm, bottom at z=-11
# Glass substrate at z=-12 (1nm gap between rod and substrate)

args['structure_name'] = '22x47_0'

args['structure'] = 'rod'
args['diameter'] = 22
args['height'] = 47
args['mesh_density'] = 2
args['materials'] = ['gold']

args['medium'] = 'water'
args['refractive_index_paths'] = {}
args['use_substrate'] = True
args['substrate'] = {
    'material': 'glass',
    'position': -12
}
