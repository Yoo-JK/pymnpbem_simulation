import os
from pathlib import Path

args = {}

# ============================================================================
# 30nm Au Sphere in Air
# ============================================================================
# MATLAB reference: ~/temporary/sphere/{stat,ret}/sphere_30nm/

args['structure_name'] = 'my_structure'

args['structure'] = 'sphere'
args['diameter'] = 30
args['mesh_density'] = 2
args['materials'] = ['gold']

args['medium'] = 'air'
args['refractive_index_paths'] = {}
args['use_substrate'] = False
