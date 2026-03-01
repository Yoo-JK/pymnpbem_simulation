import os
from pathlib import Path

args = {}

# ============================================================================
# 22x47 Au Nanorod in Water
# ============================================================================
# MATLAB reference: ~/temporary/mat2py/rod/{stat,ret}/aunr_22x47/

args['structure_name'] = '22x47_0'

args['structure'] = 'rod'
args['diameter'] = 22
args['height'] = 47
args['mesh_density'] = 2
args['materials'] = ['gold']

args['medium'] = 'water'
args['refractive_index_paths'] = {}
args['use_substrate'] = False
