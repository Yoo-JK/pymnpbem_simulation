import os
from pathlib import Path

args = {}
args['structure_name'] = 'Au_vac/table'

args['structure'] = 'rod'
args['diameter'] = 9.0  # nm
args['height'] = 33  # nm (along z-axis)
args['rod_mesh'] = [15, 20, 20]   # [nphi, ntheta, nz]: circumference / caps / length
args['materials'] = ['Au100Cu0']
args['medium'] = 'air'
# Options: 'air', 'water', 'vacuum', 'glass'
# OR custom constant: args['medium'] = {'type': 'constant', 'epsilon': 1.77}

args['refractive_index_paths'] = {
        'au100cu0': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_100.00_Cu_0.00.txt')
        }
# Example:
# args['refractive_index_paths'] = {
#     'gold': os.path.join(Path.home(), 'materials/gold_palik.dat'),
#     'silver': os.path.join(Path.home(), 'materials/silver_jc.dat')
# }

args['use_substrate'] = True
args['substrate'] = {
    'material': 'glass',  # or 'silicon', custom dict
    'position': -4.501,  # z-coordinate of interface (nm)
}

