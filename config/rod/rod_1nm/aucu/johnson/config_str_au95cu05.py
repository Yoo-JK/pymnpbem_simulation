import os
from pathlib import Path

args = {}
args['structure_name'] = 'Au95Cu05_vac/johnson_christy'
args['structure'] = 'core_shell_rod'
args['core_diameter'] = 9  # nm
args['shell_thickness'] = 2  # nm (total diameter = 25nm)
args['height'] = 37  # nm (along z-axis)
args['rod_mesh'] = [15, 20, 20]   # [nphi, ntheta, nz]: circumference / caps / length
args['materials'] = ['gold', 'Au95Cu05']  # [core, shell]
args['medium'] = 'air'
args['refractive_index_paths'] = {
        'au100cu0': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_100.00_Cu_0.00.txt'),
        'au99cu01': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_99.00_Cu_1.00.txt'),
        'au98cu02': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_98.00_Cu_2.00.txt'),
        'au97cu03': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_97.00_Cu_3.00.txt'),
        'au96cu04': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_96.00_Cu_4.00.txt'),
        'au95cu05': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_95.00_Cu_5.00.txt'),
        'au94cu06': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_94.00_Cu_6.00.txt'),
        'au93cu07': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_93.00_Cu_7.00.txt'),
        'au92cu08': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_92.00_Cu_8.00.txt'),
        'au91cu09': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_91.00_Cu_9.00.txt'),
        'au90cu10': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au_90.00_Cu_10.00.txt')
        }
# Example:
# args['refractive_index_paths'] = {
#     'gold': os.path.join(Path.home(), 'materials/gold_palik.dat'),
#     'silver': os.path.join(Path.home(), 'materials/silver_jc.dat')
# }

args['use_substrate'] = True
args['substrate'] = {
    'material': 'glass',  # or 'silicon', custom dict
    'position': -7.5,  # z-coordinate of interface (nm)
}

