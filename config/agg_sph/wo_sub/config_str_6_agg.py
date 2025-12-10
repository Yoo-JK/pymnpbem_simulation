"""
Configuration for Au Sphere Cluster on Au Substrate
Compact close-packed aggregate (1-7 spheres in contact)

Study plasmon coupling and hotspot formation in nanoparticle aggregates
"""

import os
from pathlib import Path

args = {}

# ============================================================================
# STRUCTURE
# ============================================================================
args['structure_name'] = 'au_cluster_5_on_substrate'
args['structure'] = 'sphere_cluster_aggregate'

# --- Cluster Parameters ---
args['n_spheres'] = 6  # Change to 1, 2, 3, 4, 5, 6, or 7
args['diameter'] = 50  # nm
args['gap'] = -0.1  # negative = 0.1nm overlap (conduction contact)
args['mesh_density'] = 144  # standard for spheres

# Structure type: N=5 → Pentagon (3 bottom, 2 top)

# ============================================================================
# MATERIALS
# ============================================================================
args['medium'] = {
        'type': 'constant',
        'epsilon': 1.459}
args['materials'] = ['gold']

# --- Substrate (Au substrate in contact) ---
args['use_substrate'] = False
args['substrate'] = {
    'material': 'gold',
    'position': -25.001
}

# Custom refractive index (optional)
