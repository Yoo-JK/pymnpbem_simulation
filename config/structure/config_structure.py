"""
MNPBEM Structure Configuration - Complete Recipe Book

This file contains examples for ALL available structure types.
Uncomment the section you need and customize the parameters.

For detailed documentation, see: config/structure/guide_structure.txt
"""

import os
from pathlib import Path

args = {}

# ============================================================================
# STRUCTURE NAME
# ============================================================================
args['structure_name'] = 'my_structure'

# ============================================================================
# SECTION 1: SINGLE PARTICLES
# ============================================================================

# --- Sphere ---
# args['structure'] = 'sphere'
# args['diameter'] = 50  # nm
# args['mesh_density'] = 144
# args['materials'] = ['gold']

# --- Cube ---
# args['structure'] = 'cube'
# args['size'] = 40  # edge length (nm)
# args['rounding'] = 0.25  # 0-1, smaller = sharper edges
# args['mesh_density'] = 12
# args['materials'] = ['gold']

# --- Rod (Cylinder) ---
# args['structure'] = 'rod'
# args['diameter'] = 20  # nm
# args['height'] = 80  # nm (along z-axis)
# args['mesh_density'] = 144   # Choose 'mesh_density' or 'rod_mesh'
# args['rod_mesh'] = [15, 20, 20]   # [nphi, ntheta, nz]: circumference / caps / length
# args['materials'] = ['gold']

# --- Ellipsoid ---
# args['structure'] = 'ellipsoid'
# args['axes'] = [20, 30, 40]  # [x, y, z] semi-axes (nm)
# args['mesh_density'] = 144
# args['materials'] = ['gold']

# --- Triangle ---
# args['structure'] = 'triangle'
# args['side_length'] = 50  # nm
# args['thickness'] = 10  # nm (extrusion height)
# args['materials'] = ['gold']

# ============================================================================
# SECTION 2: CORE-SHELL STRUCTURES
# ============================================================================

# --- Core-Shell Sphere ---
# args['structure'] = 'core_shell_sphere'
# args['core_diameter'] = 40  # nm
# args['shell_thickness'] = 10  # nm
# args['mesh_density'] = 144
# args['materials'] = ['silver', 'gold']  # [shell, core]

# --- Core-Shell Cube ---
# args['structure'] = 'core_shell_cube'
# args['core_size'] = 30  # nm
# args['shell_thickness'] = 5  # nm
# args['rounding'] = 0.25
# args['mesh_density'] = 12
# args['materials'] = ['silver', 'gold']  # [shell, core]

# --- Core-Shell Rod (Nanorod) ---
# args['structure'] = 'core_shell_rod'
# args['core_diameter'] = 15  # nm
# args['shell_thickness'] = 5  # nm (total diameter = 25nm)
# args['height'] = 80  # nm (total length along z-axis)
# args['mesh_density'] = 144   # Choose 'mesh_density' or 'rod_mesh'
# args['rod_mesh'] = [15, 20, 20]   # [nphi, ntheta, nz]: circumference / caps / length
# args['materials'] = ['gold', 'silver']  # [core, shell]
# Example: Gold core nanorod with silver shell
# Perfect for studying plasmonic nanorods with tunable properties

# ============================================================================
# SECTION 3: SIMPLE DIMERS (Two Particles)
# ============================================================================

# --- Dimer Sphere ---
# args['structure'] = 'dimer_sphere'
# args['diameter'] = 50  # nm
# args['gap'] = 5  # surface-to-surface gap (nm)
# args['mesh_density'] = 144
# args['materials'] = ['gold']

# --- Dimer Cube ---
# args['structure'] = 'dimer_cube'
# args['size'] = 40  # nm
# args['gap'] = 10  # surface-to-surface gap (nm)
# args['rounding'] = 0.25
# args['mesh_density'] = 12
# args['materials'] = ['gold']

# --- Dimer Core-Shell Cube (Simple) ---
# args['structure'] = 'dimer_core_shell_cube'
# args['core_size'] = 20  # nm
# args['shell_thickness'] = 5  # nm
# args['gap'] = 10  # nm
# args['rounding'] = 0.25
# args['mesh_density'] = 12
# args['materials'] = ['silver', 'gold']  # [shell, core]

# ============================================================================
# SECTION 4: ADVANCED DIMER CUBE 
# ============================================================================
# Multi-shell dimer with full transformation control
# Features:
#   - Multiple shell layers (Au@Ag@AgCl, etc.)
#   - Per-layer rounding control
#   - Gap, offset, tilt, rotation control

args['structure'] = 'advanced_dimer_cube'

# --- Core and Shells (inner → outer) ---
args['core_size'] = 30  # Au core size (nm)
args['shell_layers'] = [5, 3]  # [inner→outer] shell thickness (nm)
# Example: [5, 3] means:
#   - First shell (inner): 5 nm
#   - Second shell (outer): 3 nm
# For single shell: [5]
# For triple shell: [5, 3, 2]

# --- Materials (inner → outer) ---
args['materials'] = ['gold', 'silver', 'agcl']  # [core, inner_shell, outer_shell]
# Order: core → inner shells → outer shells
# Length must equal: 1 (core) + number of shells

# --- Per-Layer Rounding (inner → outer) ---
args['roundings'] = [0.25, 0.2, 0.15]  # [core, inner, outer]
# Order matches materials: [core, shell1, shell2, ...]
# OR use single value for all layers:
# args['rounding'] = 0.2

# --- Mesh Density ---
args['mesh_density'] = 12
# Recommended: 12-16 for cubes

# --- Dimer Configuration ---
args['gap'] = 5  # surface-to-surface distance (nm)
# Can be < 1 nm for strong coupling, or negative for overlap

args['offset'] = [0, 2, 0]  # [x, y, z] additional shift for particle 2 (nm)
# [0, 0, 0] = end-to-end (default)
# [0, 10, 0] = side-by-side (L-shape)
# [0, 0, 5] = vertical stacking

args['tilt_angle'] = 15  # degrees, particle 2 tilt
# 0 = no tilt (parallel)
# 90 = perpendicular (T-shape)

args['tilt_axis'] = [0, 1, 0]  # [x, y, z] rotation axis for tilt
# [0, 1, 0] = y-axis (left-right tilt)
# [1, 0, 0] = x-axis (front-back tilt)
# [0, 0, 1] = z-axis (in-plane rotation)
# [1, 1, 0] = diagonal axis

args['rotation_angle'] = 0  # degrees, particle 2 rotation around z-axis
# Additional rotation after tilt
# 0 = no rotation
# 45 = 45° twist

# Transformation order for Particle 2:
#   1. Rotation (around z-axis)
#   2. Tilt (around custom axis)
#   3. Shift (to gap position)
#   4. Offset (fine-tuning)

# --- Example Configurations ---

# Example 1: Simple Au@Ag dimer (no tilt/rotation)
# args['core_size'] = 40
# args['shell_layers'] = [5]
# args['materials'] = ['gold', 'silver']
# args['rounding'] = 0.2  # same for all
# args['gap'] = 5
# args['offset'] = [0, 0, 0]
# args['tilt_angle'] = 0
# args['rotation_angle'] = 0

# Example 2: T-shaped Au@Ag@AgCl dimer
# args['core_size'] = 30
# args['shell_layers'] = [5, 3]
# args['materials'] = ['gold', 'silver', 'agcl']
# args['roundings'] = [0.25, 0.2, 0.15]
# args['gap'] = 3  # narrow gap for strong coupling
# args['offset'] = [0, 0, 0]
# args['tilt_angle'] = 90  # perpendicular
# args['tilt_axis'] = [0, 1, 0]
# args['rotation_angle'] = 0

# Example 3: Complex 3D twisted dimer
# args['core_size'] = 25
# args['shell_layers'] = [5, 3]
# args['materials'] = ['gold', 'silver', 'agcl']
# args['roundings'] = [0.3, 0.2, 0.1]  # round core, sharp outer
# args['gap'] = 2
# args['offset'] = [2, 3, 1]  # 3D offset
# args['tilt_angle'] = 25
# args['tilt_axis'] = [1, 1, 0]  # diagonal tilt
# args['rotation_angle'] = 30

# Example 4: Side-by-side with sharp edges
# args['core_size'] = 35
# args['shell_layers'] = [5]
# args['materials'] = ['gold', 'silver']
# args['roundings'] = [0.1, 0.1]  # very sharp
# args['gap'] = 5
# args['offset'] = [0, 15, 0]  # shifted to side
# args['tilt_angle'] = 0
# args['rotation_angle'] = 0

# ============================================================================
# SECTION 4.5: SPHERE CLUSTER AGGREGATE
# ============================================================================
# Compact close-packed sphere clusters (1-7 spheres in contact)
# Perfect for studying plasmon coupling in nanoparticle aggregates

# args['structure'] = 'sphere_cluster_aggregate'
# args['n_spheres'] = 5  # 1-7 (see structure types below)
# args['diameter'] = 50  # nm
# args['gap'] = -0.1  # negative = 0.1nm overlap (contact)
# args['mesh_density'] = 144

# --- Structure Types by n_spheres ---
# N=1: Single sphere
# N=2: Dimer (horizontal, end-to-end)
# N=3: Triangle (2 bottom, 1 top)
# N=4: Square (2×2 grid)
# N=5: Pentagon (3 bottom, 2 top)
# N=6: Hexagon (3 bottom, 3 top, compact)
# N=7: Hexagon (4 bottom, 3 top, extended)

# All spheres are positioned in XY plane (z=0)
# Gap < 0 creates conduction contact (0.1nm overlap = true contact)
# Perfect for studying hotspot formation and coupled plasmon modes

# args['materials'] = ['gold']

# Example 1: Gold trimer on substrate
# args['structure'] = 'sphere_cluster_aggregate'
# args['n_spheres'] = 3
# args['diameter'] = 50
# args['gap'] = -0.1  # contact
# args['mesh_density'] = 144
# args['materials'] = ['gold']
# args['use_substrate'] = True
# args['substrate'] = {'material': 'gold', 'position': -25.01}

# Example 2: Silver hexagonal cluster
# args['structure'] = 'sphere_cluster_aggregate'
# args['n_spheres'] = 6
# args['diameter'] = 30
# args['gap'] = -0.05  # tight contact
# args['mesh_density'] = 144
# args['materials'] = ['silver']

# Example 3: Large 7-sphere aggregate
# args['structure'] = 'sphere_cluster_aggregate'
# args['n_spheres'] = 7
# args['diameter'] = 50
# args['gap'] = -0.1
# args['mesh_density'] = 144
# args['materials'] = ['gold']
# Perfect for maximum field enhancement at multiple junctions

# ============================================================================
# SECTION 5: DDA SHAPE FILE
# ============================================================================
# Import structures from DDA simulation .shape files

# args['structure'] = 'from_shape'
# args['shape_file'] = os.path.join(Path.home(), 'dataset/mnpbem/particle.shape')
# args['voxel_size'] = 2.0  # physical size of each voxel (nm)
# args['voxel_method'] = 'surface'  # 'surface' (fast) or 'cube' (accurate)

# --- Materials for DDA ---
# Order maps to mat_type indices in .shape file:
#   materials[0] → mat_type 1
#   materials[1] → mat_type 2
#   materials[2] → mat_type 3
# args['materials'] = ['gold', 'silver']

# Example: Multi-material DDA structure
# args['structure'] = 'from_shape'
# args['shape_file'] = os.path.join(Path.home(), 'dataset/mnpbem/au_ag_core_shell.shape')
# args['voxel_size'] = 1.0
# args['voxel_method'] = 'surface'
# args['materials'] = ['gold', 'silver', 'sio2']
# In .shape file:
#   mat_type 1 → gold
#   mat_type 2 → silver
#   mat_type 3 → sio2

# ============================================================================
# COMMON SETTINGS (All Structures)
# ============================================================================

# --- Medium (Surrounding Environment) ---
args['medium'] = 'water'
# Options: 'air', 'water', 'vacuum', 'glass'
# OR custom constant: args['medium'] = {'type': 'constant', 'epsilon': 1.77}

# --- Custom Refractive Index Paths (Optional) ---
args['refractive_index_paths'] = {}
# Override built-in material data with custom files
# Example:
# args['refractive_index_paths'] = {
#     'gold': os.path.join(Path.home(), 'materials/gold_palik.dat'),
#     'silver': os.path.join(Path.home(), 'materials/silver_jc.dat')
# }
# File format: [wavelength(nm), n, k] per line

# --- Substrate (Optional) ---
args['use_substrate'] = False
# Uncomment to add substrate (half-space) below nanoparticle:
# args['use_substrate'] = True
# args['substrate'] = {
#     'material': 'glass',  # or 'silicon', custom dict
#     'position': 0,  # z-coordinate of interface (nm)
# }

# ============================================================================
# QUICK REFERENCE: Structure Parameters
# ============================================================================
#
# Single Particles:
#   sphere: diameter, mesh_density
#   cube: size, rounding, mesh_density
#   rod: diameter, height, mesh_density
#   ellipsoid: axes [x,y,z], mesh_density
#   triangle: side_length, thickness
#
# Core-Shell:
#   core_shell_sphere: core_diameter, shell_thickness, mesh_density
#   core_shell_cube: core_size, shell_thickness, rounding, mesh_density
#
# Simple Dimers:
#   dimer_sphere: diameter, gap, mesh_density
#   dimer_cube: size, gap, rounding, mesh_density
#   dimer_core_shell_cube: core_size, shell_thickness, gap, rounding, mesh_density
#
# Advanced Dimer Cube:
#   core_size, shell_layers, materials, roundings (or rounding),
#   gap, offset, tilt_angle, tilt_axis, rotation_angle, mesh_density

# Sphere Cluster Aggregate:
#   n_spheres (1-7), diameter, gap, mesh_density
#
# DDA Shape:
#   shape_file, voxel_size, voxel_method, materials
#
# ============================================================================
# TIPS
# ============================================================================
#
# 1. Mesh Density Guidelines:
#    - Spheres/ellipsoids: 144 (good balance)
#    - Cubes: 12-16 (sufficient for most cases)
#    - Complex shapes: increase for accuracy, but slower
#
# 2. Gap Size:
#    - > 10 nm: weak coupling
#    - 2-5 nm: strong coupling, large field enhancement
#    - < 1 nm: ultra-strong coupling (quantum effects may matter)
#    - Can use gap < 0 for overlapping particles (unusual but allowed)
#
# 3. Rounding:
#    - 0.1: very sharp edges (lightning rod effect)
#    - 0.25: standard (good balance)
#    - 0.5: very round edges (sphere-like)
#
# 4. Material Order:
#    - Built-in structures: always [shell, core] or [outer→inner]
#    - DDA: matches mat_type indices [1→N]
#    - Advanced dimer: [core, inner→outer]
#
# 5. Transformation Order (advanced_dimer_cube):
#    - Design tip: think "rotate flat, then tilt up, then move to position"
#    - For T-shape: tilt_angle=90, rotation_angle=0
#    - For twisted: use both tilt and rotation
#
# ============================================================================
