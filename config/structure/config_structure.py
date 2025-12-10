"""
pyMNPBEM Structure Configuration - Complete Recipe Book

This file contains examples for ALL available structure types.
Uncomment the section you need and customize the parameters.

Structure Types Available:
- Single particles: sphere, cube, rod, ellipsoid, triangle
- Core-shell: core_shell_sphere, core_shell_cube, core_shell_rod
- Dimers: dimer_sphere, dimer_cube, dimer_core_shell_cube
- Advanced: advanced_dimer_cube, sphere_cluster_aggregate
- DDA shapes: from_shape
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
# args['mesh_density'] = 144
# args['materials'] = ['gold', 'silver']  # [core, shell]

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

# --- Core and Shells (inner -> outer) ---
args['core_size'] = 30  # Au core size (nm)
args['shell_layers'] = [5, 3]  # [inner->outer] shell thickness (nm)

# --- Materials (inner -> outer) ---
args['materials'] = ['gold', 'silver', 'sio2']  # [core, inner_shell, outer_shell]

# --- Per-Layer Rounding (inner -> outer) ---
args['roundings'] = [0.25, 0.2, 0.15]  # [core, inner, outer]
# OR use single value for all layers:
# args['rounding'] = 0.2

# --- Mesh Density ---
args['mesh_density'] = 12

# --- Dimer Configuration ---
args['gap'] = 5  # surface-to-surface distance (nm)

args['offset'] = [0, 2, 0]  # [x, y, z] additional shift for particle 2 (nm)

args['tilt_angle'] = 15  # degrees, particle 2 tilt

args['tilt_axis'] = [0, 1, 0]  # [x, y, z] rotation axis for tilt

args['rotation_angle'] = 0  # degrees, particle 2 rotation around z-axis

# Transformation order for Particle 2:
#   1. Rotation (around z-axis)
#   2. Tilt (around custom axis)
#   3. Shift (to gap position)
#   4. Offset (fine-tuning)

# ============================================================================
# SECTION 4.5: SPHERE CLUSTER AGGREGATE
# ============================================================================
# Compact close-packed sphere clusters (1-7 spheres in contact)

# args['structure'] = 'sphere_cluster_aggregate'
# args['n_spheres'] = 5  # 1-7 (see structure types below)
# args['diameter'] = 50  # nm
# args['gap'] = -0.1  # negative = 0.1nm overlap (contact)
# args['mesh_density'] = 144
# args['materials'] = ['gold']

# --- Structure Types by n_spheres ---
# N=1: Single sphere
# N=2: Dimer (horizontal, end-to-end)
# N=3: Triangle (2 bottom, 1 top)
# N=4: Square (2x2 grid)
# N=5: Pentagon (3 bottom, 2 top)
# N=6: Hexagon (3 bottom, 3 top, compact)
# N=7: Hexagon (4 bottom, 3 top, extended)

# ============================================================================
# SECTION 5: DDA SHAPE FILE
# ============================================================================
# Import structures from DDA simulation .shape files

# args['structure'] = 'from_shape'
# args['shape_file'] = os.path.join(Path.home(), 'dataset/mnpbem/particle.shape')
# args['voxel_size'] = 2.0  # physical size of each voxel (nm)
# args['voxel_method'] = 'surface'  # 'surface' (fast) or 'cube' (accurate)
# args['materials'] = ['gold', 'silver']  # maps to mat_type indices in .shape file

# ============================================================================
# COMMON SETTINGS (All Structures)
# ============================================================================

# --- Medium (Surrounding Environment) ---
args['medium'] = 'water'
# Options: 'air', 'water', 'vacuum', 'glass', 'sio2'
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
#
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
#    - Can use gap < 0 for overlapping particles
#
# 3. Rounding:
#    - 0.1: very sharp edges (lightning rod effect)
#    - 0.25: standard (good balance)
#    - 0.5: very round edges (sphere-like)
#
# 4. Material Order:
#    - Built-in structures: [shell, core] or [outer->inner]
#    - DDA: matches mat_type indices [1->N]
#    - Advanced dimer: [core, inner->outer]
#
# ============================================================================
