"""
Geometry Builder for pyMNPBEM-based simulations.

Converts structure configuration dictionaries into pyMNPBEM particle objects.
Supports all structure types from the original MATLAB-based system:
- Single particles: sphere, cube, rod, ellipsoid, triangle
- Core-shell: core_shell_sphere, core_shell_cube, core_shell_rod
- Dimers: dimer_sphere, dimer_cube, dimer_core_shell_cube
- Advanced: advanced_dimer_cube, sphere_cluster_aggregate
- DDA shapes: from_shape
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import sys
import os

# Add pyMNPBEM to path if needed
PYMNPBEM_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pyMNPBEM')
if os.path.exists(PYMNPBEM_PATH):
    sys.path.insert(0, PYMNPBEM_PATH)


class GeometryBuilder:
    """
    Builds particle geometries from configuration dictionaries.

    Supports all structure types from the original MNPBEM simulation system.
    """

    def __init__(self, structure_config: Dict[str, Any], pymnpbem_path: Optional[str] = None):
        """
        Initialize the geometry builder.

        Args:
            structure_config: Dictionary containing structure parameters
            pymnpbem_path: Path to pyMNPBEM installation (optional)
        """
        self.config = structure_config

        # Import pyMNPBEM modules
        if pymnpbem_path:
            sys.path.insert(0, pymnpbem_path)

        try:
            from mnpbem.particles.shapes import (
                trisphere, tricube, trirod, triellipsoid,
                trinanodisk, tritorus, tricone, triplate, triprism, tripolygon
            )
            from mnpbem.particles import Particle

            self.shapes = {
                'trisphere': trisphere,
                'tricube': tricube,
                'trirod': trirod,
                'triellipsoid': triellipsoid,
                'trinanodisk': trinanodisk,
                'tritorus': tritorus,
                'tricone': tricone,
                'triplate': triplate,
                'triprism': triprism,
                'tripolygon': tripolygon
            }
            self.Particle = Particle
            self._pymnpbem_available = True
        except ImportError as e:
            print(f"Warning: Could not import pyMNPBEM modules: {e}")
            self._pymnpbem_available = False
            self.shapes = {}
            self.Particle = None

    def build(self) -> Tuple[List[Any], List[List[int]]]:
        """
        Build particles from configuration.

        Returns:
            Tuple of (particles_list, inout_list):
                - particles_list: List of Particle objects
                - inout_list: List of [inside, outside] material indices for each particle
        """
        structure_type = self.config.get('structure', 'sphere')

        # Dispatch to appropriate builder method
        builder_methods = {
            'sphere': self._build_sphere,
            'cube': self._build_cube,
            'rod': self._build_rod,
            'ellipsoid': self._build_ellipsoid,
            'triangle': self._build_triangle,
            'core_shell_sphere': self._build_core_shell_sphere,
            'core_shell_cube': self._build_core_shell_cube,
            'core_shell_rod': self._build_core_shell_rod,
            'dimer_sphere': self._build_dimer_sphere,
            'dimer_cube': self._build_dimer_cube,
            'dimer_core_shell_cube': self._build_dimer_core_shell_cube,
            'advanced_dimer_cube': self._build_advanced_dimer_cube,
            'sphere_cluster_aggregate': self._build_sphere_cluster,
            'from_shape': self._build_from_shape,
        }

        if structure_type not in builder_methods:
            raise ValueError(f"Unknown structure type: {structure_type}")

        return builder_methods[structure_type]()

    def _get_mesh_density(self, default: int = 144) -> int:
        """Get mesh density from config with default."""
        return self.config.get('mesh_density', default)

    def _get_rounding(self, default: float = 0.25) -> float:
        """Get rounding parameter from config."""
        return self.config.get('rounding', default)

    # =========================================================================
    # SINGLE PARTICLES
    # =========================================================================

    def _build_sphere(self) -> Tuple[List[Any], List[List[int]]]:
        """Build a single sphere."""
        diameter = self.config.get('diameter', 50)
        n = self._get_mesh_density(144)

        sphere = self.shapes['trisphere'](n, diameter)

        # inout: [inside_material_idx, outside_material_idx]
        # Convention: 1 = medium, 2 = first material, etc.
        return [sphere], [[2, 1]]

    def _build_cube(self) -> Tuple[List[Any], List[List[int]]]:
        """Build a single cube with optional rounding."""
        size = self.config.get('size', 40)
        rounding = self._get_rounding(0.25)
        n = self._get_mesh_density(12)

        cube = self.shapes['tricube'](n, size, edge_rounding=rounding)

        return [cube], [[2, 1]]

    def _build_rod(self) -> Tuple[List[Any], List[List[int]]]:
        """Build a nanorod (cylinder with hemispherical caps)."""
        diameter = self.config.get('diameter', 20)
        height = self.config.get('height', 80)

        # Handle different mesh specification methods
        # trirod expects n as tuple (nphi, ntheta, nz)
        if 'rod_mesh' in self.config:
            nphi, ntheta, nz = self.config['rod_mesh']
            rod = self.shapes['trirod'](diameter, height, n=(nphi, ntheta, nz))
        else:
            # Default mesh density as tuple
            rod = self.shapes['trirod'](diameter, height, n=(15, 20, 20))

        return [rod], [[2, 1]]

    def _build_ellipsoid(self) -> Tuple[List[Any], List[List[int]]]:
        """Build an ellipsoid with specified semi-axes."""
        axes = self.config.get('axes', [20, 30, 40])
        n = self._get_mesh_density(144)

        ellipsoid = self.shapes['triellipsoid'](n, tuple(axes))

        return [ellipsoid], [[2, 1]]

    def _build_triangle(self) -> Tuple[List[Any], List[List[int]]]:
        """Build a triangular nanoparticle (extruded triangle)."""
        side_length = self.config.get('side_length', 50)
        thickness = self.config.get('thickness', 10)
        n_z = self.config.get('triangle_nz', 10)

        # Create triangular polygon vertices
        h = side_length * np.sqrt(3) / 2
        polygon = np.array([
            [0, 2*h/3],
            [-side_length/2, -h/3],
            [side_length/2, -h/3]
        ])

        # Use tripolygon for custom polygon extrusion
        triangle = self.shapes['tripolygon'](polygon, thickness, n_z=n_z)

        return [triangle], [[2, 1]]

    # =========================================================================
    # CORE-SHELL STRUCTURES
    # =========================================================================

    def _build_core_shell_sphere(self) -> Tuple[List[Any], List[List[int]]]:
        """Build a core-shell sphere."""
        core_diameter = self.config.get('core_diameter', 40)
        shell_thickness = self.config.get('shell_thickness', 10)
        n = self._get_mesh_density(144)

        outer_diameter = core_diameter + 2 * shell_thickness

        core = self.shapes['trisphere'](n, core_diameter)
        shell = self.shapes['trisphere'](int(n * 1.2), outer_diameter)

        # materials: [shell, core] -> indices: [2=shell, 3=core]
        # core: inside=core_material(3), outside=shell_material(2)
        # shell: inside=shell_material(2), outside=medium(1)
        return [core, shell], [[3, 2], [2, 1]]

    def _build_core_shell_cube(self) -> Tuple[List[Any], List[List[int]]]:
        """Build a core-shell cube."""
        core_size = self.config.get('core_size', 30)
        shell_thickness = self.config.get('shell_thickness', 5)
        rounding = self._get_rounding(0.25)
        n = self._get_mesh_density(12)

        outer_size = core_size + 2 * shell_thickness

        core = self.shapes['tricube'](n, core_size, edge_rounding=rounding)
        shell = self.shapes['tricube'](n, outer_size, edge_rounding=rounding)

        return [core, shell], [[3, 2], [2, 1]]

    def _build_core_shell_rod(self) -> Tuple[List[Any], List[List[int]]]:
        """Build a core-shell nanorod."""
        core_diameter = self.config.get('core_diameter', 15)
        shell_thickness = self.config.get('shell_thickness', 5)
        height = self.config.get('height', 80)

        outer_diameter = core_diameter + 2 * shell_thickness

        if 'rod_mesh' in self.config:
            nphi, ntheta, nz = self.config['rod_mesh']
            core = self.shapes['trirod'](core_diameter, height, n=(nphi, ntheta, nz))
            shell = self.shapes['trirod'](outer_diameter, height, n=(nphi, ntheta, nz))
        else:
            # trirod expects n as tuple (nphi, ntheta, nz)
            core = self.shapes['trirod'](core_diameter, height, n=(15, 20, 20))
            shell = self.shapes['trirod'](outer_diameter, height, n=(15, 20, 20))

        return [core, shell], [[3, 2], [2, 1]]

    # =========================================================================
    # SIMPLE DIMERS
    # =========================================================================

    def _build_dimer_sphere(self) -> Tuple[List[Any], List[List[int]]]:
        """Build a dimer of two spheres."""
        diameter = self.config.get('diameter', 50)
        gap = self.config.get('gap', 5)
        n = self._get_mesh_density(144)

        # Calculate center-to-center distance
        separation = diameter + gap

        sphere1 = self.shapes['trisphere'](n, diameter)
        sphere1 = sphere1.shift([-separation/2, 0, 0])

        sphere2 = self.shapes['trisphere'](n, diameter)
        sphere2 = sphere2.shift([separation/2, 0, 0])

        return [sphere1, sphere2], [[2, 1], [2, 1]]

    def _build_dimer_cube(self) -> Tuple[List[Any], List[List[int]]]:
        """Build a dimer of two cubes."""
        size = self.config.get('size', 40)
        gap = self.config.get('gap', 10)
        rounding = self._get_rounding(0.25)
        n = self._get_mesh_density(12)

        # Center-to-center distance
        separation = size + gap

        cube1 = self.shapes['tricube'](n, size, edge_rounding=rounding)
        cube1 = cube1.shift([-separation/2, 0, 0])

        cube2 = self.shapes['tricube'](n, size, edge_rounding=rounding)
        cube2 = cube2.shift([separation/2, 0, 0])

        return [cube1, cube2], [[2, 1], [2, 1]]

    def _build_dimer_core_shell_cube(self) -> Tuple[List[Any], List[List[int]]]:
        """Build a dimer of two core-shell cubes."""
        core_size = self.config.get('core_size', 20)
        shell_thickness = self.config.get('shell_thickness', 5)
        gap = self.config.get('gap', 10)
        rounding = self._get_rounding(0.25)
        n = self._get_mesh_density(12)

        outer_size = core_size + 2 * shell_thickness
        separation = outer_size + gap

        # Particle 1 (left)
        core1 = self.shapes['tricube'](n, core_size, edge_rounding=rounding)
        core1 = core1.shift([-separation/2, 0, 0])

        shell1 = self.shapes['tricube'](n, outer_size, edge_rounding=rounding)
        shell1 = shell1.shift([-separation/2, 0, 0])

        # Particle 2 (right)
        core2 = self.shapes['tricube'](n, core_size, edge_rounding=rounding)
        core2 = core2.shift([separation/2, 0, 0])

        shell2 = self.shapes['tricube'](n, outer_size, edge_rounding=rounding)
        shell2 = shell2.shift([separation/2, 0, 0])

        # materials: [shell, core] -> 2=shell, 3=core
        return [core1, shell1, core2, shell2], [[3, 2], [2, 1], [3, 2], [2, 1]]

    # =========================================================================
    # ADVANCED DIMER CUBE
    # =========================================================================

    def _build_advanced_dimer_cube(self) -> Tuple[List[Any], List[List[int]]]:
        """
        Build an advanced dimer cube with multi-shell layers and transformations.

        Features:
        - Multiple shell layers
        - Per-layer rounding control
        - Gap, offset, tilt, rotation control
        """
        core_size = self.config.get('core_size', 30)
        shell_layers = self.config.get('shell_layers', [5])
        n = self._get_mesh_density(12)

        # Get rounding - can be single value or per-layer list
        if 'roundings' in self.config:
            roundings = self.config['roundings']
        else:
            rounding = self._get_rounding(0.25)
            roundings = [rounding] * (1 + len(shell_layers))  # core + shells

        # Transformation parameters
        gap = self.config.get('gap', 5)
        offset = self.config.get('offset', [0, 0, 0])
        tilt_angle = self.config.get('tilt_angle', 0)
        tilt_axis = self.config.get('tilt_axis', [0, 1, 0])
        rotation_angle = self.config.get('rotation_angle', 0)

        # Calculate layer sizes (from inside out)
        sizes = [core_size]
        current_size = core_size
        for shell_thickness in shell_layers:
            current_size += 2 * shell_thickness
            sizes.append(current_size)

        outer_size = sizes[-1]

        # Build particle 1 (left)
        particles1 = []
        for i, (size, rnd) in enumerate(zip(sizes, roundings)):
            particle = self.shapes['tricube'](n, size, edge_rounding=rnd)
            particles1.append(particle)

        # Shift particle 1 to the left
        shift_x = -(outer_size + gap) / 2
        particles1 = [p.shift([shift_x, 0, 0]) for p in particles1]

        # Build particle 2 (right) with transformations
        particles2 = []
        for i, (size, rnd) in enumerate(zip(sizes, roundings)):
            particle = self.shapes['tricube'](n, size, edge_rounding=rnd)
            particles2.append(particle)

        # Apply transformations to particle 2:
        # 1. Rotation around z-axis
        if rotation_angle != 0:
            particles2 = [p.rotate(np.deg2rad(rotation_angle), [0, 0, 1]) for p in particles2]

        # 2. Tilt around custom axis
        if tilt_angle != 0:
            particles2 = [p.rotate(np.deg2rad(tilt_angle), tilt_axis) for p in particles2]

        # 3. Shift to gap position
        shift_x = (outer_size + gap) / 2
        particles2 = [p.shift([shift_x, 0, 0]) for p in particles2]

        # 4. Apply offset
        if any(o != 0 for o in offset):
            particles2 = [p.shift(offset) for p in particles2]

        # Combine all particles
        all_particles = particles1 + particles2

        # Build inout list
        # materials: [core, shell1, shell2, ...] -> indices: 2=core, 3=shell1, 4=shell2, ...
        n_layers = len(sizes)
        inout1 = []
        for i in range(n_layers):
            if i == 0:  # innermost (core)
                inout1.append([2, 3 if n_layers > 1 else 1])  # inside=core, outside=shell1 or medium
            elif i == n_layers - 1:  # outermost
                inout1.append([i + 2, 1])  # inside=this_layer, outside=medium
            else:  # middle shells
                inout1.append([i + 2, i + 3])  # inside=this_layer, outside=next_layer

        # Duplicate for particle 2
        inout = inout1 + inout1

        return all_particles, inout

    # =========================================================================
    # SPHERE CLUSTER AGGREGATE
    # =========================================================================

    def _build_sphere_cluster(self) -> Tuple[List[Any], List[List[int]]]:
        """
        Build a compact sphere cluster (1-7 spheres).

        Configurations:
        - N=1: Single sphere
        - N=2: Dimer (horizontal)
        - N=3: Triangle
        - N=4: Square (2x2)
        - N=5: Pentagon arrangement
        - N=6: Hexagonal
        - N=7: Extended hexagonal
        """
        n_spheres = self.config.get('n_spheres', 3)
        diameter = self.config.get('diameter', 50)
        gap = self.config.get('gap', -0.1)  # Negative for contact
        mesh_n = self._get_mesh_density(144)

        # Center-to-center distance
        spacing = diameter + gap

        # Calculate sphere positions based on configuration
        positions = self._get_cluster_positions(n_spheres, spacing)

        particles = []
        inout = []

        for pos in positions:
            sphere = self.shapes['trisphere'](mesh_n, diameter)
            sphere = sphere.shift(pos)
            particles.append(sphere)
            inout.append([2, 1])

        return particles, inout

    def _get_cluster_positions(self, n: int, spacing: float) -> List[List[float]]:
        """Get sphere positions for a cluster of n spheres."""
        if n == 1:
            return [[0, 0, 0]]
        elif n == 2:
            return [[-spacing/2, 0, 0], [spacing/2, 0, 0]]
        elif n == 3:
            # Triangle: 2 at bottom, 1 at top
            h = spacing * np.sqrt(3) / 2
            return [
                [-spacing/2, -h/3, 0],
                [spacing/2, -h/3, 0],
                [0, 2*h/3, 0]
            ]
        elif n == 4:
            # Square 2x2
            return [
                [-spacing/2, -spacing/2, 0],
                [spacing/2, -spacing/2, 0],
                [-spacing/2, spacing/2, 0],
                [spacing/2, spacing/2, 0]
            ]
        elif n == 5:
            # Pentagon: 3 bottom, 2 top
            h = spacing * np.sqrt(3) / 2
            return [
                [-spacing, 0, 0],
                [0, 0, 0],
                [spacing, 0, 0],
                [-spacing/2, h, 0],
                [spacing/2, h, 0]
            ]
        elif n == 6:
            # Hexagonal: 3 bottom, 3 top
            h = spacing * np.sqrt(3) / 2
            return [
                [-spacing, 0, 0],
                [0, 0, 0],
                [spacing, 0, 0],
                [-spacing/2, h, 0],
                [spacing/2, h, 0],
                [0, -h, 0]
            ]
        elif n == 7:
            # Extended hexagonal: 4 bottom, 3 top
            h = spacing * np.sqrt(3) / 2
            return [
                [-1.5*spacing, -h/2, 0],
                [-0.5*spacing, -h/2, 0],
                [0.5*spacing, -h/2, 0],
                [1.5*spacing, -h/2, 0],
                [-spacing, h/2, 0],
                [0, h/2, 0],
                [spacing, h/2, 0]
            ]
        else:
            raise ValueError(f"n_spheres must be 1-7, got {n}")

    # =========================================================================
    # DDA SHAPE FILE
    # =========================================================================

    def _build_from_shape(self) -> Tuple[List[Any], List[List[int]]]:
        """
        Build particle from DDA .shape file.

        This reads a DDA format file and converts it to BEM mesh.
        """
        shape_file = self.config.get('shape_file')
        voxel_size = self.config.get('voxel_size', 2.0)
        method = self.config.get('voxel_method', 'surface')

        if not shape_file or not os.path.exists(shape_file):
            raise FileNotFoundError(f"Shape file not found: {shape_file}")

        # Read .shape file
        voxels, mat_indices = self._read_shape_file(shape_file)

        # Convert voxels to mesh
        if method == 'surface':
            particles, inout = self._voxels_to_surface_mesh(voxels, mat_indices, voxel_size)
        else:
            particles, inout = self._voxels_to_cube_mesh(voxels, mat_indices, voxel_size)

        return particles, inout

    def _read_shape_file(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read DDA .shape file and return voxel positions and material indices."""
        voxels = []
        mat_indices = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Skip header lines (typically first 2 lines)
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 4:
                # Format: x y z mat_type
                x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
                mat = int(parts[3]) if len(parts) > 3 else 1
                voxels.append([x, y, z])
                mat_indices.append(mat)

        return np.array(voxels), np.array(mat_indices)

    def _voxels_to_surface_mesh(self, voxels: np.ndarray, mat_indices: np.ndarray,
                                 voxel_size: float) -> Tuple[List[Any], List[List[int]]]:
        """Convert voxels to surface mesh (faster but less accurate)."""
        from scipy.spatial import ConvexHull

        # Group voxels by material
        unique_mats = np.unique(mat_indices)
        particles = []
        inout = []

        for mat in unique_mats:
            mask = mat_indices == mat
            mat_voxels = voxels[mask] * voxel_size

            # Create convex hull
            if len(mat_voxels) >= 4:
                hull = ConvexHull(mat_voxels)

                # Create Particle from hull
                vertices = mat_voxels[hull.vertices]
                faces = hull.simplices

                particle = self.Particle(vertices, faces)
                particles.append(particle)
                inout.append([int(mat) + 1, 1])  # mat+1 inside, medium outside

        return particles, inout

    def _voxels_to_cube_mesh(self, voxels: np.ndarray, mat_indices: np.ndarray,
                             voxel_size: float) -> Tuple[List[Any], List[List[int]]]:
        """Convert voxels to cube mesh (more accurate but slower)."""
        # Group by material
        unique_mats = np.unique(mat_indices)
        particles = []
        inout = []

        for mat in unique_mats:
            mask = mat_indices == mat
            mat_voxels = voxels[mask] * voxel_size

            # Create small cube at each voxel position
            all_verts = []
            all_faces = []

            for i, voxel in enumerate(mat_voxels):
                cube = self.shapes['tricube'](4, voxel_size, edge_rounding=0.0)
                cube = cube.shift(voxel.tolist())

                # Offset face indices for combined mesh
                face_offset = len(all_verts)
                all_verts.extend(cube.verts)
                all_faces.extend(cube.faces + face_offset)

            if all_verts:
                combined = self.Particle(np.array(all_verts), np.array(all_faces))
                particles.append(combined)
                inout.append([int(mat) + 1, 1])

        return particles, inout

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_particle_bounds(self, particles: List[Any]) -> Dict[str, float]:
        """Calculate bounding box for a list of particles."""
        all_verts = []
        for p in particles:
            all_verts.append(p.verts)

        all_verts = np.vstack(all_verts)

        return {
            'x_min': float(np.min(all_verts[:, 0])),
            'x_max': float(np.max(all_verts[:, 0])),
            'y_min': float(np.min(all_verts[:, 1])),
            'y_max': float(np.max(all_verts[:, 1])),
            'z_min': float(np.min(all_verts[:, 2])),
            'z_max': float(np.max(all_verts[:, 2])),
        }

    def get_structure_info(self) -> Dict[str, Any]:
        """Get summary information about the structure configuration."""
        return {
            'structure_type': self.config.get('structure', 'unknown'),
            'structure_name': self.config.get('structure_name', 'unnamed'),
            'materials': self.config.get('materials', []),
            'medium': self.config.get('medium', 'air'),
        }

