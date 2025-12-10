"""
Geometry Cross-Section Calculator for field plot overlays.

Calculates 2D cross-sections of 3D nanoparticle structures for
overlay on field enhancement plots.
"""

import numpy as np
from typing import Dict, List, Any, Optional


class GeometryCrossSection:
    """
    Calculates 2D cross-sections of 3D structures.

    Supports all structure types:
    - Spheres, cubes, rods, ellipsoids
    - Core-shell structures
    - Dimers and clusters
    - Advanced dimer configurations
    """

    def __init__(self, structure_config: Dict[str, Any]):
        """
        Initialize the cross-section calculator.

        Args:
            structure_config: Dictionary with structure parameters
        """
        self.config = structure_config

    def calculate(self, plane: str = 'xz', position: float = 0.0) -> List[Dict[str, Any]]:
        """
        Calculate cross-section elements for a given plane.

        Args:
            plane: 'xy', 'xz', or 'yz'
            position: Position of the plane along the perpendicular axis

        Returns:
            List of geometry element dictionaries for visualization
        """
        structure_type = self.config.get('structure', 'sphere')

        method_map = {
            'sphere': self._sphere_cross_section,
            'cube': self._cube_cross_section,
            'rod': self._rod_cross_section,
            'ellipsoid': self._ellipsoid_cross_section,
            'core_shell_sphere': self._core_shell_sphere_cross_section,
            'core_shell_cube': self._core_shell_cube_cross_section,
            'dimer_sphere': self._dimer_sphere_cross_section,
            'dimer_cube': self._dimer_cube_cross_section,
            'advanced_dimer_cube': self._advanced_dimer_cube_cross_section,
            'sphere_cluster_aggregate': self._sphere_cluster_cross_section,
        }

        if structure_type in method_map:
            return method_map[structure_type](plane, position)
        else:
            return []

    def _sphere_cross_section(self, plane: str, position: float) -> List[Dict]:
        """Cross-section of a single sphere."""
        diameter = self.config.get('diameter', 50)
        radius = diameter / 2

        # Check if plane intersects sphere
        if abs(position) > radius:
            return []

        # Calculate intersection circle radius
        circle_radius = np.sqrt(radius**2 - position**2)

        return [{
            'type': 'circle',
            'x': 0,
            'y': 0,
            'radius': circle_radius,
            'color': 'white',
            'linewidth': 2
        }]

    def _cube_cross_section(self, plane: str, position: float) -> List[Dict]:
        """Cross-section of a single cube."""
        size = self.config.get('size', 40)
        half_size = size / 2

        if abs(position) > half_size:
            return []

        return [{
            'type': 'rectangle',
            'x': 0,
            'y': 0,
            'width': size,
            'height': size,
            'color': 'white',
            'linewidth': 2
        }]

    def _rod_cross_section(self, plane: str, position: float) -> List[Dict]:
        """Cross-section of a nanorod."""
        diameter = self.config.get('diameter', 20)
        height = self.config.get('height', 80)
        radius = diameter / 2

        elements = []

        if plane == 'xz':
            # Vertical cross-section shows the rod shape
            if abs(position) <= radius:
                # Rectangle for main body
                elements.append({
                    'type': 'rectangle',
                    'x': 0,
                    'y': 0,
                    'width': diameter,
                    'height': height,
                    'color': 'white',
                    'linewidth': 2
                })
        elif plane == 'xy':
            # Horizontal cross-section shows circle
            if abs(position) <= height / 2:
                elements.append({
                    'type': 'circle',
                    'x': 0,
                    'y': 0,
                    'radius': radius,
                    'color': 'white',
                    'linewidth': 2
                })

        return elements

    def _ellipsoid_cross_section(self, plane: str, position: float) -> List[Dict]:
        """Cross-section of an ellipsoid."""
        axes = self.config.get('axes', [20, 30, 40])
        a, b, c = axes  # semi-axes along x, y, z

        elements = []

        if plane == 'xz':  # y = position
            if abs(position) <= b:
                # Ellipse in xz plane
                scale_y = np.sqrt(1 - (position/b)**2)
                elements.append({
                    'type': 'ellipse',
                    'x': 0,
                    'y': 0,
                    'rx': a * scale_y,
                    'ry': c * scale_y,
                    'color': 'white',
                    'linewidth': 2
                })
        elif plane == 'xy':  # z = position
            if abs(position) <= c:
                scale_z = np.sqrt(1 - (position/c)**2)
                elements.append({
                    'type': 'ellipse',
                    'x': 0,
                    'y': 0,
                    'rx': a * scale_z,
                    'ry': b * scale_z,
                    'color': 'white',
                    'linewidth': 2
                })

        return elements

    def _core_shell_sphere_cross_section(self, plane: str, position: float) -> List[Dict]:
        """Cross-section of a core-shell sphere."""
        core_diameter = self.config.get('core_diameter', 40)
        shell_thickness = self.config.get('shell_thickness', 10)

        core_radius = core_diameter / 2
        shell_radius = core_radius + shell_thickness

        elements = []

        # Shell (outer)
        if abs(position) <= shell_radius:
            circle_radius = np.sqrt(shell_radius**2 - position**2)
            elements.append({
                'type': 'circle',
                'x': 0,
                'y': 0,
                'radius': circle_radius,
                'color': 'white',
                'linewidth': 2
            })

        # Core (inner)
        if abs(position) <= core_radius:
            circle_radius = np.sqrt(core_radius**2 - position**2)
            elements.append({
                'type': 'circle',
                'x': 0,
                'y': 0,
                'radius': circle_radius,
                'color': 'cyan',
                'linewidth': 1.5,
                'linestyle': '--'
            })

        return elements

    def _core_shell_cube_cross_section(self, plane: str, position: float) -> List[Dict]:
        """Cross-section of a core-shell cube."""
        core_size = self.config.get('core_size', 30)
        shell_thickness = self.config.get('shell_thickness', 5)

        outer_size = core_size + 2 * shell_thickness

        elements = []

        if abs(position) <= outer_size / 2:
            # Shell (outer)
            elements.append({
                'type': 'rectangle',
                'x': 0,
                'y': 0,
                'width': outer_size,
                'height': outer_size,
                'color': 'white',
                'linewidth': 2
            })

        if abs(position) <= core_size / 2:
            # Core (inner)
            elements.append({
                'type': 'rectangle',
                'x': 0,
                'y': 0,
                'width': core_size,
                'height': core_size,
                'color': 'cyan',
                'linewidth': 1.5,
                'linestyle': '--'
            })

        return elements

    def _dimer_sphere_cross_section(self, plane: str, position: float) -> List[Dict]:
        """Cross-section of a sphere dimer."""
        diameter = self.config.get('diameter', 50)
        gap = self.config.get('gap', 5)
        radius = diameter / 2
        separation = diameter + gap

        elements = []

        # Left sphere
        x_left = -separation / 2
        if abs(position) <= radius:
            circle_radius = np.sqrt(radius**2 - position**2)
            elements.append({
                'type': 'circle',
                'x': x_left,
                'y': 0,
                'radius': circle_radius,
                'color': 'white',
                'linewidth': 2
            })

        # Right sphere
        x_right = separation / 2
        if abs(position) <= radius:
            circle_radius = np.sqrt(radius**2 - position**2)
            elements.append({
                'type': 'circle',
                'x': x_right,
                'y': 0,
                'radius': circle_radius,
                'color': 'white',
                'linewidth': 2
            })

        return elements

    def _dimer_cube_cross_section(self, plane: str, position: float) -> List[Dict]:
        """Cross-section of a cube dimer."""
        size = self.config.get('size', 40)
        gap = self.config.get('gap', 10)
        separation = size + gap

        elements = []

        if abs(position) <= size / 2:
            # Left cube
            elements.append({
                'type': 'rectangle',
                'x': -separation / 2,
                'y': 0,
                'width': size,
                'height': size,
                'color': 'white',
                'linewidth': 2
            })

            # Right cube
            elements.append({
                'type': 'rectangle',
                'x': separation / 2,
                'y': 0,
                'width': size,
                'height': size,
                'color': 'white',
                'linewidth': 2
            })

        return elements

    def _advanced_dimer_cube_cross_section(self, plane: str, position: float) -> List[Dict]:
        """Cross-section of an advanced dimer cube with shells."""
        core_size = self.config.get('core_size', 30)
        shell_layers = self.config.get('shell_layers', [5])
        gap = self.config.get('gap', 5)

        # Calculate total size
        total_size = core_size + 2 * sum(shell_layers)
        separation = total_size + gap

        elements = []

        if abs(position) <= total_size / 2:
            for x_center in [-separation / 2, separation / 2]:
                # Draw from outside in
                current_size = total_size
                colors = ['white'] + ['cyan', 'yellow', 'magenta'][:len(shell_layers)]

                # Outer shell
                elements.append({
                    'type': 'rectangle',
                    'x': x_center,
                    'y': 0,
                    'width': current_size,
                    'height': current_size,
                    'color': colors[0],
                    'linewidth': 2
                })

                # Inner shells
                for i, thickness in enumerate(reversed(shell_layers)):
                    current_size -= 2 * thickness
                    if abs(position) <= current_size / 2:
                        elements.append({
                            'type': 'rectangle',
                            'x': x_center,
                            'y': 0,
                            'width': current_size,
                            'height': current_size,
                            'color': colors[min(i + 1, len(colors) - 1)],
                            'linewidth': 1.5,
                            'linestyle': '--'
                        })

        return elements

    def _sphere_cluster_cross_section(self, plane: str, position: float) -> List[Dict]:
        """Cross-section of a sphere cluster."""
        n_spheres = self.config.get('n_spheres', 3)
        diameter = self.config.get('diameter', 50)
        gap = self.config.get('gap', -0.1)
        radius = diameter / 2
        spacing = diameter + gap

        # Get sphere positions
        positions = self._get_cluster_positions(n_spheres, spacing)

        elements = []

        for pos in positions:
            # For xz plane, check if sphere intersects y=position
            dist_to_plane = abs(position - pos[1]) if plane == 'xz' else abs(position - pos[2])

            if dist_to_plane <= radius:
                circle_radius = np.sqrt(radius**2 - dist_to_plane**2)

                if plane == 'xz':
                    x, y = pos[0], pos[2]
                else:  # xy
                    x, y = pos[0], pos[1]

                elements.append({
                    'type': 'circle',
                    'x': x,
                    'y': y,
                    'radius': circle_radius,
                    'color': 'white',
                    'linewidth': 2
                })

        return elements

    def _get_cluster_positions(self, n: int, spacing: float) -> List[List[float]]:
        """Get sphere positions for cluster configurations."""
        if n == 1:
            return [[0, 0, 0]]
        elif n == 2:
            return [[-spacing/2, 0, 0], [spacing/2, 0, 0]]
        elif n == 3:
            h = spacing * np.sqrt(3) / 2
            return [
                [-spacing/2, -h/3, 0],
                [spacing/2, -h/3, 0],
                [0, 2*h/3, 0]
            ]
        elif n == 4:
            return [
                [-spacing/2, -spacing/2, 0],
                [spacing/2, -spacing/2, 0],
                [-spacing/2, spacing/2, 0],
                [spacing/2, spacing/2, 0]
            ]
        elif n >= 5:
            # Simplified for 5-7 spheres
            h = spacing * np.sqrt(3) / 2
            positions = [
                [-spacing, 0, 0],
                [0, 0, 0],
                [spacing, 0, 0],
                [-spacing/2, h, 0],
                [spacing/2, h, 0]
            ]
            if n >= 6:
                positions.append([0, -h, 0])
            if n >= 7:
                positions.append([0, 2*h, 0])
            return positions[:n]
        return [[0, 0, 0]]
