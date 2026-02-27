import os
import sys
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np


class GeometryCrossSection(object):

    def __init__(self,
            config: Dict[str, Any],
            verbose: bool = False) -> None:

        self.config = config
        self.verbose = verbose
        self.structure_type = config.get('structure', 'sphere')

        if self.verbose:
            print('[GeometryCrossSection] Structure type: {}'.format(self.structure_type))

    def get_cross_section(self,
            z_plane: float) -> List[Dict[str, Any]]:

        if self.verbose:
            print('\n[GeometryCrossSection] Calculating cross-section at z = {:.2f} nm'.format(z_plane))

        structure_handlers = {
            'sphere': self._sphere_cross_section,
            'core_shell': self._core_shell_cross_section,
            'core_shell_sphere': self._core_shell_cross_section,
            'core_shell_cube': self._cube_cross_section,
            'core_shell_rod': self._rod_cross_section,
            'cube': self._cube_cross_section,
            'dimer': self._dimer_cross_section,
            'dimer_sphere': self._dimer_cross_section,
            'dimer_cube': self._dimer_cube_cross_section,
            'dimer_core_shell_cube': self._dimer_cube_cross_section,
            'advanced_dimer_cube': self._dimer_cube_cross_section,
            'advanced_monomer_cube': self._cube_cross_section,
            'connected_dimer_cube': self._dimer_cube_cross_section,
            'rod': self._rod_cross_section,
            'ellipsoid': self._ellipsoid_cross_section,
            'sphere_cluster': self._sphere_cluster_cross_section,
            'sphere_cluster_aggregate': self._sphere_cluster_cross_section,
            'triangle': self._triangle_cross_section,
            'from_shape': self._from_shape_cross_section,
        }

        handler = structure_handlers.get(self.structure_type)

        if handler is None:
            if self.verbose:
                print('  Warning: Unknown structure type <{}>'.format(self.structure_type))
                print('  Supported structures: {}'.format(list(structure_handlers.keys())))
            return []

        return handler(z_plane)

    def _sphere_cross_section(self,
            z_plane: float) -> List[Dict[str, Any]]:

        radius = self.config.get('radius', 10.0)
        center = self.config.get('center', [0, 0, 0])

        z_center = center[2]
        z_diff = z_plane - z_center

        if abs(z_diff) > radius:
            if self.verbose:
                print('  No intersection: |z_diff| = {:.2f} > radius = {:.2f}'.format(abs(z_diff), radius))
            return []

        r_cross = np.sqrt(radius**2 - z_diff**2)

        if self.verbose:
            print('  Sphere intersection: r_cross = {:.2f} nm'.format(r_cross))

        return [{
            'type': 'circle',
            'center': [center[0], center[1]],
            'radius': r_cross,
            'layer': 0,
            'label': 'Sphere'
        }]

    def _core_shell_cross_section(self,
            z_plane: float) -> List[Dict[str, Any]]:

        radii = self.config.get('radii', [10.0])
        center = self.config.get('center', [0, 0, 0])

        if not isinstance(radii, list):
            radii = [radii]

        z_center = center[2]
        z_diff = z_plane - z_center

        sections = []

        for i, radius in enumerate(sorted(radii, reverse = True)):
            if abs(z_diff) > radius:
                continue

            r_cross = np.sqrt(radius**2 - z_diff**2)

            if len(radii) == 1:
                label = 'Sphere'
            elif i == len(radii) - 1:
                label = 'Core'
            else:
                shell_num = len(radii) - i - 1
                label = 'Shell {}'.format(shell_num)

            sections.append({
                'type': 'circle',
                'center': [center[0], center[1]],
                'radius': r_cross,
                'layer': len(radii) - i - 1,
                'label': label
            })

            if self.verbose:
                print('  Layer {} ({}): R = {:.2f} nm, r_cross = {:.2f} nm'.format(i, label, radius, r_cross))

        return sections[::-1]

    def _cube_cross_section(self,
            z_plane: float) -> List[Dict[str, Any]]:

        side_length = self.config.get('side_length', 20.0)
        center = self.config.get('center', [0, 0, 0])

        half_size = side_length / 2
        x_min = center[0] - half_size
        x_max = center[0] + half_size
        y_min = center[1] - half_size
        y_max = center[1] + half_size
        z_min = center[2] - half_size
        z_max = center[2] + half_size

        if z_plane < z_min or z_plane > z_max:
            if self.verbose:
                print('  No intersection: z_plane = {:.2f} outside [{:.2f}, {:.2f}]'.format(
                    z_plane, z_min, z_max))
            return []

        if self.verbose:
            print('  Cube intersection: square from ({:.1f}, {:.1f}) to ({:.1f}, {:.1f})'.format(
                x_min, y_min, x_max, y_max))

        return [{
            'type': 'rectangle',
            'bounds': [x_min, x_max, y_min, y_max],
            'layer': 0,
            'label': 'Cube'
        }]

    def _dimer_cross_section(self,
            z_plane: float) -> List[Dict[str, Any]]:

        radius = self.config.get('radius', 10.0)
        gap = self.config.get('gap', 2.0)
        center = self.config.get('center', [0, 0, 0])
        dimer_axis = self.config.get('dimer_axis', 'x')

        offset = radius + gap / 2

        if dimer_axis == 'x':
            pos1 = [center[0] - offset, center[1], center[2]]
            pos2 = [center[0] + offset, center[1], center[2]]
        elif dimer_axis == 'y':
            pos1 = [center[0], center[1] - offset, center[2]]
            pos2 = [center[0], center[1] + offset, center[2]]
        elif dimer_axis == 'z':
            pos1 = [center[0], center[1], center[2] - offset]
            pos2 = [center[0], center[1], center[2] + offset]
        else:
            if self.verbose:
                print('  Warning: Unknown dimer_axis <{}>'.format(dimer_axis))
            return []

        sections = []

        z_diff1 = z_plane - pos1[2]
        if abs(z_diff1) <= radius:
            r_cross1 = np.sqrt(radius**2 - z_diff1**2)
            sections.append({
                'type': 'circle',
                'center': [pos1[0], pos1[1]],
                'radius': r_cross1,
                'layer': 0,
                'label': 'Sphere 1'
            })
            if self.verbose:
                print('  Sphere 1: center = ({:.1f}, {:.1f}), r_cross = {:.2f} nm'.format(
                    pos1[0], pos1[1], r_cross1))

        z_diff2 = z_plane - pos2[2]
        if abs(z_diff2) <= radius:
            r_cross2 = np.sqrt(radius**2 - z_diff2**2)
            sections.append({
                'type': 'circle',
                'center': [pos2[0], pos2[1]],
                'radius': r_cross2,
                'layer': 1,
                'label': 'Sphere 2'
            })
            if self.verbose:
                print('  Sphere 2: center = ({:.1f}, {:.1f}), r_cross = {:.2f} nm'.format(
                    pos2[0], pos2[1], r_cross2))

        return sections

    def _dimer_cube_cross_section(self,
            z_plane: float) -> List[Dict[str, Any]]:

        side_length = self.config.get('side_length', 20.0)
        gap = self.config.get('gap', 2.0)
        center = self.config.get('center', [0, 0, 0])
        dimer_axis = self.config.get('dimer_axis', 'x')

        offset = side_length / 2 + gap / 2

        if dimer_axis == 'x':
            pos1 = [center[0] - offset, center[1], center[2]]
            pos2 = [center[0] + offset, center[1], center[2]]
        elif dimer_axis == 'y':
            pos1 = [center[0], center[1] - offset, center[2]]
            pos2 = [center[0], center[1] + offset, center[2]]
        elif dimer_axis == 'z':
            pos1 = [center[0], center[1], center[2] - offset]
            pos2 = [center[0], center[1], center[2] + offset]
        else:
            if self.verbose:
                print('  Warning: Unknown dimer_axis <{}>'.format(dimer_axis))
            return []

        half_size = side_length / 2
        sections = []

        z_min1 = pos1[2] - half_size
        z_max1 = pos1[2] + half_size

        if z_min1 <= z_plane <= z_max1:
            x_min1 = pos1[0] - half_size
            x_max1 = pos1[0] + half_size
            y_min1 = pos1[1] - half_size
            y_max1 = pos1[1] + half_size

            sections.append({
                'type': 'rectangle',
                'bounds': [x_min1, x_max1, y_min1, y_max1],
                'layer': 0,
                'label': 'Cube 1'
            })
            if self.verbose:
                print('  Cube 1: bounds = ({:.1f}, {:.1f}) to ({:.1f}, {:.1f})'.format(
                    x_min1, y_min1, x_max1, y_max1))

        z_min2 = pos2[2] - half_size
        z_max2 = pos2[2] + half_size

        if z_min2 <= z_plane <= z_max2:
            x_min2 = pos2[0] - half_size
            x_max2 = pos2[0] + half_size
            y_min2 = pos2[1] - half_size
            y_max2 = pos2[1] + half_size

            sections.append({
                'type': 'rectangle',
                'bounds': [x_min2, x_max2, y_min2, y_max2],
                'layer': 1,
                'label': 'Cube 2'
            })
            if self.verbose:
                print('  Cube 2: bounds = ({:.1f}, {:.1f}) to ({:.1f}, {:.1f})'.format(
                    x_min2, y_min2, x_max2, y_max2))

        return sections

    def _rod_cross_section(self,
            z_plane: float) -> List[Dict[str, Any]]:

        radius = self.config.get('radius', 5.0)
        length = self.config.get('length', 40.0)
        center = self.config.get('center', [0, 0, 0])
        axis = self.config.get('axis', 'z')

        if axis == 'z':
            z_min = center[2] - length / 2
            z_max = center[2] + length / 2

            if z_plane < z_min or z_plane > z_max:
                if self.verbose:
                    print('  No intersection: z_plane outside rod bounds')
                return []

            if self.verbose:
                print('  Rod (z-axis): circular cross-section with r = {:.2f} nm'.format(radius))

            return [{
                'type': 'circle',
                'center': [center[0], center[1]],
                'radius': radius,
                'layer': 0,
                'label': 'Rod'
            }]

        elif axis == 'x':
            z_center = center[2]
            if abs(z_plane - z_center) > radius:
                if self.verbose:
                    print('  No intersection: |z_diff| > radius')
                return []

            x_min = center[0] - length / 2
            x_max = center[0] + length / 2

            z_diff = z_plane - z_center
            y_half = np.sqrt(radius**2 - z_diff**2)
            y_min = center[1] - y_half
            y_max = center[1] + y_half

            if self.verbose:
                print('  Rod (x-axis): rectangular cross-section')

            return [{
                'type': 'rectangle',
                'bounds': [x_min, x_max, y_min, y_max],
                'layer': 0,
                'label': 'Rod'
            }]

        elif axis == 'y':
            z_center = center[2]
            if abs(z_plane - z_center) > radius:
                if self.verbose:
                    print('  No intersection: |z_diff| > radius')
                return []

            y_min = center[1] - length / 2
            y_max = center[1] + length / 2

            z_diff = z_plane - z_center
            x_half = np.sqrt(radius**2 - z_diff**2)
            x_min = center[0] - x_half
            x_max = center[0] + x_half

            if self.verbose:
                print('  Rod (y-axis): rectangular cross-section')

            return [{
                'type': 'rectangle',
                'bounds': [x_min, x_max, y_min, y_max],
                'layer': 0,
                'label': 'Rod'
            }]

        return []

    def _ellipsoid_cross_section(self,
            z_plane: float) -> List[Dict[str, Any]]:

        semi_axes = self.config.get('semi_axes', [10.0, 10.0, 15.0])
        center = self.config.get('center', [0, 0, 0])

        a, b, c = semi_axes
        z_center = center[2]
        z_diff = z_plane - z_center

        if abs(z_diff) > c:
            if self.verbose:
                print('  No intersection: |z_diff| = {:.2f} > c = {:.2f}'.format(abs(z_diff), c))
            return []

        scale = np.sqrt(1 - (z_diff / c)**2)
        a_cross = a * scale
        b_cross = b * scale

        r_cross = (a_cross + b_cross) / 2

        if self.verbose:
            print('  Ellipsoid intersection: a\' = {:.2f}, b\' = {:.2f}, r_avg = {:.2f} nm'.format(
                a_cross, b_cross, r_cross))
            if abs(a_cross - b_cross) > 0.1:
                print('    Note: Approximating ellipse as circle')

        return [{
            'type': 'circle',
            'center': [center[0], center[1]],
            'radius': r_cross,
            'layer': 0,
            'label': 'Ellipsoid'
        }]

    def _sphere_cluster_cross_section(self,
            z_plane: float) -> List[Dict[str, Any]]:

        num_spheres = self.config.get('n_spheres', 1)
        diameter = self.config.get('diameter', 50.0)
        gap = self.config.get('gap', -0.1)

        radius = diameter / 2
        spacing = diameter + gap

        positions = self._calculate_cluster_positions(num_spheres, spacing)

        sections = []

        for i, pos in enumerate(positions):
            z_diff = z_plane - pos[2]

            if abs(z_diff) > radius:
                continue

            r_cross = np.sqrt(radius**2 - z_diff**2)

            sections.append({
                'type': 'circle',
                'center': [pos[0], pos[1]],
                'radius': r_cross,
                'layer': i,
                'label': 'Sphere {}'.format(i + 1)
            })

            if self.verbose:
                print('  Sphere {}: center = ({:.1f}, {:.1f}, {:.1f}), r_cross = {:.2f} nm'.format(
                    i + 1, pos[0], pos[1], pos[2], r_cross))

        return sections

    def _calculate_cluster_positions(self,
            num_spheres: int,
            spacing: float) -> List[Tuple[float, float, float]]:

        dy_60deg = spacing * 0.866025404  # sin(60 deg) = sqrt(3)/2

        hex_positions = []
        for i in range(6):
            angle = i * 60 * np.pi / 180
            x = spacing * np.cos(angle)
            y = spacing * np.sin(angle)
            hex_positions.append((x, y, 0))

        cluster_positions = {
            1: [(0, 0, 0)],

            2: [(-spacing / 2, 0, 0),
                (spacing / 2, 0, 0)],

            3: [(-spacing / 2, 0, 0),
                (spacing / 2, 0, 0),
                (0, dy_60deg, 0)],

            4: [(0, 0, 0)] + hex_positions[0:3],

            5: [(0, 0, 0)] + hex_positions[0:4],

            6: [(0, 0, 0)] + hex_positions[0:5],

            7: [(0, 0, 0)] + hex_positions[0:6],
        }

        if num_spheres not in cluster_positions:
            raise ValueError('[error] n_spheres must be 1-7, got {}'.format(num_spheres))

        return cluster_positions[num_spheres]

    def _triangle_cross_section(self,
            z_plane: float) -> List[Dict[str, Any]]:

        side_length = self.config.get('side_length', 50.0)
        thickness = self.config.get('thickness', 10.0)
        center = self.config.get('center', [0, 0, 0])

        half_thick = thickness / 2
        z_min = center[2] - half_thick
        z_max = center[2] + half_thick

        if z_plane < z_min or z_plane > z_max:
            if self.verbose:
                print('  No intersection: z_plane outside triangle thickness')
            return []

        area = (np.sqrt(3) / 4) * side_length**2
        r_equiv = np.sqrt(area / np.pi)

        if self.verbose:
            print('  Triangle: approximated as circle with r = {:.2f} nm'.format(r_equiv))

        return [{
            'type': 'circle',
            'center': [center[0], center[1]],
            'radius': r_equiv,
            'layer': 0,
            'label': 'Triangle'
        }]

    def _from_shape_cross_section(self,
            z_plane: float) -> List[Dict[str, Any]]:

        if self.verbose:
            print('  DDA shape file: no cross-section visualization available')
        return []
