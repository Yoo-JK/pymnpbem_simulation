"""
Analytic geometry cross-section calculator.

Given a z-plane coordinate, computes the 2-D cross-section of the nanoparticle
structure for boundary overlays and per-sphere integration masks.

Ported from OLD post_utils/geometry_cross_section.py (GeometryCrossSection class)
into standalone functions.  The class wrapper is retained for backward-compat.

Supported structures
--------------------
sphere, core_shell / core_shell_sphere, core_shell_cube, core_shell_rod,
cube, dimer / dimer_sphere, dimer_cube / dimer_core_shell_cube /
advanced_dimer_cube / connected_dimer_cube, advanced_monomer_cube,
rod, ellipsoid, sphere_cluster / sphere_cluster_aggregate, triangle,
from_shape.

Each cross-section element is a dict with keys:
    'type'   : 'circle' | 'rectangle'
    'center' : [x, y]          (circles)
    'radius' : float           (circles)
    'bounds' : [xmin, xmax, ymin, ymax]  (rectangles)
    'layer'  : int   (0 = outermost / core)
    'label'  : str
"""

import numpy as np


# ---------------------------------------------------------------------------
# Public functional interface
# ---------------------------------------------------------------------------

def geometry_cross_section(config: dict, z_plane: float) -> list:
    """Return 2-D cross-section elements for *config* at *z_plane* (nm).

    Parameters
    ----------
    config : dict
        Simulation configuration containing structure parameters.
    z_plane : float
        Z-coordinate of the cross-section plane (nm).

    Returns
    -------
    list of dict
        Cross-section elements (see module docstring for key layout).
    """
    return GeometryCrossSection(config).get_cross_section(z_plane)


def cluster_positions(n_spheres: int, spacing: float) -> list:
    """Return hexagonal cluster sphere-centre positions matching MATLAB geometry.

    Parameters
    ----------
    n_spheres : int
        Number of spheres (1-7).
    spacing : float
        Centre-to-centre spacing (diameter + gap) in nm.

    Returns
    -------
    list of (x, y, z) tuples
    """
    return GeometryCrossSection._static_cluster_positions(n_spheres, spacing)


# ---------------------------------------------------------------------------
# Class (keeps OLD interface intact)
# ---------------------------------------------------------------------------

class GeometryCrossSection:
    """Compute material boundary cross-sections for visualisation and integration.

    Parameters
    ----------
    config : dict
        Simulation configuration.
    verbose : bool
        Print debug information.
    """

    def __init__(self, config: dict, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.structure_type = config.get('structure', 'sphere')

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get_cross_section(self, z_plane: float) -> list:
        """Return cross-section elements at *z_plane* (nm)."""
        handlers = {
            'sphere':                     self._sphere_cross_section,
            'core_shell':                 self._core_shell_cross_section,
            'core_shell_sphere':          self._core_shell_cross_section,
            'core_shell_cube':            self._cube_cross_section,
            'core_shell_rod':             self._rod_cross_section,
            'cube':                       self._cube_cross_section,
            'dimer':                      self._dimer_cross_section,
            'dimer_sphere':               self._dimer_cross_section,
            'dimer_cube':                 self._dimer_cube_cross_section,
            'dimer_core_shell_cube':      self._dimer_cube_cross_section,
            'advanced_dimer_cube':        self._dimer_cube_cross_section,
            'connected_dimer_cube':       self._dimer_cube_cross_section,
            'advanced_monomer_cube':      self._cube_cross_section,
            'rod':                        self._rod_cross_section,
            'ellipsoid':                  self._ellipsoid_cross_section,
            'sphere_cluster':             self._sphere_cluster_cross_section,
            'sphere_cluster_aggregate':   self._sphere_cluster_cross_section,
            'triangle':                   self._triangle_cross_section,
            'from_shape':                 self._from_shape_cross_section,
        }
        handler = handlers.get(self.structure_type)
        if handler is None:
            return []
        return handler(z_plane)

    # ------------------------------------------------------------------
    # Per-structure handlers
    # ------------------------------------------------------------------

    def _sphere_cross_section(self, z_plane: float) -> list:
        radius = self.config.get('radius', self.config.get('diameter', 20.0) / 2)
        center = self.config.get('center', [0, 0, 0])
        z_diff = z_plane - center[2]
        if abs(z_diff) > radius:
            return []
        r_cross = float(np.sqrt(radius ** 2 - z_diff ** 2))
        return [{'type': 'circle', 'center': [center[0], center[1]],
                 'radius': r_cross, 'layer': 0, 'label': 'Sphere'}]

    def _core_shell_cross_section(self, z_plane: float) -> list:
        radii = self.config.get('radii', [self.config.get('radius', 10.0)])
        center = self.config.get('center', [0, 0, 0])
        if not isinstance(radii, (list, tuple)):
            radii = [radii]

        z_diff = z_plane - center[2]
        sections = []
        for i, radius in enumerate(sorted(radii, reverse=True)):
            if abs(z_diff) > radius:
                continue
            r_cross = float(np.sqrt(radius ** 2 - z_diff ** 2))
            if len(radii) == 1:
                label = 'Sphere'
            elif i == len(radii) - 1:
                label = 'Core'
            else:
                label = 'Shell {}'.format(len(radii) - i - 1)
            sections.append({'type': 'circle', 'center': [center[0], center[1]],
                             'radius': r_cross, 'layer': len(radii) - i - 1, 'label': label})
        return sections[::-1]

    def _cube_cross_section(self, z_plane: float) -> list:
        side = self.config.get('side_length', 20.0)
        center = self.config.get('center', [0, 0, 0])
        h = side / 2
        z_min, z_max = center[2] - h, center[2] + h
        if z_plane < z_min or z_plane > z_max:
            return []
        return [{'type': 'rectangle',
                 'bounds': [center[0] - h, center[0] + h, center[1] - h, center[1] + h],
                 'layer': 0, 'label': 'Cube'}]

    def _dimer_cross_section(self, z_plane: float) -> list:
        radius = self.config.get('radius', self.config.get('diameter', 20.0) / 2)
        gap = self.config.get('gap', 2.0)
        center = self.config.get('center', [0, 0, 0])
        axis = self.config.get('dimer_axis', 'x')
        offset = radius + gap / 2

        if axis == 'x':
            centres = [[center[0] - offset, center[1], center[2]],
                       [center[0] + offset, center[1], center[2]]]
        elif axis == 'y':
            centres = [[center[0], center[1] - offset, center[2]],
                       [center[0], center[1] + offset, center[2]]]
        elif axis == 'z':
            centres = [[center[0], center[1], center[2] - offset],
                       [center[0], center[1], center[2] + offset]]
        else:
            return []

        sections = []
        for i, pos in enumerate(centres):
            z_diff = z_plane - pos[2]
            if abs(z_diff) <= radius:
                r_cross = float(np.sqrt(radius ** 2 - z_diff ** 2))
                sections.append({'type': 'circle', 'center': [pos[0], pos[1]],
                                 'radius': r_cross, 'layer': i,
                                 'label': 'Sphere {}'.format(i + 1)})
        return sections

    def _dimer_cube_cross_section(self, z_plane: float) -> list:
        side = self.config.get('side_length', 20.0)
        gap = self.config.get('gap', 2.0)
        center = self.config.get('center', [0, 0, 0])
        axis = self.config.get('dimer_axis', 'x')
        offset = side / 2 + gap / 2
        h = side / 2

        if axis == 'x':
            centres = [[center[0] - offset, center[1], center[2]],
                       [center[0] + offset, center[1], center[2]]]
        elif axis == 'y':
            centres = [[center[0], center[1] - offset, center[2]],
                       [center[0], center[1] + offset, center[2]]]
        elif axis == 'z':
            centres = [[center[0], center[1], center[2] - offset],
                       [center[0], center[1], center[2] + offset]]
        else:
            return []

        sections = []
        for i, pos in enumerate(centres):
            if pos[2] - h <= z_plane <= pos[2] + h:
                sections.append({'type': 'rectangle',
                                 'bounds': [pos[0] - h, pos[0] + h, pos[1] - h, pos[1] + h],
                                 'layer': i, 'label': 'Cube {}'.format(i + 1)})
        return sections

    def _rod_cross_section(self, z_plane: float) -> list:
        radius = self.config.get('radius', 5.0)
        length = self.config.get('length', 40.0)
        center = self.config.get('center', [0, 0, 0])
        axis = self.config.get('axis', 'z')

        if axis == 'z':
            if not (center[2] - length / 2 <= z_plane <= center[2] + length / 2):
                return []
            return [{'type': 'circle', 'center': [center[0], center[1]],
                     'radius': radius, 'layer': 0, 'label': 'Rod'}]

        elif axis in ('x', 'y'):
            z_diff = z_plane - center[2]
            if abs(z_diff) > radius:
                return []
            y_half = float(np.sqrt(radius ** 2 - z_diff ** 2))
            if axis == 'x':
                bounds = [center[0] - length / 2, center[0] + length / 2,
                          center[1] - y_half, center[1] + y_half]
            else:
                x_half = float(np.sqrt(radius ** 2 - z_diff ** 2))
                bounds = [center[0] - x_half, center[0] + x_half,
                          center[1] - length / 2, center[1] + length / 2]
            return [{'type': 'rectangle', 'bounds': bounds, 'layer': 0, 'label': 'Rod'}]

        return []

    def _ellipsoid_cross_section(self, z_plane: float) -> list:
        semi_axes = self.config.get('semi_axes', [10.0, 10.0, 15.0])
        center = self.config.get('center', [0, 0, 0])
        a, b, c = semi_axes
        z_diff = z_plane - center[2]
        if abs(z_diff) > c:
            return []
        scale = float(np.sqrt(1 - (z_diff / c) ** 2))
        r_cross = (a * scale + b * scale) / 2
        return [{'type': 'circle', 'center': [center[0], center[1]],
                 'radius': r_cross, 'layer': 0, 'label': 'Ellipsoid'}]

    def _sphere_cluster_cross_section(self, z_plane: float) -> list:
        n = int(self.config.get('n_spheres', 1))
        diameter = float(self.config.get('diameter', 50.0))
        gap = float(self.config.get('gap', -0.1))
        radius = diameter / 2
        spacing = diameter + gap
        positions = self._calculate_cluster_positions(n, spacing)

        sections = []
        for i, pos in enumerate(positions):
            z_diff = z_plane - pos[2]
            if abs(z_diff) <= radius:
                r_cross = float(np.sqrt(radius ** 2 - z_diff ** 2))
                sections.append({'type': 'circle', 'center': [pos[0], pos[1]],
                                 'radius': r_cross, 'layer': i,
                                 'label': 'Sphere {}'.format(i + 1)})
        return sections

    def _triangle_cross_section(self, z_plane: float) -> list:
        side = self.config.get('side_length', 50.0)
        thickness = self.config.get('thickness', 10.0)
        center = self.config.get('center', [0, 0, 0])
        h = thickness / 2
        if not (center[2] - h <= z_plane <= center[2] + h):
            return []
        area = (np.sqrt(3) / 4) * side ** 2
        r_equiv = float(np.sqrt(area / np.pi))
        return [{'type': 'circle', 'center': [center[0], center[1]],
                 'radius': r_equiv, 'layer': 0, 'label': 'Triangle'}]

    def _from_shape_cross_section(self, z_plane: float) -> list:
        return []

    # ------------------------------------------------------------------
    # Cluster-position helper (also used by field_analyzer)
    # ------------------------------------------------------------------

    def _calculate_cluster_positions(self, n_spheres: int, spacing: float) -> list:
        """Return hexagonal cluster positions matching MATLAB geometry."""
        return self._static_cluster_positions(n_spheres, spacing)

    @staticmethod
    def _static_cluster_positions(n_spheres: int, spacing: float) -> list:
        dy60 = spacing * 0.866025404

        hex_pos = []
        for i in range(6):
            angle = i * 60 * np.pi / 180
            hex_pos.append((spacing * np.cos(angle), spacing * np.sin(angle), 0.0))

        layouts = {
            1: [(0.0, 0.0, 0.0)],
            2: [(-spacing / 2, 0.0, 0.0), (spacing / 2, 0.0, 0.0)],
            3: [(-spacing / 2, 0.0, 0.0), (spacing / 2, 0.0, 0.0), (0.0, dy60, 0.0)],
            4: [(0.0, 0.0, 0.0)] + hex_pos[:3],
            5: [(0.0, 0.0, 0.0)] + hex_pos[:4],
            6: [(0.0, 0.0, 0.0)] + hex_pos[:5],
            7: [(0.0, 0.0, 0.0)] + hex_pos[:6],
        }
        if n_spheres not in layouts:
            raise ValueError('n_spheres must be 1-7, got {}'.format(n_spheres))
        return layouts[n_spheres]
