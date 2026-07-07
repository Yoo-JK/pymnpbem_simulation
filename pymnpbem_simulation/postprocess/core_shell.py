"""
Core/Shell separator for layered particles (Au@Ag, Au@Ag@AgCl etc.).

Classifies surface mesh faces into core / shell layers based on the YAML
config geometry. Used for surface charge plots, eigenmode visualization,
and any analysis that needs to distinguish core vs shell regions.

Supports:
    - Single core-shell cube / sphere (monomer)
    - Core-shell rod (capsule geometry)
    - Core-shell cube dimer (with optional rotation/tilt of particle 2)
    - Connected core-shell dimer (gap region shell removal)

Port of mnpbem_simulation/postprocess/post_utils/core_shell_separator.py
adapted to pymnpbem YAML schema.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CoreShellSeparator(object):
    """Classify mesh faces by layer (core / shell) and particle (in dimer)."""

    def __init__(self,
            structure_config: Dict[str, Any]) -> None:
        """Initialize from a pymnpbem-style structure config dict.

        Expected keys (depending on structure type):
            type: 'core_shell_cube' | 'core_shell_rod' | 'core_shell_sphere' |
                  'dimer_core_shell_cube' | 'connected_dimer_cube' (with
                  shell_layers)
            core_size, shell_layers, gap, offset, rotation_angle, tilt_angle,
                tilt_axis (cube variants)
            core_diameter, shell_thickness, height (rod)
        """
        self.config = structure_config or dict()
        self._is_core_shell = False
        self._sizes = []  # type: List[float]
        self._centers = []  # type: List[np.ndarray]
        self._thresholds = []  # type: List[float]
        self._rotation_angle = 0.0
        self._tilt_angle = 0.0
        self._tilt_axis = np.array([0.0, 1.0, 0.0])
        self._geometry_type = 'cube'
        self._is_connected = False
        self._rod_half_barrel = 0.0
        self._setup_geometry()

    def _setup_geometry(self) -> None:
        stype = str(self.config.get('type', '')).lower()

        if stype == 'core_shell_rod':
            self._setup_rod()
            return

        # cube / dimer variants. Detect either via explicit shell_layers or
        # via core_shell_cube / dimer_core_shell_cube type.
        shell_layers = self.config.get('shell_layers', None)
        if shell_layers is None and 'shell' in stype:
            # Allow legacy fields (core_size + shell_thickness, single layer / 단일 layer).
            thickness = self.config.get('shell_thickness', None)
            if thickness is not None:
                shell_layers = [float(thickness)]

        if shell_layers:
            self._setup_cube(shell_layers, stype)

    def _setup_cube(self,
            shell_layers: List[float],
            stype: str) -> None:
        core_size = float(self.config.get('core_size', 30))
        self._sizes = [core_size]
        for thickness in shell_layers:
            self._sizes.append(self._sizes[-1] + 2.0 * float(thickness))

        self._is_core_shell = True
        self._geometry_type = 'cube'

        if 'connected' in stype:
            self._is_connected = True

        if 'dimer' in stype:
            gap = float(self.config.get('gap', 10.0))
            offset = self.config.get('offset', [0.0, 0.0, 0.0])
            offset = list(offset) + [0.0] * (3 - len(offset))
            total_size = self._sizes[-1]
            shift = (total_size + gap) / 2.0
            self._centers = [
                    np.array([-shift, 0.0, 0.0]),
                    np.array([shift + offset[0], offset[1], offset[2]])]
            self._rotation_angle = float(self.config.get('rotation_angle', 0.0))
            self._tilt_angle = float(self.config.get('tilt_angle', 0.0))
            self._tilt_axis = np.array(
                    self.config.get('tilt_axis', [0.0, 1.0, 0.0]),
                    dtype = float)
        else:
            self._centers = [np.array([0.0, 0.0, 0.0])]

        for i in range(len(self._sizes) - 1):
            self._thresholds.append(self._sizes[i] / 2.0)

    def _setup_rod(self) -> None:
        core_diameter = float(self.config['core_diameter'])
        shell_thickness = float(self.config['shell_thickness'])
        height = float(self.config['height'])
        shell_diameter = core_diameter + 2.0 * shell_thickness

        self._sizes = [core_diameter, shell_diameter]
        self._centers = [np.array([0.0, 0.0, 0.0])]
        self._thresholds = [core_diameter / 2.0]
        self._rod_half_barrel = (height - shell_diameter) / 2.0
        self._geometry_type = 'rod'
        self._is_core_shell = True

    def is_core_shell_structure(self) -> bool:
        return self._is_core_shell

    def num_layers(self) -> int:
        """Total layers (core + shells)."""
        return len(self._sizes)

    @staticmethod
    def _rotation_matrix(angle_deg: float,
            axis: np.ndarray) -> np.ndarray:
        # Rodrigues' rotation formula.
        angle = np.radians(angle_deg)
        axis = axis / (np.linalg.norm(axis) + 1e-30)
        K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        return R

    def _get_inverse_rotation(self,
            particle_idx: int) -> Optional[np.ndarray]:
        # Only particle 2 (idx=1) in a dimer has rotation/tilt.
        if particle_idx == 0 or len(self._centers) == 1:
            return None

        if self._rotation_angle == 0 and self._tilt_angle == 0:
            return None

        R_z = self._rotation_matrix(self._rotation_angle, np.array([0.0, 0.0, 1.0]))
        R_tilt = self._rotation_matrix(self._tilt_angle, self._tilt_axis)
        return R_z.T @ R_tilt.T

    @staticmethod
    def _chebyshev_distance(centroids: np.ndarray,
            center: np.ndarray,
            inv_rotation: Optional[np.ndarray] = None) -> np.ndarray:
        local = centroids - center
        if inv_rotation is not None:
            local = local @ inv_rotation.T
        return np.max(np.abs(local), axis = 1)

    @staticmethod
    def _capsule_distance(centroids: np.ndarray,
            center: np.ndarray,
            half_barrel: float) -> np.ndarray:
        local = centroids - center
        x_clamped = np.clip(local[:, 0], -half_barrel, half_barrel)
        dx = local[:, 0] - x_clamped
        return np.sqrt(dx ** 2 + local[:, 1] ** 2 + local[:, 2] ** 2)

    def classify_faces(self,
            centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classify each face by (layer_index, particle_index).

        layer_index: 0 = core, 1 = first shell, 2 = second shell, ...
        particle_index: 0 (left/single) or 1 (right) for dimer.
        """
        centroids = np.asarray(centroids, dtype = float)
        n_faces = len(centroids)
        layer_indices = np.zeros(n_faces, dtype = int)
        particle_indices = np.zeros(n_faces, dtype = int)

        if not self._is_core_shell:
            return layer_indices, particle_indices

        if len(self._centers) == 1:
            if self._geometry_type == 'rod':
                dist = self._capsule_distance(
                        centroids, self._centers[0], self._rod_half_barrel)
            else:
                inv_rot = self._get_inverse_rotation(0)
                dist = self._chebyshev_distance(
                        centroids, self._centers[0], inv_rot)
        else:
            dist_to_p1 = np.linalg.norm(centroids - self._centers[0], axis = 1)
            dist_to_p2 = np.linalg.norm(centroids - self._centers[1], axis = 1)
            p2_mask = dist_to_p2 < dist_to_p1
            particle_indices[p2_mask] = 1

            dist = np.empty(n_faces, dtype = float)

            p1_mask = ~p2_mask
            if np.any(p1_mask):
                inv_rot_p1 = self._get_inverse_rotation(0)
                dist[p1_mask] = self._chebyshev_distance(
                        centroids[p1_mask], self._centers[0], inv_rot_p1)
            if np.any(p2_mask):
                inv_rot_p2 = self._get_inverse_rotation(1)
                dist[p2_mask] = self._chebyshev_distance(
                        centroids[p2_mask], self._centers[1], inv_rot_p2)

        for i, threshold in enumerate(self._thresholds):
            layer_indices[dist > threshold] = i + 1

        return layer_indices, particle_indices

    def get_core_mask(self,
            centroids: np.ndarray) -> np.ndarray:
        layer_indices, _ = self.classify_faces(centroids)
        return layer_indices == 0

    def get_shell_mask(self,
            centroids: np.ndarray) -> np.ndarray:
        layer_indices, _ = self.classify_faces(centroids)
        return layer_indices == len(self._sizes) - 1

    def get_layer_mask(self,
            centroids: np.ndarray,
            layer_index: int) -> np.ndarray:
        """Return boolean mask for a specific layer index."""
        layer_indices, _ = self.classify_faces(centroids)
        return layer_indices == int(layer_index)

    def get_cutaway_mask(self,
            centroids: np.ndarray,
            vertices: Optional[np.ndarray] = None,
            faces: Optional[np.ndarray] = None) -> np.ndarray:
        """Return mask for a cutaway view (y >= 0 shell + always core).

        - y >= 0: show all faces (shell visible from outside).
        - y < 0: show core only (shell removed to expose core).
        - For connected dimer: gap region shell is also removed.

        If `vertices` and `faces` are supplied, the cut uses min(vertex.y) of
        the face so that no shell triangle straddles the y=0 plane (cleaner
        visualization). Otherwise falls back to centroid.y comparison.
        """
        centroids = np.asarray(centroids, dtype = float)
        layer_indices, _ = self.classify_faces(centroids)
        core_mask = layer_indices == 0

        if vertices is not None and faces is not None:
            faces_arr = np.asarray(faces)
            verts_arr = np.asarray(vertices)

            # MNPBEM uses 1-based MATLAB indexing; convert to 0-based, handling
            # quad faces with NaN sentinels.
            if np.issubdtype(faces_arr.dtype, np.floating):
                nan_mask = np.isnan(faces_arr)
                faces_int = faces_arr.astype(np.int64) - 1
                faces_int[nan_mask] = 0
                vert_y = verts_arr[faces_int.ravel(), 1].reshape(len(faces_arr), -1)
                vert_y[nan_mask] = np.inf
            else:
                faces_int = faces_arr.astype(np.int64) - 1
                vert_y = verts_arr[faces_int.ravel(), 1].reshape(len(faces_arr), -1)

            min_y = vert_y.min(axis = 1)

            upper_half = np.ones(len(centroids), dtype = bool)
            shell_mask = ~core_mask
            upper_half[shell_mask & (min_y < 0)] = False
        else:
            upper_half = centroids[:, 1] >= 0

        result = upper_half | core_mask

        # Connected dimer: drop shell faces in the gap region.
        if self._is_connected and len(self._centers) == 2:
            core_half = self._sizes[0] / 2.0
            p1_inner_x = self._centers[0][0] + core_half
            p2_inner_x = self._centers[1][0] - core_half
            gap_x_min = min(p1_inner_x, p2_inner_x)
            gap_x_max = max(p1_inner_x, p2_inner_x)
            in_gap = (centroids[:, 0] >= gap_x_min) & (centroids[:, 0] <= gap_x_max)
            shell_mask = ~core_mask
            result[shell_mask & in_gap] = False

        return result


def make_separator_from_config(cfg: Dict[str, Any]) -> CoreShellSeparator:
    """Resolve the relevant structure block from a full YAML config and
    construct a CoreShellSeparator. Handles auto-wrapped substrate blocks
    where the user particle is nested under <structure.base>.
    """
    structure = cfg.get('structure', dict())

    if not isinstance(structure, dict):
        return CoreShellSeparator(dict())

    stype = str(structure.get('type', '')).lower()

    if stype == 'with_substrate' and isinstance(structure.get('base'), dict):
        return CoreShellSeparator(structure['base'])

    return CoreShellSeparator(structure)
