from typing import Any, Dict, List, Tuple

import numpy as np

from .advanced_monomer_cube import _resolve_n_per_edge
from .base import StructureBuilder
from .sphere import (_build_eps_medium, _build_eps_particle, _count_faces,
        _resolve_materials_list, _resolve_rip)
from ..util import print_info


_CONNECTED_DIMER_CUBE_DEFAULT_N = 12


_RAY_DIRS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.577, 0.577, 0.577],
    [-0.577, 0.577, 0.577],
    [0.577, -0.577, 0.577],
    [0.707, 0.707, 0.0]])

_EPSILON = 1e-10


def _ray_triangle_intersect(origin: np.ndarray,
        direction: np.ndarray,
        v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)
    if abs(a) < _EPSILON:
        return False
    f = 1.0 / a
    s = origin - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * np.dot(edge2, q)
    return t > _EPSILON


def _point_in_mesh(point: np.ndarray,
        verts: np.ndarray,
        face_tris: np.ndarray) -> bool:
    inside_votes = 0
    total = len(_RAY_DIRS)
    for direction in _RAY_DIRS:
        intersect_count = 0
        for tri in face_tris:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            if _ray_triangle_intersect(point, direction, v0, v1, v2):
                intersect_count += 1
        if intersect_count % 2 == 1:
            inside_votes += 1
    return inside_votes > (total / 2)


def _faces_to_triangles(faces: np.ndarray) -> np.ndarray:
    tris = []
    for face in faces:
        face = np.asarray(face)
        valid = face[~np.isnan(face)]
        valid = valid.astype(int)
        if len(valid) >= 3:
            tris.append([valid[0], valid[1], valid[2]])
        if len(valid) == 4:
            tris.append([valid[0], valid[2], valid[3]])
    return np.array(tris, dtype = int)


def _face_centroids(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    centroids = np.zeros((faces.shape[0], 3), dtype = float)
    for i, face in enumerate(faces):
        face = np.asarray(face)
        valid = face[~np.isnan(face)].astype(int)
        centroids[i] = np.mean(verts[valid], axis = 0)
    return centroids


def _fuse_two_meshes(verts1: np.ndarray, faces1: np.ndarray,
        verts2: np.ndarray, faces2: np.ndarray,
        margin: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    centroids1 = _face_centroids(verts1, faces1)
    centroids2 = _face_centroids(verts2, faces2)

    tris1 = _faces_to_triangles(faces1)
    tris2 = _faces_to_triangles(faces2)

    bbox1_min = verts1.min(axis = 0) - margin
    bbox1_max = verts1.max(axis = 0) + margin
    bbox2_min = verts2.min(axis = 0) - margin
    bbox2_max = verts2.max(axis = 0) + margin

    inside1 = np.zeros(len(centroids1), dtype = bool)
    for i, pt in enumerate(centroids1):
        if np.all(pt >= bbox2_min) and np.all(pt <= bbox2_max):
            inside1[i] = _point_in_mesh(pt, verts2, tris2)

    inside2 = np.zeros(len(centroids2), dtype = bool)
    for i, pt in enumerate(centroids2):
        if np.all(pt >= bbox1_min) and np.all(pt <= bbox1_max):
            inside2[i] = _point_in_mesh(pt, verts1, tris1)

    keep1 = ~inside1
    keep2 = ~inside2

    faces1_kept = faces1[keep1]
    faces2_kept = faces2[keep2]

    n_verts1 = verts1.shape[0]
    merged_verts = np.vstack([verts1, verts2])

    faces2_offset = faces2_kept.copy().astype(float)
    valid_mask = ~np.isnan(faces2_offset)
    faces2_offset[valid_mask] = faces2_offset[valid_mask] + n_verts1

    merged_faces = np.vstack([faces1_kept.astype(float), faces2_offset])

    used = np.unique(merged_faces[~np.isnan(merged_faces)]).astype(int)
    vert_map = -np.ones(n_verts1 + verts2.shape[0], dtype = int)
    vert_map[used] = np.arange(len(used))

    clean_faces = merged_faces.copy()
    valid_mask = ~np.isnan(clean_faces)
    clean_faces[valid_mask] = vert_map[clean_faces[valid_mask].astype(int)]
    clean_verts = merged_verts[used]

    return clean_verts, clean_faces


class ConnectedDimerCubeBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        shell_layers = list(self.cfg_struct.get('shell_layers', []) or [])
        if shell_layers:
            return self._build_core_shell(shell_layers)
        return self._build_single()

    # ------------------------------------------------------------------
    # Single-material branch (backward compatibility)
    # ------------------------------------------------------------------
    def _build_single(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import tricube, ComParticle
        from mnpbem.geometry.particle import Particle

        core_size = float(self.cfg_struct.get('core_size',
                self.cfg_struct.get('size', 30.0)))
        gap = float(self.cfg_struct.get('gap', 0.0))
        if self.cfg_struct.get('mesh_density') is None and self.cfg_struct.get('n_per_edge') is None:
            n_per_edge = _CONNECTED_DIMER_CUBE_DEFAULT_N
        else:
            n_per_edge = _resolve_n_per_edge(self.cfg_struct, 1, edge_override = core_size)[0]
        e = float(self.cfg_struct.get('e', self.cfg_struct.get('rounding', 0.25)))
        offset = list(self.cfg_struct.get('offset', [0.0, 0.0, 0.0]))
        tilt_angle = float(self.cfg_struct.get('tilt_angle', 0.0))
        tilt_axis = list(self.cfg_struct.get('tilt_axis', [0.0, 1.0, 0.0]))
        rotation_angle = float(self.cfg_struct.get('rotation_angle', 0.0))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        if gap > 0.0:
            raise ValueError(
                '[error] connected_dimer_cube requires <gap> <= 0; '
                'got <{}>. Use <advanced_dimer_cube> for separated cubes.'.format(gap))

        shift_distance = (core_size + gap) / 2.0

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        rip = _resolve_rip(self.cfg_struct, self.cfg_materials)
        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name, rip)
        epstab = [eps_medium, eps_particle]

        c1 = tricube(n_per_edge, core_size, e = e)
        c1.shift([-shift_distance, 0.0, 0.0])

        c2 = tricube(n_per_edge, core_size, e = e)
        if rotation_angle != 0.0:
            c2.rot(rotation_angle, [0.0, 0.0, 1.0])
        if tilt_angle != 0.0:
            c2.rot(tilt_angle, tilt_axis)
        c2.shift([shift_distance + offset[0], offset[1], offset[2]])

        clean_verts, clean_faces = _fuse_two_meshes(
                c1.verts, c1.faces, c2.verts, c2.faces)

        # See note in _build_core_shell: fused-mesh Particle must be created
        # with 'flat' so verts2/faces2 stay None. ComParticle then promotes
        # to 'curv' (calling curved() -> midpoints()) when the user asks for it.
        fused = Particle(clean_verts, clean_faces, interp = 'flat')

        p = ComParticle(epstab, [fused], [[2, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('ConnectedDimerCubeBuilder: core={}nm, gap={}nm, fused_nfaces={}'.format(
            core_size, gap, nfaces))

        return p, epstab, nfaces

    # ------------------------------------------------------------------
    # Core-shell branch (shell_layers given)
    # ------------------------------------------------------------------
    def _build_core_shell(self, shell_layers: List[Any]) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import tricube, ComParticle
        from mnpbem.geometry.particle import Particle

        if len(shell_layers) != 1:
            raise ValueError(
                '[error] connected_dimer_cube core-shell mode requires exactly '
                '1 shell layer, got <{}>.'.format(len(shell_layers)))

        core_size = float(self.cfg_struct.get('core_size',
                self.cfg_struct.get('size', 30.0)))
        shell_thickness = float(shell_layers[0])
        shell_size = core_size + 2.0 * shell_thickness

        gap = float(self.cfg_struct.get('gap', 0.0))
        if gap > 0.0:
            raise ValueError(
                '[error] connected_dimer_cube requires <gap> <= 0; '
                'got <{}>. Use <advanced_dimer_cube> for separated cubes.'.format(gap))

        offset = list(self.cfg_struct.get('offset', [0.0, 0.0, 0.0]))
        tilt_angle = float(self.cfg_struct.get('tilt_angle', 0.0))
        tilt_axis = list(self.cfg_struct.get('tilt_axis', [0.0, 1.0, 0.0]))
        rotation_angle = float(self.cfg_struct.get('rotation_angle', 0.0))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        # rounding: roundings=[core, shell] or single rounding/e
        if 'roundings' in self.cfg_struct:
            roundings = list(self.cfg_struct['roundings'])
            if len(roundings) != 2:
                raise ValueError(
                    '[error] connected_dimer_cube core-shell requires '
                    '<roundings> with 2 values [core, shell]; got <{}>.'.format(
                        len(roundings)))
            core_rounding = float(roundings[0])
            shell_rounding = float(roundings[1])
        else:
            single_e = float(self.cfg_struct.get('e',
                    self.cfg_struct.get('rounding', 0.25)))
            core_rounding = single_e
            shell_rounding = single_e

        # n_per_edge per layer (mesh_density honors actual layer size)
        if (self.cfg_struct.get('mesh_density') is None
                and self.cfg_struct.get('n_per_edge') is None
                and 'n_per_edges' not in self.cfg_struct):
            core_n_per_edge = _CONNECTED_DIMER_CUBE_DEFAULT_N
            shell_n_per_edge = _CONNECTED_DIMER_CUBE_DEFAULT_N
        else:
            core_n_per_edge = _resolve_n_per_edge(
                    self.cfg_struct, 1, edge_override = core_size)[0]
            shell_n_per_edge = _resolve_n_per_edge(
                    self.cfg_struct, 1, edge_override = shell_size)[0]

        # Materials: cfg_struct.materials list > cfg_materials.particle_list >
        # [cfg_materials.particle, cfg_materials.shell]
        materials = self._resolve_materials()

        # Shifts: particle centers at +/- shell_shift (cores share that center)
        shell_shift = (shell_size + gap) / 2.0
        core_gap = gap + 2.0 * shell_thickness
        fuse_cores = (core_gap <= 0.0)

        medium_name = self.cfg_materials.get('medium', 'water')
        rip = _resolve_rip(self.cfg_struct, self.cfg_materials)
        eps_medium = _build_eps_medium(medium_name)
        eps_core = _build_eps_particle(materials[0], rip)
        eps_shell = _build_eps_particle(materials[1], rip)
        epstab = [eps_medium, eps_core, eps_shell]

        # ----- Shell meshes (always fused) -----
        s1 = tricube(shell_n_per_edge, shell_size, e = shell_rounding)
        s1.shift([-shell_shift, 0.0, 0.0])

        s2 = tricube(shell_n_per_edge, shell_size, e = shell_rounding)
        if rotation_angle != 0.0:
            s2.rot(rotation_angle, [0.0, 0.0, 1.0])
        if tilt_angle != 0.0:
            s2.rot(tilt_angle, tilt_axis)
        s2.shift([shell_shift + offset[0], offset[1], offset[2]])

        shell_verts, shell_faces = _fuse_two_meshes(
                s1.verts, s1.faces, s2.verts, s2.faces)
        # Always build the fused-mesh Particle as 'flat'. If the user asked
        # for 'curv', ComParticle.__init__ will call particle.curved() which
        # generates midpoints on the fused mesh — that path works only when
        # the Particle was constructed with interp='flat' first (verts2 None).
        p_shell = Particle(shell_verts, shell_faces, interp = 'flat')

        # ----- Core meshes (fused or separate) -----
        c1 = tricube(core_n_per_edge, core_size, e = core_rounding)
        c1.shift([-shell_shift, 0.0, 0.0])

        c2 = tricube(core_n_per_edge, core_size, e = core_rounding)
        if rotation_angle != 0.0:
            c2.rot(rotation_angle, [0.0, 0.0, 1.0])
        if tilt_angle != 0.0:
            c2.rot(tilt_angle, tilt_axis)
        c2.shift([shell_shift + offset[0], offset[1], offset[2]])

        if fuse_cores:
            core_verts, core_faces = _fuse_two_meshes(
                    c1.verts, c1.faces, c2.verts, c2.faces)
            p_core = Particle(core_verts, core_faces, interp = 'flat')
            particles = [p_core, p_shell]
            inout = [[2, 3], [3, 1]]
        else:
            particles = [c1, c2, p_shell]
            inout = [[2, 3], [2, 3], [3, 1]]

        p = ComParticle(epstab, particles, inout,
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info(
                'ConnectedDimerCubeBuilder[core-shell]: core={}nm, '
                'shell_thickness={}nm, shell_size={}nm, gap={}nm, '
                'core_gap={:.4f}nm ({}), materials={}, '
                'n_per_edge(core/shell)=({}/{}), nfaces={}'.format(
                    core_size, shell_thickness, shell_size, gap,
                    core_gap, 'FUSED' if fuse_cores else 'SEPARATE',
                    materials, core_n_per_edge, shell_n_per_edge, nfaces))

        return p, epstab, nfaces

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_materials(self) -> List[str]:
        """Resolve [core_material, shell_material] for core-shell mode.

        Precedence (via _resolve_materials_list, which tolerates both the
        ``structure.materials`` and migrated ``materials.particle_list``
        layouts):
            1. cfg_struct.materials  (list of length 2)
            2. cfg_materials.particle_list  (list of length 2)
            3. [cfg_materials.particle, cfg_materials.shell] fallback
        """
        mats = _resolve_materials_list(self.cfg_struct, self.cfg_materials)
        if len(mats) == 2:
            return [str(mats[0]), str(mats[1])]

        if len(mats) != 0:
            raise ValueError(
                '[error] connected_dimer_cube core-shell requires <materials> '
                'with 2 entries [core, shell]; got <{}>.'.format(len(mats)))

        core_name = self.cfg_materials.get('core',
                self.cfg_materials.get('particle', 'gold'))
        shell_name = self.cfg_materials.get('shell', 'silver')
        return [str(core_name), str(shell_name)]
