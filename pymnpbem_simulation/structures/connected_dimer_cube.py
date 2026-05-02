from typing import Any, Dict, List, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import _build_eps_medium, _build_eps_particle, _count_faces
from ..util import print_info


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
        from mnpbem.geometry import tricube, ComParticle
        from mnpbem.geometry.particle import Particle

        core_size = float(self.cfg_struct.get('core_size',
                self.cfg_struct.get('size', 30.0)))
        gap = float(self.cfg_struct.get('gap', 0.0))
        n_per_edge = int(self.cfg_struct.get('n_per_edge', 12))
        e = float(self.cfg_struct.get('e', self.cfg_struct.get('rounding', 0.25)))
        offset = list(self.cfg_struct.get('offset', [0.0, 0.0, 0.0]))
        tilt_angle = float(self.cfg_struct.get('tilt_angle', 0.0))
        tilt_axis = list(self.cfg_struct.get('tilt_axis', [0.0, 1.0, 0.0]))
        rotation_angle = float(self.cfg_struct.get('rotation_angle', 0.0))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'flat')

        if gap > 0.0:
            raise ValueError(
                '[error] connected_dimer_cube requires <gap> <= 0; '
                'got <{}>. Use <advanced_dimer_cube> for separated cubes.'.format(gap))

        shift_distance = (core_size + gap) / 2.0

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name)
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

        fused = Particle(clean_verts, clean_faces, interp = interp)

        p = ComParticle(epstab, [fused], [[2, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('ConnectedDimerCubeBuilder: core={}nm, gap={}nm, fused_nfaces={}'.format(
            core_size, gap, nfaces))

        return p, epstab, nfaces
