from typing import Any, Dict, List, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import (_build_eps_medium, _build_eps_particle, _count_faces,
        _resolve_materials_list, _resolve_rip)
from ..util import print_info


def _load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if 'verts' in data and 'faces' in data:
        return np.asarray(data['verts'], dtype = float), np.asarray(data['faces'])
    if 'vertices' in data and 'faces' in data:
        return np.asarray(data['vertices'], dtype = float), np.asarray(data['faces'])
    raise KeyError('[error] npz must contain <verts/vertices> + <faces>')


def _load_mat(path: str, key_prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.io import loadmat
    mat = loadmat(path)
    verts_key = '{}_verts'.format(key_prefix)
    faces_key = '{}_faces'.format(key_prefix)
    if verts_key not in mat or faces_key not in mat:
        raise KeyError('[error] .mat must contain <{}> and <{}>'.format(verts_key, faces_key))
    return np.asarray(mat[verts_key], dtype = float), np.asarray(mat[faces_key])


def _normalize_faces(faces: np.ndarray, n_verts: int) -> np.ndarray:
    faces = np.asarray(faces)

    if faces.dtype.kind == 'f' and np.any(np.isnan(faces)):
        valid = ~np.isnan(faces)
        if valid.any():
            min_idx = float(faces[valid].min())
            if min_idx >= 1.0 and faces[valid].max() > n_verts - 1:
                faces = faces.copy()
                faces[valid] = faces[valid] - 1.0
        return faces.astype(float)

    if faces.dtype.kind == 'f':
        if np.nanmin(faces) >= 1.0 and np.nanmax(faces) >= n_verts:
            return (faces - 1).astype(int)
        return faces.astype(int)

    if faces.min() >= 1 and faces.max() >= n_verts:
        return (faces - 1).astype(int)
    return faces.astype(int)


class FromShapeBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import ComParticle
        from mnpbem.geometry.particle import Particle

        verts: np.ndarray
        faces: np.ndarray

        if 'vertices' in self.cfg_struct and 'faces' in self.cfg_struct:
            verts = np.asarray(self.cfg_struct['vertices'], dtype = float)
            faces = np.asarray(self.cfg_struct['faces'])
        elif 'mesh_file' in self.cfg_struct:
            mesh_file = str(self.cfg_struct['mesh_file'])
            if mesh_file.endswith('.npz'):
                verts, faces = _load_npz(mesh_file)
            elif mesh_file.endswith('.mat'):
                key = str(self.cfg_struct.get('mesh_key', 'p'))
                verts, faces = _load_mat(mesh_file, key)
            else:
                raise ValueError('[error] <mesh_file> must be .npz or .mat')
        else:
            raise ValueError('[error] from_shape requires <vertices>+<faces> or <mesh_file>')

        faces = _normalize_faces(faces, verts.shape[0])

        if faces.ndim != 2 or faces.shape[1] not in (3, 4):
            raise ValueError('[error] <faces> must have shape (N,3) or (N,4); got <{}>'.format(faces.shape))

        if faces.shape[1] == 3:
            faces = np.column_stack([faces.astype(float), np.full(faces.shape[0], np.nan)])

        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'flat')

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        rip = _resolve_rip(self.cfg_struct, self.cfg_materials)
        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name, rip)
        epstab = [eps_medium, eps_particle]

        part = Particle(verts, faces, interp = interp)

        p = ComParticle(epstab, [part], [[2, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('FromShapeBuilder: nverts={}, nfaces={}'.format(verts.shape[0], nfaces))

        return p, epstab, nfaces
