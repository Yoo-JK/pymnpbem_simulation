from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import (_build_eps_medium, _build_eps_particle, _count_faces,
        _resolve_materials_list, _resolve_rip)
from ..util import print_info


# ---------------------------------------------------------------------------
# Pre-triangulated mesh loaders (existing path)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DDA .shape voxel loader  (ported from OLD ShapeFileLoader)
# ---------------------------------------------------------------------------

def _load_shape_file(path: str) -> np.ndarray:
    """Read a DDA .shape file and return integer array [N, 4] = [i, j, k, mat].

    Lines that do not start with a digit or '-' are treated as header/comments
    and skipped.  Only the first 4 columns are kept.
    """
    rows = []
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if not (line[0].isdigit() or line[0] == '-'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                rows.append([int(parts[0]), int(parts[1]),
                             int(parts[2]), int(parts[3])])
            except ValueError:
                continue

    if not rows:
        raise ValueError('[error] No valid voxel data found in <{}>'.format(path))

    return np.array(rows, dtype = int)


def _voxels_to_surface_mesh(
        voxel_coords: np.ndarray,
        voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Surface-only triangulation: emit faces only where a neighbour is absent.

    Returns vertices (0-indexed) and faces with shape (M, 4) where column 3 is
    NaN (MNPBEM convention for triangles).
    """
    voxel_set = set(map(tuple, voxel_coords))

    # Each cube face defined by 4 corner offsets (CCW from outside)
    face_offsets = [
        ([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], [ 0, 0,-1]),  # bottom  / -z neighbour
        ([[0,0,1],[0,1,1],[1,1,1],[1,0,1]], [ 0, 0, 1]),  # top     / +z
        ([[0,0,0],[0,1,0],[0,1,1],[0,0,1]], [-1, 0, 0]),  # left    / -x
        ([[1,0,0],[1,0,1],[1,1,1],[1,1,0]], [ 1, 0, 0]),  # right   / +x
        ([[0,0,0],[0,0,1],[1,0,1],[1,0,0]], [ 0,-1, 0]),  # front   / -y
        ([[0,1,0],[1,1,0],[1,1,1],[0,1,1]], [ 0, 1, 0]),  # back    / +y
    ]

    vert_map: Dict[Tuple, int] = {}
    verts_list: List[List[float]] = []
    faces_list: List[List[float]] = []

    for voxel in voxel_coords:
        i, j, k = int(voxel[0]), int(voxel[1]), int(voxel[2])
        for corners, nb_off in face_offsets:
            nb = (i + nb_off[0], j + nb_off[1], k + nb_off[2])
            if nb in voxel_set:
                continue  # interior face — skip
            vi = []
            for co in corners:
                key = (
                    (i + co[0]) * voxel_size,
                    (j + co[1]) * voxel_size,
                    (k + co[2]) * voxel_size,
                )
                if key not in vert_map:
                    vert_map[key] = len(verts_list)
                    verts_list.append(list(key))
                vi.append(vert_map[key])
            # Split quad into two triangles (0-indexed, NaN 4th col)
            faces_list.append([vi[0], vi[1], vi[2], np.nan])
            faces_list.append([vi[0], vi[2], vi[3], np.nan])

    return np.array(verts_list, dtype = float), np.array(faces_list, dtype = float)


def _voxels_to_cube_mesh(
        voxel_coords: np.ndarray,
        voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Full-cube triangulation: emit all 12 triangles per voxel (no neighbour check).

    Returns 0-indexed verts and faces with NaN in column 3.
    """
    cube_verts_tpl = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype = float)

    # 12 triangles, 0-indexed
    cube_faces_tpl = np.array([
        [0,1,2, np.nan],[0,2,3, np.nan],
        [4,7,6, np.nan],[4,6,5, np.nan],
        [0,4,5, np.nan],[0,5,1, np.nan],
        [3,2,6, np.nan],[3,6,7, np.nan],
        [0,3,7, np.nan],[0,7,4, np.nan],
        [1,5,6, np.nan],[1,6,2, np.nan],
    ], dtype = float)

    all_verts: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []

    for voxel in voxel_coords:
        offset_v = len(all_verts) * 8  # will rebuild below
        origin = np.array([voxel[0], voxel[1], voxel[2]], dtype = float) * voxel_size
        cv = cube_verts_tpl * voxel_size + origin
        v_start = sum(len(v) for v in all_verts)
        cf = cube_faces_tpl.copy()
        cf[:, :3] += v_start
        all_verts.append(cv)
        all_faces.append(cf)

    verts = np.vstack(all_verts) if all_verts else np.zeros((0, 3), dtype = float)
    faces = np.vstack(all_faces) if all_faces else np.zeros((0, 4), dtype = float)
    return verts, faces


def _voxels_to_mesh(
        voxel_coords: np.ndarray,
        voxel_size: float,
        method: str) -> Tuple[np.ndarray, np.ndarray]:
    if method == 'cube':
        return _voxels_to_cube_mesh(voxel_coords, voxel_size)
    return _voxels_to_surface_mesh(voxel_coords, voxel_size)


# ---------------------------------------------------------------------------
# FromShapeBuilder
# ---------------------------------------------------------------------------

class FromShapeBuilder(StructureBuilder):
    """Load a pre-triangulated mesh OR a DDA .shape voxel file.

    Pre-triangulated path (existing behaviour, unchanged)
    -------------------------------------------------------
    Config keys:  ``vertices`` + ``faces``  OR  ``mesh_file`` (.npz / .mat)

    DDA voxel path (new)
    ---------------------
    Config keys:
      ``shape_file``     — path to DDA .shape file (columns: i j k mat_type)
      ``voxel_size``     — nm per voxel (default 1.0)
      ``voxel_method``   — 'surface' (default) or 'cube'
      ``materials``      — list of material names in mat-index order
                           e.g. ['gold', 'silver'] for mat_type 1 and 2.
                           If omitted, all materials use ``particle``.

    Multi-material: each distinct ``mat_type`` value becomes a separate
    ``Particle`` inside one ``ComParticle``.  The ``epstab`` index for
    material ``mat_type`` (1-based) maps to ``materials[mat_type - 1]``.
    """

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import ComParticle
        from mnpbem.geometry.particle import Particle

        # ---- branch: voxel path ----
        if 'shape_file' in self.cfg_struct:
            return self._build_from_voxels()

        # ---- branch: pre-triangulated path (original) ----
        return self._build_prebuilt()

    # ------------------------------------------------------------------
    def _build_prebuilt(self) -> Tuple[Any, Any, int]:
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
            raise ValueError('[error] from_shape requires <vertices>+<faces>, '
                             '<mesh_file>, or <shape_file>')

        faces = _normalize_faces(faces, verts.shape[0])

        if faces.ndim != 2 or faces.shape[1] not in (3, 4):
            raise ValueError(
                '[error] <faces> must have shape (N,3) or (N,4); got <{}>'.format(
                    faces.shape))

        if faces.shape[1] == 3:
            faces = np.column_stack(
                [faces.astype(float), np.full(faces.shape[0], np.nan)])

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
        print_info('FromShapeBuilder: nverts={}, nfaces={}'.format(
            verts.shape[0], nfaces))

        return p, epstab, nfaces

    # ------------------------------------------------------------------
    def _build_from_voxels(self) -> Tuple[Any, Any, int]:
        """Build multi-material ComParticle from DDA .shape voxel file."""
        from mnpbem.geometry import ComParticle
        from mnpbem.geometry.particle import Particle

        shape_file = str(self.cfg_struct['shape_file'])
        voxel_size = float(self.cfg_struct.get('voxel_size', 1.0))
        voxel_method = str(self.cfg_struct.get('voxel_method', 'surface'))
        if voxel_method not in ('surface', 'cube'):
            raise ValueError(
                '[error] <voxel_method> must be "surface" or "cube", '
                'got <{}>'.format(voxel_method))

        mat_names_cfg = _resolve_materials_list(self.cfg_struct, self.cfg_materials)

        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'flat')

        medium_name = self.cfg_materials.get('medium', 'water')
        fallback_particle = self.cfg_materials.get('particle', 'gold')
        rip = _resolve_rip(self.cfg_struct, self.cfg_materials)

        # Load voxel data: [N, 4] = [i, j, k, mat_type]
        voxel_data = _load_shape_file(shape_file)
        unique_mats = np.unique(voxel_data[:, 3])
        n_mats = len(unique_mats)

        print_info('FromShapeBuilder (voxel): file={}, voxels={}, '
                   'materials={}, method={}'.format(
                       shape_file, len(voxel_data), unique_mats.tolist(),
                       voxel_method))

        eps_medium = _build_eps_medium(medium_name)
        epstab = [eps_medium]

        # Build eps entry per unique mat_type (sorted order)
        for idx, mat_idx in enumerate(unique_mats):
            if idx < len(mat_names_cfg):
                name = mat_names_cfg[idx]
            else:
                name = fallback_particle
            epstab.append(_build_eps_particle(name, rip))

        # Build one Particle per material
        particles = []
        inout = []
        mat_nfaces = []

        for eps_col, mat_idx in enumerate(unique_mats):
            # eps_col 0 → epstab[1], eps_col 1 → epstab[2], …
            mat_voxels = voxel_data[voxel_data[:, 3] == mat_idx, :3]
            verts, faces = _voxels_to_mesh(mat_voxels, voxel_size, voxel_method)

            if len(verts) == 0:
                raise ValueError(
                    '[error] Material {} produced empty mesh'.format(mat_idx))

            part = Particle(verts, faces, interp = interp)
            particles.append(part)

            # inside = this material's eps, outside = medium (eps index 1)
            inout.append([eps_col + 2, 1])
            mat_nfaces.append(len(faces))
            print_info('  mat_type={}: {} voxels → {} faces'.format(
                mat_idx, len(mat_voxels), len(faces)))

        p = ComParticle(epstab, particles, inout,
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('FromShapeBuilder (voxel): n_particles={}, total_nfaces={}'.format(
            n_mats, nfaces))

        return p, epstab, nfaces
