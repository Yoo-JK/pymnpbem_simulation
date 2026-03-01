import os
import sys

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path

from mnpbem.geometry.particle import Particle
from mnpbem.geometry.mesh_generators import (
    trisphere, tricube, trirod, tripolygon, trispheresegment)
from mnpbem.geometry.polygon import Polygon
from mnpbem.geometry.edgeprofile import EdgeProfile


# ============================================================================
# DDA Shape File Loader
# ============================================================================

class ShapeFileLoader(object):

    def __init__(self,
            shape_path: Union[str, Path],
            voxel_size: float = 1.0,
            method: str = 'surface',
            verbose: bool = False) -> None:

        self.shape_path = Path(shape_path)
        self.voxel_size = voxel_size
        self.method = method
        self.verbose = verbose

        if not self.shape_path.exists():
            raise FileNotFoundError(
                '[error] Shape file not found: {}'.format(self.shape_path))

        assert method in ('surface', 'cube'), \
            '[error] <method> must be "surface" or "cube", got "{}"'.format(method)

        self.voxel_data = None  # [i, j, k, mat_idx]
        self.unique_materials = None
        self.material_particles = {}  # {mat_idx: {'vertices': ..., 'faces': ...}}

    def load(self) -> Dict[int, Dict[str, np.ndarray]]:

        if self.verbose:
            print('[info] Loading DDA shape file: {}'.format(self.shape_path))

        data_lines = []
        with open(self.shape_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not (line[0].isdigit() or line[0] == '-'):
                    if self.verbose:
                        print('  Skipping header/comment line: {}'.format(line))
                    continue
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        i, j, k, mat_type = (
                            int(parts[0]), int(parts[1]),
                            int(parts[2]), int(parts[3]))
                        data_lines.append([i, j, k, mat_type])
                except (ValueError, IndexError):
                    if self.verbose:
                        print('  Skipping invalid line: {}'.format(line))
                    continue

        if not data_lines:
            raise ValueError(
                '[error] No valid voxel data found in {}'.format(self.shape_path))

        data = np.array(data_lines, dtype = int)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        assert data.shape[1] >= 4, \
            '[error] Shape file must have >= 4 columns [i,j,k,mat_type], got {}'.format(
                data.shape[1])

        self.voxel_data = data[:, :4]
        self.unique_materials = np.unique(self.voxel_data[:, 3])

        if self.verbose:
            print('  Total voxels: {}'.format(len(self.voxel_data)))
            print('  Unique materials: {}'.format(self.unique_materials.tolist()))
            for mat_idx in self.unique_materials:
                count = np.sum(self.voxel_data[:, 3] == mat_idx)
                print('    Material {}: {} voxels'.format(mat_idx, count))

        # Convert each material to mesh
        for mat_idx in self.unique_materials:
            mat_voxels = self.voxel_data[self.voxel_data[:, 3] == mat_idx][:, :3]

            if self.verbose:
                print('  Converting material {}...'.format(mat_idx))

            if self.method == 'surface':
                vertices, faces = self._voxels_to_surface_mesh(mat_voxels)
            else:
                vertices, faces = self._voxels_to_cube_mesh(mat_voxels)

            self.material_particles[mat_idx] = {
                'vertices': vertices,
                'faces': faces,
            }

            if self.verbose:
                print('    -> {} vertices, {} faces'.format(
                    len(vertices), len(faces)))

        return self.material_particles

    def generate(self) -> List[Particle]:

        if self.material_particles is None or len(self.material_particles) == 0:
            self.load()

        particles = []
        for mat_idx in sorted(self.unique_materials):
            data = self.material_particles[mat_idx]
            vertices = data['vertices']
            faces = data['faces']

            # Faces from voxel mesh are 1-indexed; Particle expects 0-indexed
            faces_0idx = faces[:, :3].copy()
            nan_mask = ~np.isnan(faces_0idx)
            faces_0idx[nan_mask] = faces_0idx[nan_mask] - 1
            faces_0idx = faces_0idx.astype(int)

            p = Particle(vertices, faces_0idx, interp = 'flat')
            particles.append(p)

            if self.verbose:
                print('[info] Material {}: Particle with {} verts, {} faces'.format(
                    mat_idx, p.nverts, p.nfaces))

        return particles

    def _voxels_to_surface_mesh(self,
            voxel_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        voxel_set = set(map(tuple, voxel_coords))

        vertices_list = []
        faces_list = []
        vertex_map = {}

        cube_face_offsets = [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # bottom
            [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],  # top
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],  # left
            [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],  # right
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],  # front
            [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],  # back
        ]

        neighbors = [
            [0, 0, -1], [0, 0, 1], [-1, 0, 0],
            [1, 0, 0], [0, -1, 0], [0, 1, 0],
        ]

        for voxel in voxel_coords:
            i, j, k = voxel

            for face_idx, neighbor_offset in enumerate(neighbors):
                neighbor = (i + neighbor_offset[0],
                            j + neighbor_offset[1],
                            k + neighbor_offset[2])

                if neighbor not in voxel_set:
                    face_verts_offsets = cube_face_offsets[face_idx]
                    vert_indices = []

                    for vert_offset in face_verts_offsets:
                        vx = (i + vert_offset[0]) * self.voxel_size
                        vy = (j + vert_offset[1]) * self.voxel_size
                        vz = (k + vert_offset[2]) * self.voxel_size
                        vert_key = (vx, vy, vz)

                        if vert_key not in vertex_map:
                            vertex_map[vert_key] = len(vertices_list)
                            vertices_list.append([vx, vy, vz])

                        vert_indices.append(vertex_map[vert_key] + 1)

                    faces_list.append(
                        [vert_indices[0], vert_indices[1], vert_indices[2], np.nan])
                    faces_list.append(
                        [vert_indices[0], vert_indices[2], vert_indices[3], np.nan])

        return np.array(vertices_list), np.array(faces_list)

    def _voxels_to_cube_mesh(self,
            voxel_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        cube_vert_template = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype = float)

        cube_face_template = np.array([
            [1, 2, 3, np.nan], [1, 3, 4, np.nan],
            [5, 8, 7, np.nan], [5, 7, 6, np.nan],
            [1, 5, 6, np.nan], [1, 6, 2, np.nan],
            [4, 3, 7, np.nan], [4, 7, 8, np.nan],
            [1, 4, 8, np.nan], [1, 8, 5, np.nan],
            [2, 6, 7, np.nan], [2, 7, 3, np.nan],
        ])

        n_voxels = len(voxel_coords)
        n_verts_per = 8
        n_faces_per = 12

        all_verts = np.empty((n_voxels * n_verts_per, 3), dtype = float)
        all_faces = np.empty((n_voxels * n_faces_per, 4), dtype = float)

        for idx, voxel in enumerate(voxel_coords):
            i, j, k = voxel

            cube_verts = cube_vert_template * self.voxel_size
            cube_verts[:, 0] += i * self.voxel_size
            cube_verts[:, 1] += j * self.voxel_size
            cube_verts[:, 2] += k * self.voxel_size

            vert_offset = idx * n_verts_per
            all_verts[vert_offset:vert_offset + n_verts_per] = cube_verts
            all_faces[idx * n_faces_per:(idx + 1) * n_faces_per] = (
                cube_face_template + vert_offset)

        return all_verts, all_faces


# ============================================================================
# Adaptive Cube Mesh Generator
# ============================================================================

class AdaptiveCubeMesh(object):

    def __init__(self,
            size: float,
            rounding: float = 0.2,
            verbose: bool = False) -> None:

        self.size = size
        self.rounding = rounding
        self.verbose = verbose
        self.half_size = size / 2
        self.r = rounding * self.half_size * 0.5

    def generate(self,
            densities: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:

        # Define which faces share each edge
        edge_adjacency = {
            'x_pp': ('+y', '+z'), 'x_pm': ('+y', '-z'),
            'x_mp': ('-y', '+z'), 'x_mm': ('-y', '-z'),
            'y_pp': ('+x', '+z'), 'y_pm': ('+x', '-z'),
            'y_mp': ('-x', '+z'), 'y_mm': ('-x', '-z'),
            'z_pp': ('+x', '+y'), 'z_pm': ('+x', '-y'),
            'z_mp': ('-x', '+y'), 'z_mm': ('-x', '-y'),
        }

        edge_densities = {}
        for edge_key, (face1, face2) in edge_adjacency.items():
            d1 = densities.get(face1, 12)
            d2 = densities.get(face2, 12)
            edge_densities[edge_key] = max(d1, d2)

        face_edges = {
            '+x': ['y_pm', 'z_pp', 'y_pp', 'z_pm'],
            '-x': ['y_mm', 'z_mp', 'y_mp', 'z_mm'],
            '+y': ['x_pm', 'z_pp', 'x_pp', 'z_mp'],
            '-y': ['x_mm', 'z_pm', 'x_mp', 'z_mm'],
            '+z': ['x_mp', 'y_pp', 'x_pp', 'y_mp'],
            '-z': ['x_mm', 'y_pm', 'x_pm', 'y_mm'],
        }

        if self.verbose:
            print('[info] Gradual adaptive mesh:')
            for face_name, d in densities.items():
                edges = face_edges[face_name]
                edge_d = [edge_densities[e] for e in edges]
                print('  {}: interior={}, edges={}'.format(face_name, d, edge_d))

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        face_defs = {
            '+x': (0, +1), '-x': (0, -1),
            '+y': (1, +1), '-y': (1, -1),
            '+z': (2, +1), '-z': (2, -1),
        }

        for face_name, (axis, sign) in face_defs.items():
            inner_density = densities.get(face_name, 12)

            edges = face_edges[face_name]
            face_edge_densities = [edge_densities[e] for e in edges]

            verts, faces = self._generate_face_gradual(
                axis, sign, face_edge_densities, inner_density)

            verts = self._apply_rounding(verts)

            faces_offset = faces + vertex_offset

            all_vertices.append(verts)
            all_faces.append(faces_offset)

            vertex_offset += len(verts)

        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces)

        vertices, faces = self._merge_vertices(vertices, faces)

        valid_mask = []
        for face in faces:
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            is_valid = (v1 != v2) and (v2 != v3) and (v1 != v3)
            valid_mask.append(is_valid)

        faces = faces[valid_mask]

        if self.verbose:
            n_removed = sum(1 for v in valid_mask if not v)
            if n_removed > 0:
                print('  Removed {} degenerate triangles'.format(n_removed))
            print('  Total: {} vertices, {} faces'.format(
                len(vertices), len(faces)))

        return vertices, faces

    def generate_particle(self,
            densities: Dict[str, int]) -> Particle:
        # 0-indexed faces for Particle
        verts, faces = self.generate(densities)
        faces_0idx = faces[:, :3].copy()
        nan_mask = ~np.isnan(faces_0idx)
        faces_0idx[nan_mask] = faces_0idx[nan_mask] - 1
        faces_0idx = faces_0idx.astype(int)
        return Particle(verts, faces_0idx, interp = 'flat')

    # ------------------------------------------------------------------
    # Internal mesh generation helpers
    # ------------------------------------------------------------------

    def _generate_face_gradual(self,
            axis: int,
            sign: int,
            edge_densities: List[int],
            inner_n: int) -> Tuple[np.ndarray, np.ndarray]:

        h = self.half_size

        if all(d == inner_n for d in edge_densities):
            return self._generate_face_simple(axis, sign, inner_n)

        if axis == 0:
            ax1, ax2 = 1, 2
        elif axis == 1:
            ax1, ax2 = 0, 2
        else:
            ax1, ax2 = 0, 1

        vertices = []
        faces_list = []

        n_bottom, n_right, n_top, n_left = edge_densities

        # === 1. Boundary vertices ===
        boundary_verts = []

        bottom_coords = np.linspace(-h, h, n_bottom + 1)
        for i in range(n_bottom):
            v = [0, 0, 0]
            v[axis] = sign * h
            v[ax1] = bottom_coords[i]
            v[ax2] = -h
            boundary_verts.append(v)
        bottom_start = 0
        bottom_count = n_bottom

        right_coords = np.linspace(-h, h, n_right + 1)
        for i in range(n_right):
            v = [0, 0, 0]
            v[axis] = sign * h
            v[ax1] = h
            v[ax2] = right_coords[i]
            boundary_verts.append(v)
        right_start = bottom_count
        right_count = n_right

        top_coords = np.linspace(h, -h, n_top + 1)
        for i in range(n_top):
            v = [0, 0, 0]
            v[axis] = sign * h
            v[ax1] = top_coords[i]
            v[ax2] = h
            boundary_verts.append(v)
        top_start = right_start + right_count
        top_count = n_top

        left_coords = np.linspace(h, -h, n_left + 1)
        for i in range(n_left):
            v = [0, 0, 0]
            v[axis] = sign * h
            v[ax1] = -h
            v[ax2] = left_coords[i]
            boundary_verts.append(v)
        left_start = top_start + top_count
        left_count = n_left

        n_boundary = len(boundary_verts)

        # === 2. Interior grid ===
        margin = h * 0.15  # 15% margin for transition
        inner_h = h - margin
        inner_coords = np.linspace(-inner_h, inner_h, inner_n + 1)

        interior_verts = []
        for i in range(inner_n + 1):
            for j in range(inner_n + 1):
                v = [0, 0, 0]
                v[axis] = sign * h
                v[ax1] = inner_coords[i]
                v[ax2] = inner_coords[j]
                interior_verts.append(v)

        vertices = boundary_verts + interior_verts

        # === 3. Interior grid faces ===
        for i in range(inner_n):
            for j in range(inner_n):
                v00 = n_boundary + i * (inner_n + 1) + j
                v10 = n_boundary + (i + 1) * (inner_n + 1) + j
                v01 = n_boundary + i * (inner_n + 1) + (j + 1)
                v11 = n_boundary + (i + 1) * (inner_n + 1) + (j + 1)

                if sign > 0:
                    faces_list.append([v00 + 1, v10 + 1, v11 + 1, np.nan])
                    faces_list.append([v00 + 1, v11 + 1, v01 + 1, np.nan])
                else:
                    faces_list.append([v00 + 1, v11 + 1, v10 + 1, np.nan])
                    faces_list.append([v00 + 1, v01 + 1, v11 + 1, np.nan])

        # === 4. Transition triangles ===
        self._add_transition_gradual(
            faces_list, boundary_verts, n_boundary, inner_n, sign,
            bottom_start, bottom_count, 'bottom')
        self._add_transition_gradual(
            faces_list, boundary_verts, n_boundary, inner_n, sign,
            right_start, right_count, 'right')
        self._add_transition_gradual(
            faces_list, boundary_verts, n_boundary, inner_n, sign,
            top_start, top_count, 'top')
        self._add_transition_gradual(
            faces_list, boundary_verts, n_boundary, inner_n, sign,
            left_start, left_count, 'left')

        return np.array(vertices), np.array(faces_list)

    def _add_transition_gradual(self,
            faces: List,
            boundary_verts: List,
            n_boundary: int,
            inner_n: int,
            sign: int,
            edge_start: int,
            edge_count: int,
            side: str) -> None:

        b_indices = []
        for i in range(edge_count):
            b_indices.append(edge_start + i)
        next_start = (edge_start + edge_count) % n_boundary
        b_indices.append(next_start)

        if side == 'bottom':
            i_indices = [n_boundary + j for j in range(inner_n + 1)]
        elif side == 'right':
            i_indices = [
                n_boundary + inner_n + j * (inner_n + 1)
                for j in range(inner_n + 1)]
        elif side == 'top':
            i_indices = [
                n_boundary + (inner_n + 1) * inner_n + (inner_n - j)
                for j in range(inner_n + 1)]
        else:  # left
            i_indices = [
                n_boundary + (inner_n - j) * (inner_n + 1)
                for j in range(inner_n + 1)]

        n_b = len(b_indices)
        n_i = len(i_indices)

        bi, ii = 0, 0
        while bi < n_b - 1 or ii < n_i - 1:
            if bi >= n_b - 1:
                if ii < n_i - 1:
                    v1, v2, v3 = b_indices[-1], i_indices[ii], i_indices[ii + 1]
                    if sign > 0:
                        faces.append([v1 + 1, v2 + 1, v3 + 1, np.nan])
                    else:
                        faces.append([v1 + 1, v3 + 1, v2 + 1, np.nan])
                    ii += 1
            elif ii >= n_i - 1:
                if bi < n_b - 1:
                    v1, v2, v3 = b_indices[bi], b_indices[bi + 1], i_indices[-1]
                    if sign > 0:
                        faces.append([v1 + 1, v2 + 1, v3 + 1, np.nan])
                    else:
                        faces.append([v1 + 1, v3 + 1, v2 + 1, np.nan])
                    bi += 1
            else:
                b_ratio = bi / max(n_b - 1, 1)
                i_ratio = ii / max(n_i - 1, 1)

                if b_ratio <= i_ratio:
                    v1, v2, v3 = b_indices[bi], b_indices[bi + 1], i_indices[ii]
                    if sign > 0:
                        faces.append([v1 + 1, v2 + 1, v3 + 1, np.nan])
                    else:
                        faces.append([v1 + 1, v3 + 1, v2 + 1, np.nan])
                    bi += 1
                else:
                    v1, v2, v3 = b_indices[bi], i_indices[ii], i_indices[ii + 1]
                    if sign > 0:
                        faces.append([v1 + 1, v2 + 1, v3 + 1, np.nan])
                    else:
                        faces.append([v1 + 1, v3 + 1, v2 + 1, np.nan])
                    ii += 1

    def _generate_face_simple(self,
            axis: int,
            sign: int,
            n: int) -> Tuple[np.ndarray, np.ndarray]:

        h = self.half_size

        u = np.linspace(-h, h, n + 1)
        v = np.linspace(-h, h, n + 1)
        uu, vv = np.meshgrid(u, v)
        uu = uu.flatten()
        vv = vv.flatten()

        vertices = np.zeros((len(uu), 3))

        if axis == 0:
            vertices[:, 0] = sign * h
            vertices[:, 1] = uu
            vertices[:, 2] = vv
        elif axis == 1:
            vertices[:, 0] = uu
            vertices[:, 1] = sign * h
            vertices[:, 2] = vv
        else:
            vertices[:, 0] = uu
            vertices[:, 1] = vv
            vertices[:, 2] = sign * h

        faces = []
        for i in range(n):
            for j in range(n):
                v00 = i * (n + 1) + j
                v10 = (i + 1) * (n + 1) + j
                v01 = i * (n + 1) + (j + 1)
                v11 = (i + 1) * (n + 1) + (j + 1)

                if sign > 0:
                    faces.append([v00 + 1, v10 + 1, v11 + 1, np.nan])
                    faces.append([v00 + 1, v11 + 1, v01 + 1, np.nan])
                else:
                    faces.append([v00 + 1, v11 + 1, v10 + 1, np.nan])
                    faces.append([v00 + 1, v01 + 1, v11 + 1, np.nan])

        return vertices, np.array(faces)

    def _apply_rounding(self,
            vertices: np.ndarray) -> np.ndarray:

        if self.rounding <= 0:
            return vertices

        h = self.half_size
        r = self.r

        edge_threshold = h - r
        rounded_verts = vertices.copy()

        for i, v in enumerate(vertices):
            x, y, z = v

            near_x = abs(abs(x) - h) < 1e-10
            near_y = abs(abs(y) - h) < 1e-10
            near_z = abs(abs(z) - h) < 1e-10
            n_near = near_x + near_y + near_z

            if n_near == 1:
                if not near_x:
                    if abs(y) > edge_threshold or abs(z) > edge_threshold:
                        rounded_verts[i] = self._round_edge_vertex(v, r, h)
                elif not near_y:
                    if abs(x) > edge_threshold or abs(z) > edge_threshold:
                        rounded_verts[i] = self._round_edge_vertex(v, r, h)
                else:
                    if abs(x) > edge_threshold or abs(y) > edge_threshold:
                        rounded_verts[i] = self._round_edge_vertex(v, r, h)
            elif n_near >= 2:
                rounded_verts[i] = self._round_corner_vertex(v, r, h)

        return rounded_verts

    def _round_edge_vertex(self,
            v: np.ndarray,
            r: float,
            h: float) -> np.ndarray:

        x, y, z = v

        if abs(abs(x) - h) < 1e-10:
            new_y, new_z = self._round_2d(y, z, r, h)
            return np.array([x, new_y, new_z])
        elif abs(abs(y) - h) < 1e-10:
            new_x, new_z = self._round_2d(x, z, r, h)
            return np.array([new_x, y, new_z])
        else:
            new_x, new_y = self._round_2d(x, y, r, h)
            return np.array([new_x, new_y, z])

    def _round_2d(self,
            u: float,
            v: float,
            r: float,
            h: float) -> Tuple[float, float]:

        edge_h = h - r
        new_u, new_v = u, v
        in_corner_u = abs(u) > edge_h
        in_corner_v = abs(v) > edge_h

        if in_corner_u and in_corner_v:
            su = np.sign(u)
            sv = np.sign(v)
            lu = abs(u) - edge_h
            lv = abs(v) - edge_h
            dist = np.sqrt(lu ** 2 + lv ** 2)
            if dist > 0:
                new_u = su * (edge_h + r * lu / dist)
                new_v = sv * (edge_h + r * lv / dist)
        elif in_corner_u:
            su = np.sign(u)
            new_u = su * h
        elif in_corner_v:
            sv = np.sign(v)
            new_v = sv * h

        return new_u, new_v

    def _round_corner_vertex(self,
            v: np.ndarray,
            r: float,
            h: float) -> np.ndarray:

        x, y, z = v
        edge_h = h - r

        sx = np.sign(x) if abs(x) > edge_h else 0
        sy = np.sign(y) if abs(y) > edge_h else 0
        sz = np.sign(z) if abs(z) > edge_h else 0

        if sx != 0 and sy != 0 and sz != 0:
            cx = sx * edge_h
            cy = sy * edge_h
            cz = sz * edge_h
            dx = x - cx
            dy = y - cy
            dz = z - cz
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if dist > 0:
                return np.array([
                    cx + r * dx / dist,
                    cy + r * dy / dist,
                    cz + r * dz / dist])
        elif sx != 0 and sy != 0:
            new_x, new_y = self._round_2d(x, y, r, h)
            return np.array([new_x, new_y, z])
        elif sx != 0 and sz != 0:
            new_x, new_z = self._round_2d(x, z, r, h)
            return np.array([new_x, y, new_z])
        elif sy != 0 and sz != 0:
            new_y, new_z = self._round_2d(y, z, r, h)
            return np.array([x, new_y, new_z])

        return v

    def _merge_vertices(self,
            vertices: np.ndarray,
            faces: np.ndarray,
            tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:

        n = len(vertices)

        unique_verts = []
        index_map = np.zeros(n, dtype = int)

        for i, v in enumerate(vertices):
            found = False
            for j, uv in enumerate(unique_verts):
                if np.linalg.norm(v - uv) < tol:
                    index_map[i] = j + 1  # 1-indexed
                    found = True
                    break
            if not found:
                unique_verts.append(v)
                index_map[i] = len(unique_verts)  # 1-indexed

        new_faces = faces.copy()
        for i in range(len(faces)):
            for j in range(3):
                old_idx = int(faces[i, j])
                new_faces[i, j] = index_map[old_idx - 1]

        return np.array(unique_verts), new_faces


# ============================================================================
# Geometry Generator
# ============================================================================

class GeometryGenerator(object):

    def __init__(self,
            config: Dict[str, Any],
            verbose: bool = False) -> None:

        self.config = config
        self.verbose = verbose
        self.structure = config['structure']

    def generate(self) -> List[Particle]:

        structure_map = {
            'sphere': self._generate_sphere,
            'cube': self._generate_cube,
            'rod': self._generate_rod,
            'ellipsoid': self._generate_ellipsoid,
            'triangle': self._generate_triangle,
            'dimer_sphere': self._generate_dimer_sphere,
            'dimer_cube': self._generate_dimer_cube,
            'core_shell_sphere': self._generate_core_shell_sphere,
            'core_shell_cube': self._generate_core_shell_cube,
            'core_shell_rod': self._generate_core_shell_rod,
            'dimer_core_shell_cube': self._generate_dimer_core_shell_cube,
            'advanced_dimer_cube': self._generate_advanced_dimer_cube,
            'connected_dimer_cube': self._generate_connected_dimer_cube,
            'advanced_monomer_cube': self._generate_advanced_monomer_cube,
            'sphere_cluster_aggregate': self._generate_sphere_cluster_aggregate,
            'from_shape': self._generate_from_shape,
        }

        assert self.structure in structure_map, \
            '[error] Unknown structure type: <{}>'.format(self.structure)

        particles = structure_map[self.structure]()
        return particles

    # ====================================================================
    # Mesh parameter conversion helpers
    # ====================================================================

    def _is_legacy_mesh_mode(self) -> bool:
        return any(key in self.config for key in ['nphi', 'ntheta', 'nz'])

    def _element_size_to_n_rod(self,
            element_size: float,
            diameter: float,
            height: float) -> List[int]:

        nphi = max(8, int(np.ceil(np.pi * diameter / element_size)))
        ntheta = max(6, int(np.ceil(0.5 * diameter / element_size)))
        cylinder_length = height - diameter
        nz = max(4, int(np.ceil(cylinder_length / element_size)))

        result = [nphi, ntheta, nz]
        if self.verbose:
            print('  element_size={}nm -> [{}, {}, {}]'.format(
                element_size, nphi, ntheta, nz))
            print('    (actual: phi={:.2f}nm, theta={:.2f}nm, z={:.2f}nm)'.format(
                np.pi * diameter / nphi,
                0.5 * diameter / ntheta,
                cylinder_length / nz))
        return result

    def _legacy_mesh_to_n_rod(self,
            nphi_param: float,
            ntheta_param: float,
            nz_param: float,
            diameter: float,
            height: float) -> List[int]:

        nphi = max(8, int(np.ceil((diameter + 1) * np.pi / nphi_param)))
        ntheta = max(6, int(np.ceil((diameter + 1) / ntheta_param)))
        nz = max(4, int(np.ceil((height - diameter + 1) / nz_param)))

        result = [nphi, ntheta, nz]
        if self.verbose:
            print('  legacy mode: nphi={}, ntheta={}, nz={}'.format(
                nphi_param, ntheta_param, nz_param))
            print('  -> [{}, {}, {}]'.format(nphi, ntheta, nz))
        return result

    def _element_size_to_n_sphere(self,
            element_size: float,
            diameter: float) -> int:

        surface_area = np.pi * diameter ** 2
        target_faces = surface_area / (element_size ** 2)
        target_vertices = int(target_faces / 2)

        available = [
            32, 60, 144, 169, 225, 256, 289, 324, 361, 400,
            441, 484, 529, 576, 625, 676, 729, 784, 841, 900,
            961, 1024, 1089, 1156, 1225, 1296, 1369, 1444, 1521, 1600]

        closest = min(available, key = lambda x: abs(x - target_vertices))

        if self.verbose:
            print('  element_size={}nm, diameter={}nm'.format(
                element_size, diameter))
            print('  target_vertices={} -> trisphere({})'.format(
                target_vertices, closest))
        return closest

    def _element_size_to_n_cube(self,
            element_size: float,
            size: float) -> int:

        n = max(4, int(np.ceil(size / element_size)))

        if self.verbose:
            actual_element = size / n
            print('  element_size={}nm, size={}nm -> n={} (actual: {:.2f}nm)'.format(
                element_size, size, n, actual_element))
        return n

    # ====================================================================
    # Built-in Structures
    # ====================================================================

    def _generate_sphere(self) -> List[Particle]:

        diameter = self.config.get('diameter', 10)
        use_mirror = self.config.get('use_mirror_symmetry', False)

        if use_mirror:
            return self._generate_sphere_segment(diameter, use_mirror)

        if self._is_legacy_mesh_mode():
            nphi = self.config.get('nphi', 4)
            target_vertices = int(
                np.ceil((diameter + 1) * np.pi / nphi) ** 2 / 2)
            available = [
                32, 60, 144, 169, 225, 256, 289, 324, 361, 400,
                441, 484, 529, 576, 625, 676, 729, 784, 841, 900]
            mesh = min(available, key = lambda x: abs(x - target_vertices))

            if self.verbose:
                print('[info] Sphere (legacy mode): nphi={} -> trisphere({})'.format(
                    nphi, mesh))
        else:
            element_size = self.config.get('mesh_density', 2.0)
            mesh = self._element_size_to_n_sphere(element_size, diameter)

        p = trisphere(mesh, diameter)
        return [p]

    def _generate_sphere_segment(self,
            diameter: float,
            sym: Any) -> List[Particle]:
        """Generate partial sphere mesh for mirror symmetry.

        For 'x' or 'y' mirror: hemisphere.
        For 'xy' mirror: quarter sphere.
        """
        if isinstance(sym, str):
            sym_key = sym
        else:
            sym_key = 'xy'

        # Determine phi range based on symmetry
        if sym_key == 'x':
            # Hemisphere: x >= 0 -> phi in [-pi/2, pi/2]
            phi_range = (-np.pi / 2, np.pi / 2)
        elif sym_key == 'y':
            # Hemisphere: y >= 0 -> phi in [0, pi]
            phi_range = (0, np.pi)
        elif sym_key == 'xy':
            # Quarter sphere: x >= 0, y >= 0 -> phi in [0, pi/2]
            phi_range = (0, np.pi / 2)
        else:
            phi_range = (0, 2 * np.pi)

        # Determine mesh resolution
        if self._is_legacy_mesh_mode():
            nphi = self.config.get('nphi', 4)
            n_phi = max(8, int(np.ceil(
                (diameter + 1) * np.pi / nphi * (phi_range[1] - phi_range[0]) / (2 * np.pi))))
            n_theta = max(8, int(np.ceil((diameter + 1) * np.pi / nphi)))
        else:
            element_size = self.config.get('mesh_density', 2.0)
            arc_phi = diameter / 2 * (phi_range[1] - phi_range[0])
            arc_theta = diameter / 2 * np.pi
            n_phi = max(8, int(np.ceil(arc_phi / element_size)))
            n_theta = max(8, int(np.ceil(arc_theta / element_size)))

        phi = np.linspace(phi_range[0], phi_range[1], n_phi + 1)
        theta = np.linspace(0.01, np.pi - 0.01, n_theta + 1)

        p = trispheresegment(phi, theta, diameter)

        if self.verbose:
            print('[info] Sphere segment (sym={}): n_phi={}, n_theta={}, {} faces'.format(
                sym_key, n_phi, n_theta, p.nfaces))

        return [p]

    def _generate_cube(self) -> List[Particle]:

        size = self.config.get('size', 20)
        rounding = self.config.get('rounding', 0.25)
        element_size = self.config.get('mesh_density', 4.0)

        mesh = self._element_size_to_n_cube(element_size, size)

        p = tricube(mesh, size, e = rounding)
        return [p]

    def _generate_rod(self) -> List[Particle]:

        diameter = self.config.get('diameter', 10)
        height = self.config.get('height', 50)

        if 'rod_mesh' in self.config:
            n = self.config['rod_mesh']
            nphi, ntheta, nz = n
        elif self._is_legacy_mesh_mode():
            nphi_param = self.config.get('nphi', 4)
            ntheta_param = self.config.get('ntheta', 4)
            nz_param = self.config.get('nz', 4)
            n = self._legacy_mesh_to_n_rod(
                nphi_param, ntheta_param, nz_param, diameter, height)
        else:
            element_size = self.config.get('mesh_density', 2.0)
            n = self._element_size_to_n_rod(element_size, diameter, height)

        interp = self.config.get('interp', 'flat')
        p = trirod(diameter, height, n = n, triangles = True)
        if interp == 'curv':
            p.interp = 'curv'
            p._norm()
        # Rotate 90 degrees to lie along x-axis
        p.rot(90, [0, 1, 0])
        return [p]

    def _generate_ellipsoid(self) -> List[Particle]:

        axes = self.config.get('axes', [10, 15, 20])
        element_size = self.config.get('mesh_density', 2.0)

        avg_diameter = 2 * (axes[0] * axes[1] * axes[2]) ** (1.0 / 3.0)
        mesh = self._element_size_to_n_sphere(element_size, avg_diameter)

        p = trisphere(mesh, 1.0)
        # Scale each axis independently
        p.scale(np.array(axes))
        return [p]

    def _generate_triangle(self) -> List[Particle]:

        side_length = self.config.get('side_length', 30)
        thickness = self.config.get('thickness', 5)

        poly = Polygon(3, size = [side_length, side_length * 2 / np.sqrt(3)])
        # Round polygon positions
        poly.pos = np.round(poly.pos)
        edge = EdgeProfile(thickness, 11)
        p = tripolygon(poly, edge)
        return [p]

    def _generate_dimer_sphere(self) -> List[Particle]:

        diameter = self.config.get('diameter', 10)
        gap = self.config.get('gap', 5)

        if self._is_legacy_mesh_mode():
            nphi = self.config.get('nphi', 4)
            target_vertices = int(
                np.ceil((diameter + 1) * np.pi / nphi) ** 2 / 2)
            available = [
                32, 60, 144, 169, 225, 256, 289, 324, 361, 400,
                441, 484, 529, 576, 625, 676, 729, 784, 841, 900]
            mesh = min(available, key = lambda x: abs(x - target_vertices))

            if self.verbose:
                print('[info] Dimer sphere (legacy mode): nphi={} -> trisphere({})'.format(
                    nphi, mesh))
        else:
            element_size = self.config.get('mesh_density', 2.0)
            mesh = self._element_size_to_n_sphere(element_size, diameter)

        shift_distance = (diameter + gap) / 2

        p1 = trisphere(mesh, diameter)
        p1.shift([-shift_distance, 0, 0])

        p2 = trisphere(mesh, diameter)
        p2.shift([shift_distance, 0, 0])

        return [p1, p2]

    def _generate_dimer_cube(self) -> List[Particle]:

        size = self.config.get('size', 20)
        gap = self.config.get('gap', 10)
        rounding = self.config.get('rounding', 0.25)
        element_size = self.config.get('mesh_density', 4.0)

        mesh = self._element_size_to_n_cube(element_size, size)
        shift_distance = (size + gap) / 2

        p1 = tricube(mesh, size, e = rounding)
        p1.shift([-shift_distance, 0, 0])

        p2 = tricube(mesh, size, e = rounding)
        p2.shift([shift_distance, 0, 0])

        return [p1, p2]

    def _generate_core_shell_sphere(self) -> List[Particle]:

        core_diameter = self.config.get('core_diameter', 10)
        shell_thickness = self.config.get('shell_thickness', 5)
        element_size = self.config.get('mesh_density', 2.0)
        shell_diameter = core_diameter + 2 * shell_thickness

        mesh_core = self._element_size_to_n_sphere(element_size, core_diameter)
        mesh_shell = self._element_size_to_n_sphere(element_size, shell_diameter)

        p_core = trisphere(mesh_core, core_diameter)
        p_shell = trisphere(mesh_shell, shell_diameter)

        return [p_core, p_shell]

    def _generate_core_shell_cube(self) -> List[Particle]:

        core_size = self.config.get('core_size')
        shell_thickness = self.config.get('shell_thickness')
        rounding = self.config.get('rounding')
        element_size = self.config.get('mesh_density', 4.0)
        shell_size = core_size + 2 * shell_thickness

        mesh_core = self._element_size_to_n_cube(element_size, core_size)
        mesh_shell = self._element_size_to_n_cube(element_size, shell_size)

        p_core = tricube(mesh_core, core_size, e = rounding)
        p_shell = tricube(mesh_shell, shell_size, e = rounding)

        return [p_core, p_shell]

    def _generate_core_shell_rod(self) -> List[Particle]:

        core_diameter = self.config.get('core_diameter', 15)
        shell_thickness = self.config.get('shell_thickness', 5)
        height = self.config.get('height', 80)

        shell_diameter = core_diameter + 2 * shell_thickness
        shell_height = height
        core_height = height - 2 * shell_thickness

        if 'rod_mesh' in self.config:
            n_core = self.config['rod_mesh']
            n_shell = self.config['rod_mesh']
            assert len(n_core) == 3, \
                '[error] <rod_mesh> must have 3 values [nphi, ntheta, nz]'
        elif self._is_legacy_mesh_mode():
            nphi_param = self.config.get('nphi', 4)
            ntheta_param = self.config.get('ntheta', 4)
            nz_param = self.config.get('nz', 4)
            n_core = self._legacy_mesh_to_n_rod(
                nphi_param, ntheta_param, nz_param, core_diameter, core_height)
            n_shell = self._legacy_mesh_to_n_rod(
                nphi_param, ntheta_param, nz_param, shell_diameter, shell_height)
        else:
            element_size = self.config.get('mesh_density', 2.0)
            n_core = self._element_size_to_n_rod(
                element_size, core_diameter, core_height)
            n_shell = self._element_size_to_n_rod(
                element_size, shell_diameter, shell_height)

        interp = self.config.get('interp', 'flat')
        p_core = trirod(core_diameter, core_height, n = n_core, triangles = True)
        p_shell = trirod(shell_diameter, shell_height, n = n_shell, triangles = True)
        if interp == 'curv':
            p_core.interp = 'curv'
            p_core._norm()
            p_shell.interp = 'curv'
            p_shell._norm()

        # Rotate 90 degrees to lie along x-axis
        p_core.rot(90, [0, 1, 0])
        p_shell.rot(90, [0, 1, 0])

        return [p_core, p_shell]

    def _generate_dimer_core_shell_cube(self) -> List[Particle]:

        core_size = self.config.get('core_size', 20)
        shell_thickness = self.config.get('shell_thickness', 5)
        gap = self.config.get('gap', 10)
        rounding = self.config.get('rounding', 0.25)
        element_size = self.config.get('mesh_density', 4.0)
        shell_size = core_size + 2 * shell_thickness

        mesh_core = self._element_size_to_n_cube(element_size, core_size)
        mesh_shell = self._element_size_to_n_cube(element_size, shell_size)
        shift_distance = (shell_size + gap) / 2

        # Particle 1 (Left)
        core1 = tricube(mesh_core, core_size, e = rounding)
        core1.shift([-shift_distance, 0, 0])

        shell1 = tricube(mesh_shell, shell_size, e = rounding)
        shell1.shift([-shift_distance, 0, 0])

        # Particle 2 (Right)
        core2 = tricube(mesh_core, core_size, e = rounding)
        core2.shift([shift_distance, 0, 0])

        shell2 = tricube(mesh_shell, shell_size, e = rounding)
        shell2.shift([shift_distance, 0, 0])

        return [core1, shell1, core2, shell2]

    def _generate_advanced_dimer_cube(self) -> List[Particle]:

        core_size = self.config.get('core_size', 30)
        shell_layers = self.config.get('shell_layers', [])
        materials = self.config.get('materials', [])
        element_size = self.config.get('mesh_density', 2.0)

        assert len(materials) == 1 + len(shell_layers), \
            '[error] <materials> length ({}) must equal 1 (core) + {} (shells) = {}'.format(
                len(materials), len(shell_layers), 1 + len(shell_layers))

        if 'roundings' in self.config:
            roundings = self.config.get('roundings')
            assert len(roundings) == len(materials), \
                '[error] <roundings> length ({}) must equal <materials> length ({})'.format(
                    len(roundings), len(materials))
        elif 'rounding' in self.config:
            roundings = [self.config.get('rounding', 0.25)] * len(materials)
        else:
            roundings = [0.25] * len(materials)

        gap = self.config.get('gap', 10)
        offset = self.config.get('offset', [0, 0, 0])
        tilt_angle = self.config.get('tilt_angle', 0)
        tilt_axis = self.config.get('tilt_axis', [0, 1, 0])
        rotation_angle = self.config.get('rotation_angle', 0)

        sizes = [core_size]
        for thickness in shell_layers:
            sizes.append(sizes[-1] + 2 * thickness)

        total_size = sizes[-1]
        shift_distance = (total_size + gap) / 2

        mesh_subdivs = [
            self._element_size_to_n_cube(element_size, s) for s in sizes]

        if self.verbose:
            print('[info] Advanced dimer cube mesh (element_size={}nm):'.format(
                element_size))
            for i, (s, m) in enumerate(zip(sizes, mesh_subdivs)):
                layer_name = 'core' if i == 0 else 'shell{}'.format(i)
                print('  {}: size={}nm -> n={}'.format(layer_name, s, m))

        particles = []

        # Particle 1 (Left)
        for i, (s, rounding, m) in enumerate(
                zip(sizes, roundings, mesh_subdivs)):
            p = tricube(m, s, e = rounding)
            p.shift([-shift_distance, 0, 0])
            particles.append(p)

        # Particle 2 (Right with transformations)
        for i, (s, rounding, m) in enumerate(
                zip(sizes, roundings, mesh_subdivs)):
            p = tricube(m, s, e = rounding)
            p.rot(rotation_angle, [0, 0, 1])
            p.rot(tilt_angle, tilt_axis)
            p.shift([shift_distance, 0, 0])
            p.shift(offset)
            particles.append(p)

        return particles

    def _generate_connected_dimer_cube(self) -> List[Particle]:

        gap = self.config.get('gap', 0)
        shell_layers = self.config.get('shell_layers', [])

        assert gap <= 0, \
            ('[error] connected_dimer_cube requires gap <= 0 (got gap={}). '
             'Use "advanced_dimer_cube" for gap > 0.'.format(gap))

        is_core_shell = len(shell_layers) > 0

        if is_core_shell:
            return self._connected_dimer_cube_core_shell()
        else:
            return self._connected_dimer_cube_single()

    def _connected_dimer_cube_single(self) -> List[Particle]:

        core_size = self.config.get('core_size', 30)
        element_size = self.config.get('mesh_density', 2.0)
        rounding = self.config.get('rounding', 0.25)
        gap = self.config.get('gap', 0)
        offset = self.config.get('offset', [0, 0, 0])
        tilt_angle = self.config.get('tilt_angle', 0)
        tilt_axis = self.config.get('tilt_axis', [0, 1, 0])
        rotation_angle = self.config.get('rotation_angle', 0)

        mesh_n = self._element_size_to_n_cube(element_size, core_size)
        shift_distance = (core_size + gap) / 2

        if self.verbose:
            print('[info] Connected dimer cube (single material):')
            print('  core_size={}nm, gap={}nm, rounding={}'.format(
                core_size, gap, rounding))
            print('  mesh: n={} (~{:.2f}nm elements)'.format(
                mesh_n, core_size / mesh_n))

        p1 = tricube(mesh_n, core_size, e = rounding)
        p1.shift([-shift_distance, 0, 0])

        p2 = tricube(mesh_n, core_size, e = rounding)
        p2.rot(rotation_angle, [0, 0, 1])
        p2.rot(tilt_angle, tilt_axis)
        p2.shift([shift_distance, 0, 0])
        p2.shift(offset)

        p_fused = self._fuse_two_particles(p1, p2)

        if self.verbose:
            print('[info] Fused mesh: {} vertices, {} faces'.format(
                p_fused.nverts, p_fused.nfaces))

        return [p_fused]

    def _connected_dimer_cube_core_shell(self) -> List[Particle]:

        core_size = self.config.get('core_size', 30)
        shell_layers = self.config.get('shell_layers', [5])
        materials = self.config.get('materials', ['gold', 'silver'])
        element_size = self.config.get('mesh_density', 2.0)
        gap = self.config.get('gap', 0)
        offset = self.config.get('offset', [0, 0, 0])
        tilt_angle = self.config.get('tilt_angle', 0)
        tilt_axis = self.config.get('tilt_axis', [0, 1, 0])
        rotation_angle = self.config.get('rotation_angle', 0)

        assert len(shell_layers) == 1, \
            '[error] connected_dimer_cube core-shell requires 1 shell layer, got {}'.format(
                len(shell_layers))
        assert len(materials) == 2, \
            '[error] connected_dimer_cube core-shell requires 2 materials, got {}'.format(
                len(materials))

        shell_thickness = shell_layers[0]
        shell_size = core_size + 2 * shell_thickness

        core_gap = gap + 2 * shell_thickness
        fuse_cores = core_gap <= 0

        if 'roundings' in self.config:
            roundings = self.config.get('roundings')
            assert len(roundings) == 2, \
                '[error] <roundings> must have 2 values, got {}'.format(len(roundings))
            core_rounding = roundings[0]
            shell_rounding = roundings[1]
        else:
            single_rounding = self.config.get('rounding', 0.25)
            core_rounding = single_rounding
            shell_rounding = single_rounding

        core_mesh_n = self._element_size_to_n_cube(element_size, core_size)
        shell_mesh_n = self._element_size_to_n_cube(element_size, shell_size)

        shell_shift = (shell_size + gap) / 2
        core_shift = (core_size + core_gap) / 2

        if self.verbose:
            print('[info] Connected dimer cube (core-shell mode):')
            print('  core_size={}nm, shell_thickness={}nm'.format(
                core_size, shell_thickness))
            print('  core_gap={}nm -> {}'.format(
                core_gap, 'FUSE CORES' if fuse_cores else 'SEPARATE CORES'))

        # Fuse shells
        s1 = tricube(shell_mesh_n, shell_size, e = shell_rounding)
        s1.shift([-shell_shift, 0, 0])

        s2 = tricube(shell_mesh_n, shell_size, e = shell_rounding)
        s2.rot(rotation_angle, [0, 0, 1])
        s2.rot(tilt_angle, tilt_axis)
        s2.shift([shell_shift, 0, 0])
        s2.shift(offset)

        p_fused_shell = self._fuse_two_particles(s1, s2)

        if fuse_cores:
            c1 = tricube(core_mesh_n, core_size, e = core_rounding)
            c1.shift([-core_shift, 0, 0])

            c2 = tricube(core_mesh_n, core_size, e = core_rounding)
            c2.rot(rotation_angle, [0, 0, 1])
            c2.rot(tilt_angle, tilt_axis)
            c2.shift([core_shift, 0, 0])
            c2.shift(offset)

            p_fused_core = self._fuse_two_particles(c1, c2)
            return [p_fused_core, p_fused_shell]
        else:
            p_core1 = tricube(core_mesh_n, core_size, e = core_rounding)
            p_core1.shift([-core_shift, 0, 0])

            p_core2 = tricube(core_mesh_n, core_size, e = core_rounding)
            p_core2.rot(rotation_angle, [0, 0, 1])
            p_core2.rot(tilt_angle, tilt_axis)
            p_core2.shift([core_shift, 0, 0])
            p_core2.shift(offset)

            return [p_core1, p_core2, p_fused_shell]

    def _generate_advanced_monomer_cube(self) -> List[Particle]:

        core_size = self.config.get('core_size', 30)
        shell_layers = self.config.get('shell_layers', [])
        materials = self.config.get('materials', [])
        element_size = self.config.get('mesh_density', 2.0)

        assert len(materials) == 1 + len(shell_layers), \
            '[error] <materials> length ({}) must equal 1 (core) + {} (shells) = {}'.format(
                len(materials), len(shell_layers), 1 + len(shell_layers))

        if 'roundings' in self.config:
            roundings = self.config.get('roundings')
            assert len(roundings) == len(materials), \
                '[error] <roundings> length ({}) must equal <materials> length ({})'.format(
                    len(roundings), len(materials))
        elif 'rounding' in self.config:
            roundings = [self.config.get('rounding', 0.25)] * len(materials)
        else:
            roundings = [0.25] * len(materials)

        sizes = [core_size]
        for thickness in shell_layers:
            sizes.append(sizes[-1] + 2 * thickness)

        mesh_subdivs = [
            self._element_size_to_n_cube(element_size, s) for s in sizes]

        if self.verbose:
            print('[info] Advanced monomer cube mesh (element_size={}nm):'.format(
                element_size))
            for i, (s, m) in enumerate(zip(sizes, mesh_subdivs)):
                layer_name = 'core' if i == 0 else 'shell{}'.format(i)
                print('  {}: size={}nm -> n={}'.format(layer_name, s, m))

        particles = []
        for i, (s, rounding, m) in enumerate(
                zip(sizes, roundings, mesh_subdivs)):
            p = tricube(m, s, e = rounding)
            particles.append(p)

        return particles

    def _generate_sphere_cluster_aggregate(self) -> List[Particle]:

        n_spheres = self.config.get('n_spheres', 1)
        diameter = self.config.get('diameter', 50)
        gap = self.config.get('gap', -0.1)

        element_size = self.config.get('mesh_density', 2.0)
        mesh = self._element_size_to_n_sphere(element_size, diameter)

        spacing = diameter + gap

        # 60-degree triangle height
        dy_60deg = spacing * 0.866025404  # sin(60) = sqrt(3)/2

        # Hexagonal surrounding positions
        hex_positions = []
        for i in range(6):
            angle = i * 60 * np.pi / 180
            x = spacing * np.cos(angle)
            y = spacing * np.sin(angle)
            hex_positions.append((x, y))

        cluster_positions = {
            1: [(0, 0)],
            2: [(-spacing / 2, 0), (spacing / 2, 0)],
            3: [(-spacing / 2, 0), (spacing / 2, 0), (0, dy_60deg)],
            4: [(0, 0)] + hex_positions[0:3],
            5: [(0, 0)] + hex_positions[0:4],
            6: [(0, 0)] + hex_positions[0:5],
            7: [(0, 0)] + hex_positions[0:6],
        }

        assert n_spheres in cluster_positions, \
            '[error] <n_spheres> must be 1-7, got {}'.format(n_spheres)

        positions = cluster_positions[n_spheres]

        particles = []
        for i, (x, y) in enumerate(positions):
            p = trisphere(mesh, diameter)
            p.shift([x, y, 0])
            particles.append(p)

            if self.verbose:
                print('[info] Sphere {}: ({:.2f}, {:.2f}, 0) nm'.format(
                    i + 1, x, y))

        return particles

    def _generate_from_shape(self) -> List[Particle]:

        shape_file = self.config.get('shape_file')
        assert shape_file is not None, \
            '[error] <shape_file> must be specified for "from_shape" structure'

        voxel_size = self.config.get('voxel_size', 1.0)
        method = self.config.get('voxel_method', 'surface')

        if self.verbose:
            print('[info] Loading DDA shape file...')

        loader = ShapeFileLoader(
            shape_file, voxel_size = voxel_size,
            method = method, verbose = self.verbose)
        particles = loader.generate()

        return particles

    # ====================================================================
    # Mesh Fusion Utility
    # ====================================================================

    def _fuse_two_particles(self,
            p1: Particle,
            p2: Particle) -> Particle:

        verts1 = p1.verts
        faces1 = p1.faces
        verts2 = p2.verts
        faces2 = p2.faces

        if self.verbose:
            print('  Particle 1: {} vertices, {} faces'.format(
                len(verts1), len(faces1)))
            print('  Particle 2: {} vertices, {} faces'.format(
                len(verts2), len(faces2)))

        # Detect internal faces
        inside1 = self._detect_internal_faces(
            verts1, faces1, verts2, faces2)
        inside2 = self._detect_internal_faces(
            verts2, faces2, verts1, faces1)

        if self.verbose:
            print('  Faces of p1 inside p2: {}'.format(np.sum(inside1)))
            print('  Faces of p2 inside p1: {}'.format(np.sum(inside2)))

        keep1 = ~inside1
        keep2 = ~inside2

        faces1_kept = faces1[keep1]
        faces2_kept = faces2[keep2]

        # Merge into single mesh
        n_verts1 = len(verts1)
        merged_verts = np.vstack([verts1, verts2])

        # Offset face indices for mesh2
        faces2_offset = faces2_kept.copy()
        valid_mask = ~np.isnan(faces2_offset)
        faces2_offset[valid_mask] = faces2_offset[valid_mask] + n_verts1

        merged_faces = np.vstack([faces1_kept, faces2_offset])

        # Clean up: remove unused vertices
        used_verts_flat = merged_faces[~np.isnan(merged_faces)].astype(int)
        unique_vert_idx = np.unique(used_verts_flat)

        vert_map = np.zeros(len(merged_verts), dtype = int)
        vert_map[unique_vert_idx] = np.arange(len(unique_vert_idx))

        clean_faces = merged_faces.copy()
        valid = ~np.isnan(clean_faces)
        clean_faces[valid] = vert_map[clean_faces[valid].astype(int)]

        clean_verts = merged_verts[unique_vert_idx]

        if self.verbose:
            print('  Fused mesh: {} vertices, {} faces'.format(
                len(clean_verts), len(clean_faces)))

        p_fused = Particle(clean_verts, clean_faces, interp = 'flat')
        return p_fused

    def _detect_internal_faces(self,
            verts_a: np.ndarray,
            faces_a: np.ndarray,
            verts_b: np.ndarray,
            faces_b: np.ndarray) -> np.ndarray:

        n_faces = len(faces_a)
        inside = np.zeros(n_faces, dtype = bool)

        # Compute centroids of faces_a
        centroids = np.zeros((n_faces, 3))
        for i in range(n_faces):
            face = faces_a[i]
            valid_idx = face[~np.isnan(face)].astype(int)
            centroids[i] = np.mean(verts_a[valid_idx], axis = 0)

        # Bounding box of mesh_b
        margin = 0.1
        bbox_min = np.min(verts_b, axis = 0) - margin
        bbox_max = np.max(verts_b, axis = 0) + margin

        for i in range(n_faces):
            pt = centroids[i]
            if np.all(pt >= bbox_min) and np.all(pt <= bbox_max):
                inside[i] = self._point_in_mesh(pt, verts_b, faces_b)

        return inside

    @staticmethod
    def _point_in_mesh(
            point: np.ndarray,
            verts: np.ndarray,
            faces: np.ndarray) -> bool:

        ray_dirs = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.577, 0.577, 0.577],
            [-0.577, 0.577, 0.577],
            [0.577, -0.577, 0.577],
            [0.707, 0.707, 0],
        ])

        inside_votes = 0
        total_votes = len(ray_dirs)

        for ray_dir in ray_dirs:
            intersect_count = 0

            for fi in range(len(faces)):
                face_idx = faces[fi]
                face_idx = face_idx[~np.isnan(face_idx)].astype(int)

                if len(face_idx) < 3:
                    continue

                v0 = verts[face_idx[0]]
                v1 = verts[face_idx[1]]
                v2 = verts[face_idx[2]]

                if GeometryGenerator._ray_triangle_intersect(
                        point, ray_dir, v0, v1, v2):
                    intersect_count += 1

                if len(face_idx) == 4:
                    v2b = verts[face_idx[2]]
                    v3 = verts[face_idx[3]]
                    if GeometryGenerator._ray_triangle_intersect(
                            point, ray_dir, v0, v2b, v3):
                        intersect_count += 1

            if intersect_count % 2 == 1:
                inside_votes += 1

        return inside_votes > (total_votes / 2)

    @staticmethod
    def _ray_triangle_intersect(
            ray_origin: np.ndarray,
            ray_dir: np.ndarray,
            v0: np.ndarray,
            v1: np.ndarray,
            v2: np.ndarray) -> bool:

        EPSILON = 1e-10

        edge1 = v1 - v0
        edge2 = v2 - v0

        h = np.cross(ray_dir, edge2)
        a = np.dot(edge1, h)

        if abs(a) < EPSILON:
            return False

        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return False

        q = np.cross(s, edge1)
        v = f * np.dot(ray_dir, q)

        if v < 0.0 or u + v > 1.0:
            return False

        t = f * np.dot(edge2, q)
        return t > EPSILON

    # ====================================================================
    # Rotation utility (for connected dimer transforms)
    # ====================================================================

    @staticmethod
    def _rotate_vertices(
            vertices: np.ndarray,
            angle_deg: float,
            axis: List[float]) -> np.ndarray:

        rad = np.radians(angle_deg)
        axis_arr = np.array(axis, dtype = float)
        axis_arr = axis_arr / np.linalg.norm(axis_arr)

        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        rotated = np.zeros_like(vertices)
        for i, v in enumerate(vertices):
            rotated[i] = (
                v * cos_a
                + np.cross(axis_arr, v) * sin_a
                + axis_arr * np.dot(axis_arr, v) * (1 - cos_a))
        return rotated
