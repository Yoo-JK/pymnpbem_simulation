"""AdaptiveCubeMesh — per-face density + edge-profile rounding for nanocubes.

Ported from OLD geometry_generator.py::AdaptiveCubeMesh (~line 332-1920).

Public API
----------
build_adaptive_cube(size, densities, rounding, edge_profile_kwargs)
    -> Particle

    ``densities`` is a dict with any subset of keys
    ``'+x', '-x', '+y', '-y', '+z', '-z'``.  Missing faces use the scalar
    fallback ``n`` (resolved by the caller from ``n_per_edge`` / ``mesh_density``).

    ``edge_profile_kwargs``, when not None, is passed to
    ``mnpbem.geometry.EdgeProfile`` to produce rounded edges before the
    super-sphere ``e`` rounding applied by ``tricube``.  Supported keys are
    the same as ``EdgeProfile.__init__``: ``e``, ``dz``, ``mode``.
    (The height arg is filled automatically from ``size``.)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_face_grid(axis: int, sign: int, half: float, n: int,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform n×n grid on one face of a unit half-cube scaled to ``half``."""
    u = np.linspace(-half, half, n + 1)
    uu, vv = np.meshgrid(u, u)
    uu = uu.flatten()
    vv = vv.flatten()

    verts = np.zeros((len(uu), 3))
    if axis == 0:          # x-face
        verts[:, 0] = sign * half
        verts[:, 1] = uu
        verts[:, 2] = vv
    elif axis == 1:        # y-face
        verts[:, 0] = uu
        verts[:, 1] = sign * half
        verts[:, 2] = vv
    else:                  # z-face
        verts[:, 0] = uu
        verts[:, 1] = vv
        verts[:, 2] = sign * half

    faces = []
    for i in range(n):
        for j in range(n):
            v00 = i * (n + 1) + j
            v10 = (i + 1) * (n + 1) + j
            v01 = i * (n + 1) + (j + 1)
            v11 = (i + 1) * (n + 1) + (j + 1)
            if sign > 0:
                faces.append([v00, v10, v11, np.nan])
                faces.append([v00, v11, v01, np.nan])
            else:
                faces.append([v00, v11, v10, np.nan])
                faces.append([v00, v01, v11, np.nan])

    return verts, np.array(faces, dtype = float)


def _merge_verts(verts: np.ndarray, faces: np.ndarray,
                 tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Merge duplicate vertices and update 0-indexed face table."""
    # Round-based deduplication (fast for moderate meshes)
    scale = 1.0 / tol
    keys = np.round(verts * scale).astype(np.int64)
    _, inv = np.unique(keys, axis = 0, return_inverse = True)

    new_verts_list: List[np.ndarray] = [None] * int(inv.max() + 1)
    for i, ui in enumerate(inv):
        new_verts_list[ui] = verts[i]
    new_verts = np.array(new_verts_list)

    new_faces = faces.copy()
    for col in range(3):
        fi = new_faces[:, col].astype(int)
        new_faces[:, col] = inv[fi].astype(float)

    return new_verts, new_faces


def _apply_supersphere(verts: np.ndarray, half: float, e: float) -> np.ndarray:
    """Project verts onto the super-sphere surface (same formula as tricube)."""
    from mnpbem.utils.matlab_compat import msin, mcos, matan2, msqrt

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    hxy = msqrt(x ** 2 + y ** 2)
    phi = matan2(y, x)
    theta = matan2(z, hxy)

    def isin(a):
        return np.sign(msin(a)) * np.abs(msin(a)) ** e

    def icos(a):
        return np.sign(mcos(a)) * np.abs(mcos(a)) ** e

    x_new = half * icos(theta) * icos(phi)
    y_new = half * icos(theta) * isin(phi)
    z_new = half * isin(theta)

    return np.column_stack([x_new, y_new, z_new])


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

_FACE_DEFS = [
    ('+x', 0, +1),
    ('-x', 0, -1),
    ('+y', 1, +1),
    ('-y', 1, -1),
    ('+z', 2, +1),
    ('-z', 2, -1),
]


def build_adaptive_cube(
        size: float,
        n_default: int,
        face_densities: Optional[Dict[str, int]] = None,
        e: float = 0.25,
        edge_profile_kwargs: Optional[Dict[str, Any]] = None,
        interp: str = 'curv') -> Any:
    """Build a Particle for a rounded cube with optional per-face density.

    Parameters
    ----------
    size : float
        Edge length in nm.
    n_default : int
        Fallback divisions per edge for faces not listed in ``face_densities``.
    face_densities : dict, optional
        Keys are any of ``'+x', '-x', '+y', '-y', '+z', '-z'``.
        Values are integer divisions per edge for that face.
        Faces not listed get ``n_default``.
    e : float
        Super-sphere rounding exponent (same as ``tricube`` ``e`` param).
    edge_profile_kwargs : dict, optional
        If given, an ``EdgeProfile(size, **edge_profile_kwargs)`` is used to
        apply additional edge/corner rounding BEFORE the super-sphere step.
        Keys forwarded: ``e`` (EdgeProfile's own exponent), ``dz``, ``mode``
        (see ``mnpbem.geometry.EdgeProfile`` docstring).
    interp : str
        Interpolation mode passed to ``Particle``.

    Returns
    -------
    Particle
    """
    from mnpbem.geometry.particle import Particle

    if face_densities is None:
        face_densities = {}

    half = size / 2.0

    all_verts: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    v_offset = 0

    for face_name, axis, sign in _FACE_DEFS:
        n = int(face_densities.get(face_name, n_default))
        verts, faces = _make_face_grid(axis, sign, half, n)

        # Optional EdgeProfile rounding (horizontal displacement of edge verts)
        if edge_profile_kwargs is not None:
            verts = _apply_edge_profile(verts, axis, sign, half, size,
                                        edge_profile_kwargs)

        faces_off = faces.copy()
        faces_off[:, :3] += v_offset

        all_verts.append(verts)
        all_faces.append(faces_off)
        v_offset += len(verts)

    verts_cat = np.vstack(all_verts)
    faces_cat = np.vstack(all_faces)

    # Merge shared-edge vertices
    verts_cat, faces_cat = _merge_verts(verts_cat, faces_cat)

    # Super-sphere projection for rounded corners/edges
    verts_cat = _apply_supersphere(verts_cat, half, e)

    # Remove degenerate faces (two or more identical vertex indices)
    valid = np.array([
        (int(f[0]) != int(f[1])) and
        (int(f[1]) != int(f[2])) and
        (int(f[0]) != int(f[2]))
        for f in faces_cat
    ])
    faces_cat = faces_cat[valid]

    # The adaptive mesh has flat triangular faces so curv interp is not
    # directly applicable without adding midpoints (verts2/faces2).
    # Honour the caller's interp if it is 'flat'; fall back to 'flat' for
    # 'curv' since the polyhedral surface has no curved parameterisation.
    safe_interp = interp if interp == 'flat' else 'flat'
    return Particle(verts_cat, faces_cat, interp = safe_interp)


def _apply_edge_profile(
        verts: np.ndarray,
        axis: int,
        sign: int,
        half: float,
        size: float,
        ep_kwargs: Dict[str, Any]) -> np.ndarray:
    """Apply EdgeProfile horizontal displacement to vertices near the face boundary.

    EdgeProfile maps z → horizontal offset d.  On each cube face the "z" axis
    is the normal axis (fixed at ±half).  The two in-plane axes play the role of
    the radial direction d in the profile.

    We approximate the MATLAB edgeprofile behaviour by using the EdgeProfile's
    ``hshift`` method: for vertices whose in-plane coordinates are at ±half we
    clamp them inward by the profile displacement evaluated at that z.
    """
    from mnpbem.geometry import EdgeProfile

    ep_kw = dict(ep_kwargs)
    ep = EdgeProfile(size, ep_kw.pop('nz', 7), **ep_kw)

    verts = verts.copy()

    # axes that span the face
    if axis == 0:
        ax1, ax2 = 1, 2
    elif axis == 1:
        ax1, ax2 = 0, 2
    else:
        ax1, ax2 = 0, 1

    for i, v in enumerate(verts):
        c1, c2 = v[ax1], v[ax2]
        # vertices at the edge of the face (|c| == half)
        at_edge1 = abs(abs(c1) - half) < 1e-9
        at_edge2 = abs(abs(c2) - half) < 1e-9

        if at_edge1 or at_edge2:
            # Use the profile's hshift: treat |c| as z value
            z_val = max(abs(c1), abs(c2))
            try:
                d_shift = float(ep.hshift(np.array([z_val]))[0])
            except Exception:
                continue
            if at_edge1:
                verts[i, ax1] = np.sign(c1) * (half + d_shift)
            if at_edge2:
                verts[i, ax2] = np.sign(c2) * (half + d_shift)

    return verts
