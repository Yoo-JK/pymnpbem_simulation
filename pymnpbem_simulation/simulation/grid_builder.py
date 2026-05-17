import os
import sys

from typing import Any, List, Sequence, Tuple

import numpy as np


def split_grid(
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        grid_z: np.ndarray,
        n_chunks: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, int, int]]:
    """Split a grid (x, y, z) into ``n_chunks`` contiguous slices along the
    flattened point axis.

    The grid is treated as a flat list of ``N = grid_x.size`` points so the
    split is robust to any grid dimensionality (rectangular 3-D, spherical,
    single z-plane, custom points). Each chunk is returned as a tuple of
    1-D ``np.ndarray`` of identical length, so the downstream ``MeshField``
    constructor sees fully matching shapes and skips its ``_expand``
    broadcast path.

    Parameters
    ----------
    grid_x, grid_y, grid_z : np.ndarray
        Source grid arrays of identical shape (rectangular 3-D meshgrid,
        spherical meshgrid, or 1-D custom points).
    n_chunks : int
        Number of chunks to split into. Must be >= 1.

    Returns
    -------
    list of (x_chunk, y_chunk, z_chunk, pts_chunk, start, stop)
        ``x/y/z_chunk`` are 1-D arrays of length ``stop - start``.
        ``pts_chunk`` is (stop - start, 3). ``start, stop`` are the flat
        index slice the chunk covers (so callers can scatter results back).
    """
    assert n_chunks >= 1, '[error] n_chunks must be >= 1!'

    x_flat = np.asarray(grid_x, dtype = np.float64).ravel()
    y_flat = np.asarray(grid_y, dtype = np.float64).ravel()
    z_flat = np.asarray(grid_z, dtype = np.float64).ravel()

    n_total = x_flat.size

    assert y_flat.size == n_total and z_flat.size == n_total, \
            '[error] grid x/y/z must have identical size, got <{}, {}, {}>!'.format(
                    x_flat.size, y_flat.size, z_flat.size)

    if n_chunks > n_total:
        n_chunks = max(1, n_total)

    chunk_size = (n_total + n_chunks - 1) // n_chunks

    chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]] = []

    for g in range(n_chunks):
        start = g * chunk_size
        stop = min(n_total, (g + 1) * chunk_size)

        if start >= stop:
            continue

        x_chunk = x_flat[start:stop].copy()
        y_chunk = y_flat[start:stop].copy()
        z_chunk = z_flat[start:stop].copy()

        pts_chunk = np.empty((stop - start, 3), dtype = np.float64)
        pts_chunk[:, 0] = x_chunk
        pts_chunk[:, 1] = y_chunk
        pts_chunk[:, 2] = z_chunk

        chunks.append((x_chunk, y_chunk, z_chunk, pts_chunk, start, stop))

    return chunks


def make_rectangular_grid(
        x_range: Sequence[float],
        y_range: Sequence[float],
        z_range: Sequence[float],
        n_points: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert len(x_range) == 2 and len(y_range) == 2 and len(z_range) == 2, \
            '[error] *_range must be length-2: [min, max]!'

    if isinstance(n_points, int):
        nx = ny = nz = int(n_points)
    else:
        assert len(n_points) == 3, '[error] <n_points> must be length-3 (nx, ny, nz)!'
        nx, ny, nz = int(n_points[0]), int(n_points[1]), int(n_points[2])

    if nx == 1:
        xs = np.array([0.5 * (float(x_range[0]) + float(x_range[1]))], dtype = np.float64)
    else:
        xs = np.linspace(float(x_range[0]), float(x_range[1]), nx, dtype = np.float64)

    if ny == 1:
        ys = np.array([0.5 * (float(y_range[0]) + float(y_range[1]))], dtype = np.float64)
    else:
        ys = np.linspace(float(y_range[0]), float(y_range[1]), ny, dtype = np.float64)

    if nz == 1:
        zs = np.array([0.5 * (float(z_range[0]) + float(z_range[1]))], dtype = np.float64)
    else:
        zs = np.linspace(float(z_range[0]), float(z_range[1]), nz, dtype = np.float64)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing = 'ij')

    n_total = xx.size
    pts = np.empty((n_total, 3), dtype = np.float64)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()
    pts[:, 2] = zz.ravel()

    return xx, yy, zz, pts


def make_spherical_grid(
        r_range: Sequence[float],
        theta_range: Sequence[float],
        phi_range: Sequence[float],
        n_points: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert len(r_range) == 2 and len(theta_range) == 2 and len(phi_range) == 2, \
            '[error] *_range must be length-2!'

    if isinstance(n_points, int):
        nr = nt = np_ = int(n_points)
    else:
        assert len(n_points) == 3, '[error] <n_points> must be length-3 (nr, ntheta, nphi)!'
        nr, nt, np_ = int(n_points[0]), int(n_points[1]), int(n_points[2])

    rs = np.linspace(float(r_range[0]), float(r_range[1]), nr, dtype = np.float64) \
            if nr > 1 else np.array([float(r_range[0])], dtype = np.float64)
    ts = np.linspace(float(theta_range[0]), float(theta_range[1]), nt, dtype = np.float64) \
            if nt > 1 else np.array([float(theta_range[0])], dtype = np.float64)
    ps = np.linspace(float(phi_range[0]), float(phi_range[1]), np_, dtype = np.float64) \
            if np_ > 1 else np.array([float(phi_range[0])], dtype = np.float64)

    rr, tt, pp = np.meshgrid(rs, ts, ps, indexing = 'ij')

    sin_t = np.sin(tt)
    cos_t = np.cos(tt)
    sin_p = np.sin(pp)
    cos_p = np.cos(pp)

    xx = rr * sin_t * cos_p
    yy = rr * sin_t * sin_p
    zz = rr * cos_t

    n_total = xx.size
    pts = np.empty((n_total, 3), dtype = np.float64)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()
    pts[:, 2] = zz.ravel()

    return xx, yy, zz, pts


def make_custom_points(
        points: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    pts = np.asarray(points, dtype = np.float64)

    assert pts.ndim == 2 and pts.shape[1] == 3, \
            '[error] <points> must have shape (N, 3), got <{}>!'.format(pts.shape)

    n_total = pts.shape[0]

    xx = pts[:, 0].reshape(n_total, 1)
    yy = pts[:, 1].reshape(n_total, 1)
    zz = pts[:, 2].reshape(n_total, 1)

    return xx, yy, zz, pts.copy()
