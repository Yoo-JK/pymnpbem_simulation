"""
Eigenmode visualization helpers.

These functions produce 3D mode pattern plots and magnitude/phase spectra
across wavelength sweeps, using the output of qs_eigenmodes / svd_decomposition
/ retarded_eigenmodes.

Lighter-weight than mnpbem_simulation's visualizer (single-method focus,
no cross-method grid) but sufficient for the common case.
"""

import os

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..util import ensure_dir, print_info


def plot_mode_patterns(out_dir: str,
        eigenmode: Dict[str, Any],
        particle: Any,
        mode_indices: Optional[List[int]] = None,
        n_modes: int = 5,
        title: str = '',
        plot_format: Optional[List[str]] = None,
        dpi: int = 150,
        cmap: str = 'seismic',
        fname: str = 'mode_patterns_grid') -> List[str]:
    """3D plot grid of selected eigenmode patterns Re[psi_n(r)] on the surface mesh.

    Args:
        out_dir: directory for output.
        eigenmode: dict from qs_eigenmodes (must have 'eigenvectors_r' Nx M).
        particle: object with .verts and .faces (mnpbem ComParticle).
        mode_indices: explicit indices to plot. If None, uses [0..n_modes-1].
        n_modes: how many modes to plot when mode_indices is None.

    Returns: list of saved file paths.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ensure_dir(out_dir)

    if plot_format is None:
        plot_format = ['png']

    ur = np.asarray(eigenmode.get('eigenvectors_r'))
    if ur.ndim != 2:
        raise ValueError('[error] eigenvectors_r must be 2D, got shape <{}>'.format(ur.shape))

    n_avail = ur.shape[1]
    if mode_indices is None:
        mode_indices = list(range(min(n_modes, n_avail)))
    else:
        mode_indices = [int(i) for i in mode_indices if 0 <= int(i) < n_avail]

    if not mode_indices:
        return []

    verts = np.asarray(particle.verts)
    faces = np.asarray(particle.faces)

    # Triangulate quads (drop NaN sentinel column if present).
    faces_tri = _triangulate_faces(faces)
    if faces_tri.shape[0] == 0:
        return []

    verts_tri = verts[faces_tri]

    n_cols = min(len(mode_indices), 5)
    n_rows = (len(mode_indices) + n_cols - 1) // n_cols

    fig = plt.figure(figsize = (4.0 * n_cols, 3.6 * n_rows))

    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    margin = max((x_max - x_min) * 0.05, 1.0)
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    z_range = max(z_max - z_min, 1.0)
    max_range = max(x_range, y_range, z_range)

    for plot_idx, mode_idx in enumerate(mode_indices):
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection = '3d')

        psi = ur[:, mode_idx]
        comp = np.real(psi)

        # Per-face value: if comp is per-face length matches faces_tri rows.
        if comp.shape[0] == faces_tri.shape[0]:
            comp_face = comp
        elif comp.shape[0] == faces.shape[0]:
            # Replicate when triangulation split quads (faces_tri may be longer).
            comp_face = _replicate_for_triangulation(comp, faces, faces_tri.shape[0])
        elif comp.shape[0] == verts.shape[0]:
            # Per-vertex; average to per-face.
            comp_face = comp[faces_tri].mean(axis = 1)
        else:
            # Fallback: zero-fill.
            comp_face = np.zeros(faces_tri.shape[0])

        p95 = float(np.percentile(np.abs(comp_face), 95))
        if p95 < 1e-12:
            p95 = float(np.max(np.abs(comp_face))) + 1e-30
        comp_plot = np.clip(comp_face / p95, -1.0, 1.0)

        poly = Poly3DCollection(verts_tri, alpha = 0.95,
                edgecolor = 'k', linewidth = 0.15)
        poly.set_array(comp_plot)
        poly.set_cmap(cmap)
        poly.set_clim(-1.0, 1.0)
        ax.add_collection3d(poly)

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)
        ax.set_box_aspect([x_range / max_range,
                y_range / max_range, z_range / max_range])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ene = eigenmode.get('eigenvalues')
        if ene is not None and mode_idx < len(np.asarray(ene)):
            try:
                e_val = complex(np.asarray(ene)[mode_idx])
                ax.set_title('mode {} (Re={:.3g})'.format(mode_idx, e_val.real),
                        fontsize = 10)
            except Exception:
                ax.set_title('mode {}'.format(mode_idx), fontsize = 10)
        else:
            ax.set_title('mode {}'.format(mode_idx), fontsize = 10)

    fig.suptitle(title if title else 'Eigenmode patterns Re[$\\psi_n(r)$]',
            fontsize = 14, fontweight = 'bold')

    fig.tight_layout(rect = [0.02, 0.02, 1.0, 0.96])

    saved_files = []
    for fmt in plot_format:
        path = os.path.join(out_dir, '{}.{}'.format(fname, fmt))
        fig.savefig(path, dpi = dpi)
        saved_files.append(path)
        print_info('saved <{}>'.format(path))

    plt.close(fig)
    return saved_files


def plot_eigenvalue_spectrum(out_dir: str,
        eigenmode: Dict[str, Any],
        title: str = '',
        plot_format: Optional[List[str]] = None,
        dpi: int = 150,
        fname: str = 'eigenvalue_spectrum') -> List[str]:
    """Plot complex eigenvalues in Re/Im scatter + bar chart of |ene_n|.

    Useful for inspecting the QS eigenmode spectrum at a glance.
    """
    ensure_dir(out_dir)

    if plot_format is None:
        plot_format = ['png']

    ene = np.asarray(eigenmode.get('eigenvalues'))
    if ene.size == 0:
        return []

    fig, axes = plt.subplots(1, 2, figsize = (12, 4.5))

    axes[0].scatter(np.real(ene), np.imag(ene),
            s = 60, color = 'tab:blue', edgecolor = 'k', alpha = 0.8)

    for i, e in enumerate(ene):
        axes[0].annotate(str(i), (np.real(e), np.imag(e)),
                fontsize = 8, alpha = 0.7,
                xytext = (3, 3), textcoords = 'offset points')

    axes[0].axhline(0, color = 'gray', linewidth = 0.5)
    axes[0].axvline(0, color = 'gray', linewidth = 0.5)
    axes[0].set_xlabel('Re(ene_n)', fontsize = 11)
    axes[0].set_ylabel('Im(ene_n)', fontsize = 11)
    axes[0].set_title('Eigenvalues (complex plane)', fontsize = 12, fontweight = 'bold')
    axes[0].grid(True, alpha = 0.3)

    mags = np.abs(ene)
    axes[1].bar(np.arange(len(mags)), mags,
            color = plt.cm.viridis(np.linspace(0.2, 0.9, len(mags))),
            edgecolor = 'k')
    axes[1].set_xlabel('Mode index n', fontsize = 11)
    axes[1].set_ylabel('|ene_n|', fontsize = 11)
    axes[1].set_title('|ene_n| per mode', fontsize = 12, fontweight = 'bold')
    axes[1].grid(True, alpha = 0.3, axis = 'y')

    fig.suptitle(title if title else 'Eigenmode spectrum',
            fontsize = 13, fontweight = 'bold')
    fig.tight_layout()

    saved_files = []
    for fmt in plot_format:
        path = os.path.join(out_dir, '{}.{}'.format(fname, fmt))
        fig.savefig(path, dpi = dpi)
        saved_files.append(path)
        print_info('saved <{}>'.format(path))

    plt.close(fig)
    return saved_files


def plot_singular_value_decay(out_dir: str,
        svd_result: Dict[str, Any],
        title: str = '',
        plot_format: Optional[List[str]] = None,
        dpi: int = 150,
        fname: str = 'svd_decay') -> List[str]:
    """Plot singular value decay (log scale) + cumulative energy from SVD result."""
    ensure_dir(out_dir)

    if plot_format is None:
        plot_format = ['png']

    s = np.asarray(svd_result.get('singular_values', np.array([])))
    if s.size == 0:
        return []

    energy_cum = np.asarray(svd_result.get('energy_cumulative', np.cumsum(s ** 2) / np.sum(s ** 2)))
    rank_eff = int(svd_result.get('rank_eff', 0))

    fig, axes = plt.subplots(1, 2, figsize = (12, 4.5))

    axes[0].semilogy(np.arange(len(s)), s, 'o-', color = 'crimson', linewidth = 1.5)
    axes[0].axvline(rank_eff, color = 'tab:blue', linestyle = '--', alpha = 0.7,
            label = 'rank_eff = {}'.format(rank_eff))
    axes[0].set_xlabel('Mode index', fontsize = 11)
    axes[0].set_ylabel('Singular value', fontsize = 11)
    axes[0].set_title('Singular values (log)', fontsize = 12, fontweight = 'bold')
    axes[0].grid(True, alpha = 0.3, which = 'both')
    axes[0].legend(fontsize = 9)

    axes[1].plot(np.arange(len(energy_cum)), energy_cum, 'o-',
            color = 'tab:green', linewidth = 1.5)
    axes[1].axvline(rank_eff, color = 'tab:blue', linestyle = '--', alpha = 0.7)
    axes[1].axhline(0.99, color = 'gray', linestyle = ':', alpha = 0.7,
            label = '99% energy')
    axes[1].set_xlabel('Mode index', fontsize = 11)
    axes[1].set_ylabel('Cumulative energy', fontsize = 11)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title('Cumulative energy', fontsize = 12, fontweight = 'bold')
    axes[1].grid(True, alpha = 0.3)
    axes[1].legend(fontsize = 9)

    fig.suptitle(title if title else 'SVD decay', fontsize = 13, fontweight = 'bold')
    fig.tight_layout()

    saved_files = []
    for fmt in plot_format:
        path = os.path.join(out_dir, '{}.{}'.format(fname, fmt))
        fig.savefig(path, dpi = dpi)
        saved_files.append(path)
        print_info('saved <{}>'.format(path))

    plt.close(fig)
    return saved_files


def _triangulate_faces(faces: np.ndarray) -> np.ndarray:
    """Convert mnpbem face arrays to pure-triangle (Nx3) form."""
    faces = np.asarray(faces)

    if faces.ndim != 2:
        raise ValueError('[error] faces must be 2D, got shape <{}>'.format(faces.shape))

    n_cols = faces.shape[1]
    out = []

    if np.issubdtype(faces.dtype, np.floating):
        for f in faces:
            valid = f[np.isfinite(f)].astype(int)
            if valid.size == 3:
                out.append(valid)
            elif valid.size == 4:
                # Split quad to two tris.
                out.append(np.array([valid[0], valid[1], valid[2]]))
                out.append(np.array([valid[0], valid[2], valid[3]]))
            elif valid.size > 4:
                # Triangle fan.
                for j in range(1, valid.size - 1):
                    out.append(np.array([valid[0], valid[j], valid[j + 1]]))
    else:
        for f in faces:
            valid = f[f >= 0]
            if valid.size == 3:
                out.append(valid)
            elif valid.size == 4:
                out.append(np.array([valid[0], valid[1], valid[2]]))
                out.append(np.array([valid[0], valid[2], valid[3]]))
            elif valid.size > 4:
                for j in range(1, valid.size - 1):
                    out.append(np.array([valid[0], valid[j], valid[j + 1]]))

    if not out:
        return np.zeros((0, 3), dtype = int)

    return np.asarray(out, dtype = int)


def _replicate_for_triangulation(per_face: np.ndarray,
        faces_raw: np.ndarray,
        n_tri: int) -> np.ndarray:
    """When mnpbem faces include quads (split into 2 tris), replicate the
    per-face value so the new triangle array has the right length."""
    faces_raw = np.asarray(faces_raw)
    out = []

    if np.issubdtype(faces_raw.dtype, np.floating):
        for i, f in enumerate(faces_raw):
            valid_count = int(np.sum(np.isfinite(f)))
            if valid_count == 3:
                out.append(per_face[i])
            elif valid_count == 4:
                out.append(per_face[i])
                out.append(per_face[i])
            elif valid_count > 4:
                for _ in range(valid_count - 2):
                    out.append(per_face[i])
            else:
                pass
    else:
        for i, f in enumerate(faces_raw):
            valid_count = int(np.sum(f >= 0))
            if valid_count == 3:
                out.append(per_face[i])
            elif valid_count == 4:
                out.append(per_face[i])
                out.append(per_face[i])
            elif valid_count > 4:
                for _ in range(valid_count - 2):
                    out.append(per_face[i])

    arr = np.asarray(out)
    if arr.shape[0] != n_tri:
        # Pad / truncate to expected length.
        if arr.shape[0] < n_tri:
            arr = np.concatenate([arr, np.zeros(n_tri - arr.shape[0])])
        else:
            arr = arr[:n_tri]
    return arr
