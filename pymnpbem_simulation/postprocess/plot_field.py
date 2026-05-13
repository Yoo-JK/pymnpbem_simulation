import os
import sys

from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..util import ensure_dir, print_info


def plot_field_2d_slice(
        field_result: Any,
        axis: str = 'z',
        value: float = 0.0,
        log_scale: bool = True,
        save: Optional[str] = None,
        title: str = '') -> str:

    pos = np.asarray(field_result['pos'] if isinstance(field_result, dict) else field_result.pos)
    e = np.asarray(field_result['e'] if isinstance(field_result, dict) else field_result.e)

    e2 = np.sum(np.abs(e) ** 2, axis = -1)
    while e2.ndim > 1:
        e2 = e2.mean(axis = -1)

    if e2.shape[0] != pos.shape[0]:
        e2 = e2.reshape(pos.shape[0], -1).mean(axis = 1)

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    assert axis in axis_map, '[error] Invalid <axis> = <{}>! Use x|y|z.'.format(axis)
    ax_idx = axis_map[axis]

    other_axes = [i for i in range(3) if i != ax_idx]

    coords = pos[:, ax_idx]
    if coords.size == 0:
        raise ValueError('[error] empty <pos> array!')

    if np.allclose(coords, value):
        mask = np.ones(pos.shape[0], dtype = bool)
    else:
        diffs = np.abs(coords - value)
        nearest = float(diffs.min())
        tol = max(1e-9, 0.5 * (coords.max() - coords.min()) / max(1, np.unique(np.round(coords, 9)).size))
        mask = diffs <= max(tol, nearest + 1e-12)

    if mask.sum() == 0:
        mask[np.argmin(np.abs(coords - value))] = True

    sub_pos = pos[mask]
    sub_e2 = e2[mask]

    u = sub_pos[:, other_axes[0]]
    v = sub_pos[:, other_axes[1]]

    fig, ax = plt.subplots(figsize = (7, 6))

    if log_scale:
        plot_val = np.log10(np.maximum(sub_e2, 1e-30))
        clabel = 'log10 |E|^2'
    else:
        plot_val = sub_e2
        clabel = '|E|^2'

    sc = ax.scatter(u, v, c = plot_val, s = 30, cmap = 'inferno')

    ax_names = ['x', 'y', 'z']
    ax.set_xlabel('{} (nm)'.format(ax_names[other_axes[0]]))
    ax.set_ylabel('{} (nm)'.format(ax_names[other_axes[1]]))

    ax.set_title('{} (slice {} = {:.2f})'.format(title if title else '|E|^2 slice', axis, value))

    cbar = fig.colorbar(sc, ax = ax)
    cbar.set_label(clabel)

    ax.set_aspect('equal', adjustable = 'box')
    ax.grid(True, alpha = 0.3)

    fig.tight_layout()

    if save is not None:
        ensure_dir(os.path.dirname(save))
        fig.savefig(save, dpi = 150)
        print_info('saved <{}>'.format(save))
    plt.close(fig)

    return save if save is not None else ''


def plot_hotspots_3d(
        field_result: Any,
        particle: Any,
        save: Optional[str] = None,
        threshold_quantile: float = 0.99,
        title: str = '') -> str:

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    pos = np.asarray(field_result['pos'] if isinstance(field_result, dict) else field_result.pos)
    e = np.asarray(field_result['e'] if isinstance(field_result, dict) else field_result.e)

    e2 = np.sum(np.abs(e) ** 2, axis = -1)
    while e2.ndim > 1:
        e2 = e2.mean(axis = -1)

    if e2.shape[0] != pos.shape[0]:
        e2 = e2.reshape(pos.shape[0], -1).mean(axis = 1)

    threshold = float(np.quantile(e2, threshold_quantile))
    mask = e2 >= threshold

    fig = plt.figure(figsize = (9, 8))
    ax = fig.add_subplot(111, projection = '3d')

    try:
        verts = np.asarray(particle.verts)
        faces = np.asarray(particle.faces)

        face_polys = []
        for f in faces:
            f = np.asarray(f)
            f = f[f >= 0]
            if f.size >= 3:
                face_polys.append(verts[f[:3]])

        if len(face_polys) > 0:
            pc = Poly3DCollection(face_polys,
                    facecolor = (0.85, 0.7, 0.2, 0.3),
                    edgecolor = (0.4, 0.3, 0.1, 0.4),
                    linewidth = 0.2)
            ax.add_collection3d(pc)
    except (AttributeError, TypeError, ValueError) as ex:
        print_info('plot_hotspots_3d: skipped particle mesh ({})'.format(type(ex).__name__))

    if mask.sum() > 0:
        sc = ax.scatter(
                pos[mask, 0], pos[mask, 1], pos[mask, 2],
                c = e2[mask], cmap = 'inferno', s = 25, alpha = 0.9)
        cbar = fig.colorbar(sc, ax = ax, shrink = 0.6)
        cbar.set_label('|E|^2')

    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_zlabel('z (nm)')
    ax.set_title(title if title else 'Hotspots (top {:.1f}%)'.format(100 * (1 - threshold_quantile)))

    fig.tight_layout()

    if save is not None:
        ensure_dir(os.path.dirname(save))
        fig.savefig(save, dpi = 150)
        print_info('saved <{}>'.format(save))
    plt.close(fig)

    return save if save is not None else ''


def plot_field_intensity_2d(
        field_result: Any,
        axis: str = 'z',
        value: float = 0.0,
        log_scale: bool = True,
        save: Optional[str] = None,
        title: str = '',
        cmap: str = 'inferno',
        percentile_range: Tuple[float, float] = (2.0, 99.5)) -> str:
    """Plot |E|^2 intensity on a 2D slice with percentile-based color scaling.

    Distinct from plot_field_2d_slice (which uses uniform scaling) by using
    LogNorm + percentile clipping to avoid hotspot saturation. Useful for
    visualizing weak features alongside strong hotspots.
    """
    pos = np.asarray(field_result['pos'] if isinstance(field_result, dict) else field_result.pos)
    e = np.asarray(field_result['e'] if isinstance(field_result, dict) else field_result.e)

    e2 = np.sum(np.abs(e) ** 2, axis = -1)
    while e2.ndim > 1:
        e2 = e2.mean(axis = -1)
    if e2.shape[0] != pos.shape[0]:
        e2 = e2.reshape(pos.shape[0], -1).mean(axis = 1)

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    assert axis in axis_map, '[error] Invalid <axis> = <{}>!'.format(axis)
    ax_idx = axis_map[axis]
    other = [i for i in range(3) if i != ax_idx]

    coords = pos[:, ax_idx]
    if coords.size == 0:
        raise ValueError('[error] empty pos array!')

    if np.allclose(coords, value):
        mask = np.ones(pos.shape[0], dtype = bool)
    else:
        diffs = np.abs(coords - value)
        nearest = float(diffs.min())
        tol = max(1e-9, 0.5 * (coords.max() - coords.min())
                / max(1, np.unique(np.round(coords, 9)).size))
        mask = diffs <= max(tol, nearest + 1e-12)

    if mask.sum() == 0:
        mask[np.argmin(np.abs(coords - value))] = True

    sub_pos = pos[mask]
    sub_e2 = e2[mask]

    u = sub_pos[:, other[0]]
    v = sub_pos[:, other[1]]

    fig, ax = plt.subplots(figsize = (8, 7))

    if log_scale:
        try:
            from matplotlib.colors import LogNorm
        except ImportError:
            LogNorm = None

        positive = sub_e2[sub_e2 > 0]
        if positive.size > 0 and LogNorm is not None:
            vmin = max(float(np.percentile(positive, percentile_range[0])), 1e-12)
            vmax = float(np.percentile(positive, percentile_range[1]))
            if vmin >= vmax:
                vmin = vmax / 1000.0

            sc = ax.scatter(u, v, c = sub_e2, s = 30, cmap = cmap,
                    norm = LogNorm(vmin = vmin, vmax = vmax))
        else:
            sc = ax.scatter(u, v, c = sub_e2, s = 30, cmap = cmap)
        cbar_label = '|E|^2 (log)'
    else:
        sc = ax.scatter(u, v, c = sub_e2, s = 30, cmap = cmap)
        cbar_label = '|E|^2'

    ax_names = ['x', 'y', 'z']
    ax.set_xlabel('{} (nm)'.format(ax_names[other[0]]))
    ax.set_ylabel('{} (nm)'.format(ax_names[other[1]]))
    ax.set_title('{} (slice {} = {:.2f})'.format(
            title if title else 'Field intensity |E|^2', axis, value))

    cbar = fig.colorbar(sc, ax = ax)
    cbar.set_label(cbar_label)

    ax.set_aspect('equal', adjustable = 'box')
    ax.grid(True, alpha = 0.3)

    fig.tight_layout()

    if save is not None:
        ensure_dir(os.path.dirname(save) or '.')
        fig.savefig(save, dpi = 150)
        print_info('saved <{}>'.format(save))
    plt.close(fig)

    return save if save is not None else ''


def plot_field_vectors_2d(
        field_result: Any,
        axis: str = 'z',
        value: float = 0.0,
        save: Optional[str] = None,
        title: str = '',
        scale: Optional[float] = None,
        density: int = 1,
        background: bool = True) -> str:
    """Plot E-field vectors as quiver arrows on a 2D slice.

    Uses real(E) projection (in-plane only). Optional |E|^2 background
    is drawn underneath when background=True.
    """
    pos = np.asarray(field_result['pos'] if isinstance(field_result, dict) else field_result.pos)
    e = np.asarray(field_result['e'] if isinstance(field_result, dict) else field_result.e)

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax_idx = axis_map[axis]
    other = [i for i in range(3) if i != ax_idx]

    coords = pos[:, ax_idx]
    if np.allclose(coords, value):
        mask = np.ones(pos.shape[0], dtype = bool)
    else:
        diffs = np.abs(coords - value)
        tol = max(1e-9, 0.5 * (coords.max() - coords.min())
                / max(1, np.unique(np.round(coords, 9)).size))
        mask = diffs <= max(tol, float(diffs.min()) + 1e-12)

    if mask.sum() == 0:
        mask[np.argmin(np.abs(coords - value))] = True

    sub_pos = pos[mask]
    sub_e = e[mask]

    if sub_e.ndim < 2 or sub_e.shape[-1] < 3:
        raise ValueError('[error] need 3-component E vectors, got shape <{}>'.format(sub_e.shape))

    u_pos = sub_pos[:, other[0]]
    v_pos = sub_pos[:, other[1]]
    u_e = np.real(sub_e[..., other[0]])
    v_e = np.real(sub_e[..., other[1]])

    if u_e.ndim > 1:
        u_e = u_e.mean(axis = tuple(range(1, u_e.ndim)))
        v_e = v_e.mean(axis = tuple(range(1, v_e.ndim)))

    if density > 1:
        u_pos = u_pos[::density]
        v_pos = v_pos[::density]
        u_e = u_e[::density]
        v_e = v_e[::density]

    fig, ax = plt.subplots(figsize = (8, 7))

    if background:
        e2 = np.sum(np.abs(sub_e) ** 2, axis = -1)
        while e2.ndim > 1:
            e2 = e2.mean(axis = -1)
        if e2.shape[0] != sub_pos.shape[0]:
            e2 = e2.reshape(sub_pos.shape[0], -1).mean(axis = 1)

        e2_sub = e2[::density] if density > 1 else e2
        ax.scatter(u_pos, v_pos, c = e2_sub, s = 8, cmap = 'inferno', alpha = 0.6)

    ax.quiver(u_pos, v_pos, u_e, v_e,
            scale = scale,
            color = 'cyan', alpha = 0.85, width = 0.003)

    ax_names = ['x', 'y', 'z']
    ax.set_xlabel('{} (nm)'.format(ax_names[other[0]]))
    ax.set_ylabel('{} (nm)'.format(ax_names[other[1]]))
    ax.set_title(title if title else 'E-field vectors (slice {}={:.2f})'.format(axis, value))
    ax.set_aspect('equal', adjustable = 'box')
    ax.grid(True, alpha = 0.3)

    fig.tight_layout()

    if save is not None:
        ensure_dir(os.path.dirname(save) or '.')
        fig.savefig(save, dpi = 150)
        print_info('saved <{}>'.format(save))
    plt.close(fig)

    return save if save is not None else ''


def plot_near_field_decay(
        decay_result: Any,
        save: Optional[str] = None,
        log_scale: bool = True,
        title: str = '') -> str:

    distances = np.asarray(decay_result['distances'] if isinstance(decay_result, dict) else decay_result.distances)
    e2 = np.asarray(decay_result['e2'] if isinstance(decay_result, dict) else decay_result.e2)

    fig, ax = plt.subplots(figsize = (7, 5))

    if log_scale:
        ax.set_yscale('log')

    ax.plot(distances, e2, '.-', alpha = 0.7, markersize = 3)
    ax.set_xlabel('distance to surface (nm)')
    ax.set_ylabel('|E|^2')
    ax.set_title(title if title else 'Near-field decay')
    ax.grid(True, alpha = 0.3, which = 'both')

    fig.tight_layout()

    if save is not None:
        ensure_dir(os.path.dirname(save))
        fig.savefig(save, dpi = 150)
        print_info('saved <{}>'.format(save))
    plt.close(fig)

    return save if save is not None else ''
