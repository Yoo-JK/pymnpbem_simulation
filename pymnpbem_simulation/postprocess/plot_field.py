import os
import sys

from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Internal/external field separation maps  (ported from OLD visualizer)
# ---------------------------------------------------------------------------

def _determine_plane(x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray):
    """Return (plane_type, extent, x_label, y_label) for a 2-D field grid."""
    x_grid = np.atleast_2d(np.asarray(x_grid, dtype=float))
    y_grid = np.atleast_2d(np.asarray(y_grid, dtype=float))
    z_grid = np.atleast_2d(np.asarray(z_grid, dtype=float))

    x_const = (np.unique(x_grid).size == 1)
    y_const = (np.unique(y_grid).size == 1)
    z_const = (np.unique(z_grid).size == 1)

    def _ext(arr_min, arr_max):
        if arr_min == arr_max:
            return arr_min - 0.5, arr_max + 0.5
        return arr_min, arr_max

    if y_const:
        xmin, xmax = _ext(float(x_grid.min()), float(x_grid.max()))
        zmin, zmax = _ext(float(z_grid.min()), float(z_grid.max()))
        return 'xz', [xmin, xmax, zmin, zmax], 'x (nm)', 'z (nm)'
    elif z_const:
        xmin, xmax = _ext(float(x_grid.min()), float(x_grid.max()))
        ymin, ymax = _ext(float(y_grid.min()), float(y_grid.max()))
        return 'xy', [xmin, xmax, ymin, ymax], 'x (nm)', 'y (nm)'
    elif x_const:
        ymin, ymax = _ext(float(y_grid.min()), float(y_grid.max()))
        zmin, zmax = _ext(float(z_grid.min()), float(z_grid.max()))
        return 'yz', [ymin, ymax, zmin, zmax], 'y (nm)', 'z (nm)'
    else:
        xmin, xmax = _ext(float(x_grid.min()), float(x_grid.max()))
        ymin, ymax = _ext(float(y_grid.min()), float(y_grid.max()))
        return 'xy', [xmin, xmax, ymin, ymax], 'x (nm)', 'y (nm)'


def _draw_material_boundary(ax, section: Dict) -> None:
    """Draw a circle or rectangle boundary on *ax*."""
    if section['type'] == 'circle':
        cx, cy = section['center']
        r = section['radius']
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                'w--', linewidth=1.2, alpha=0.8)
    elif section['type'] == 'rectangle':
        xmin, xmax, ymin, ymax = section['bounds']
        ax.plot([xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                'w--', linewidth=1.2, alpha=0.8)


def plot_field_separate(field_data: Dict[str, Any],
        geometry=None,
        save: Optional[str] = None,
        title: str = '',
        dpi: int = 150) -> str:
    """2-panel plot: external field | internal field with boundary overlays.

    Parameters
    ----------
    field_data : dict
        Must contain ``'enhancement_ext'``, ``'enhancement_int'``, ``'x_grid'``,
        ``'y_grid'``, ``'z_grid'``, ``'wavelength'``, optionally
        ``'polarization_idx'``.
    geometry : GeometryCrossSection, optional
        For material-boundary overlays.
    save : str, optional
    title : str
    dpi : int

    Returns
    -------
    str  saved path (or '' if *save* is None)
    """
    enh_ext = np.abs(np.asarray(field_data['enhancement_ext'], dtype=complex)).real
    enh_int = np.abs(np.asarray(field_data['enhancement_int'], dtype=complex)).real
    x_grid = np.asarray(field_data['x_grid'])
    y_grid = np.asarray(field_data['y_grid'])
    z_grid = np.asarray(field_data['z_grid'])
    wavelength = float(field_data.get('wavelength', 0))
    pol_idx = field_data.get('polarization_idx')
    pol_label = 'pol {}'.format(pol_idx + 1) if pol_idx is not None else ''

    plane_type, extent, xlabel, ylabel = _determine_plane(x_grid, y_grid, z_grid)

    valid = np.concatenate([
        enh_ext[np.isfinite(enh_ext)].ravel(),
        enh_int[np.isfinite(enh_int)].ravel()])
    vmin = float(np.percentile(valid, 1)) if valid.size else 0.0
    vmax = float(np.percentile(valid, 99)) if valid.size else 1.0

    fig, axes = plt.subplots(1, 2, figsize = (16, 6))
    labels = ['External Field', 'Internal Field']
    arrays = [np.ma.masked_invalid(enh_ext), np.ma.masked_invalid(enh_int)]

    z_plane = float(z_grid.flat[0])
    sections = geometry.get_cross_section(z_plane) if geometry is not None else []

    for ax, lbl, arr in zip(axes, labels, arrays):
        im = ax.imshow(arr, extent = extent, origin = 'lower', cmap = 'hot',
                       aspect = 'auto', vmin = vmin, vmax = vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('{}\nlambda={:.1f} nm {}'.format(
            title or lbl, wavelength, pol_label))
        cb = fig.colorbar(im, ax = ax)
        cb.set_label('|E|^2/|E0|^2')
        for sec in sections:
            _draw_material_boundary(ax, sec)

    fig.tight_layout()
    path = ''
    if save is not None:
        ensure_dir(os.path.dirname(save) or '.')
        fig.savefig(save, dpi = dpi)
        print_info('saved <{}>'.format(save))
        path = save
    plt.close(fig)
    return path


def plot_field_comparison(field_data: Dict[str, Any],
        geometry=None,
        save: Optional[str] = None,
        title: str = '',
        dpi: int = 150) -> str:
    """3-panel plot: external | internal | merged with boundary overlays.

    Parameters
    ----------
    field_data : dict
        As for :func:`plot_field_separate`, plus ``'enhancement'`` (merged).
    geometry, save, title, dpi : same as :func:`plot_field_separate`.
    """
    enh_ext = np.abs(np.asarray(field_data['enhancement_ext'], dtype=complex)).real
    enh_int = np.abs(np.asarray(field_data['enhancement_int'], dtype=complex)).real
    enh_mrg = np.abs(np.asarray(field_data['enhancement'], dtype=complex)).real
    x_grid = np.asarray(field_data['x_grid'])
    y_grid = np.asarray(field_data['y_grid'])
    z_grid = np.asarray(field_data['z_grid'])
    wavelength = float(field_data.get('wavelength', 0))
    pol_idx = field_data.get('polarization_idx')
    pol_label = 'pol {}'.format(pol_idx + 1) if pol_idx is not None else ''

    plane_type, extent, xlabel, ylabel = _determine_plane(x_grid, y_grid, z_grid)

    arrays = [np.ma.masked_invalid(enh_ext),
              np.ma.masked_invalid(enh_int),
              np.ma.masked_invalid(enh_mrg)]
    all_valid = np.concatenate([a.compressed() for a in arrays])
    vmin = float(np.percentile(all_valid, 1)) if all_valid.size else 0.0
    vmax = float(np.percentile(all_valid, 99)) if all_valid.size else 1.0

    z_plane = float(z_grid.flat[0])
    sections = geometry.get_cross_section(z_plane) if geometry is not None else []

    fig, axes = plt.subplots(1, 3, figsize = (22, 6))
    panel_titles = ['External', 'Internal', 'Merged']

    for ax, lbl, arr in zip(axes, panel_titles, arrays):
        im = ax.imshow(arr, extent = extent, origin = 'lower', cmap = 'hot',
                       aspect = 'auto', vmin = vmin, vmax = vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('{}\nlambda={:.1f} nm {}'.format(lbl, wavelength, pol_label))
        cb = fig.colorbar(im, ax = ax)
        cb.set_label('|E|^2/|E0|^2')
        for sec in sections:
            _draw_material_boundary(ax, sec)

    if title:
        fig.suptitle(title, fontweight = 'bold')
    fig.tight_layout()
    path = ''
    if save is not None:
        ensure_dir(os.path.dirname(save) or '.')
        fig.savefig(save, dpi = dpi)
        print_info('saved <{}>'.format(save))
        path = save
    plt.close(fig)
    return path


def plot_field_overlay(field_data: Dict[str, Any],
        geometry=None,
        save: Optional[str] = None,
        title: str = '',
        dpi: int = 150) -> str:
    """External field heatmap + internal field scatter overlay.

    Parameters
    ----------
    field_data : dict
        As for :func:`plot_field_separate`.
    geometry, save, title, dpi : same as :func:`plot_field_separate`.
    """
    enh_ext = np.abs(np.asarray(field_data['enhancement_ext'], dtype=complex)).real
    enh_int = np.abs(np.asarray(field_data['enhancement_int'], dtype=complex)).real
    x_grid = np.asarray(field_data['x_grid'])
    y_grid = np.asarray(field_data['y_grid'])
    z_grid = np.asarray(field_data['z_grid'])
    wavelength = float(field_data.get('wavelength', 0))
    pol_idx = field_data.get('polarization_idx')
    pol_label = 'pol {}'.format(pol_idx + 1) if pol_idx is not None else ''

    plane_type, extent, xlabel, ylabel = _determine_plane(x_grid, y_grid, z_grid)

    enh_ext_m = np.ma.masked_invalid(enh_ext)
    valid_ext = enh_ext_m.compressed()
    vmin_ext = float(np.percentile(valid_ext, 1)) if valid_ext.size else 0.0
    vmax_ext = float(np.percentile(valid_ext, 99)) if valid_ext.size else 1.0

    fig, ax = plt.subplots(figsize = (12, 9))
    im = ax.imshow(enh_ext_m, extent = extent, origin = 'lower', cmap = 'hot',
                   aspect = 'auto', vmin = vmin_ext, vmax = vmax_ext, alpha = 0.7)
    cb_ext = fig.colorbar(im, ax = ax)
    cb_ext.set_label('|E|^2/|E0|^2 (External)')

    int_mask = np.isfinite(enh_int) & (enh_int > 0)
    if int_mask.any():
        if plane_type == 'xz':
            xu = x_grid[int_mask]
            yu = z_grid[int_mask]
        elif plane_type == 'yz':
            xu = y_grid[int_mask]
            yu = z_grid[int_mask]
        else:
            xu = x_grid[int_mask]
            yu = y_grid[int_mask]
        vals = enh_int[int_mask]
        sc = ax.scatter(xu, yu, c = vals, cmap = 'viridis',
                        s = 18, edgecolors = 'black', linewidth = 0.3, alpha = 0.9)
        cb_int = fig.colorbar(sc, ax = ax, pad = 0.12)
        cb_int.set_label('|E|^2/|E0|^2 (Internal)')

    z_plane = float(z_grid.flat[0])
    sections = geometry.get_cross_section(z_plane) if geometry is not None else []
    for sec in sections:
        _draw_material_boundary(ax, sec)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or 'Internal (scatter) + External (heatmap)\nlambda={:.1f} nm {}'.format(
        wavelength, pol_label))
    fig.tight_layout()

    path = ''
    if save is not None:
        ensure_dir(os.path.dirname(save) or '.')
        fig.savefig(save, dpi = dpi)
        print_info('saved <{}>'.format(save))
        path = save
    plt.close(fig)
    return path


def plot_unpolarized_fields(fields_by_pol: List[Dict[str, Any]],
        save: Optional[str] = None,
        title: str = '',
        dpi: int = 150) -> str:
    """Linear + log panels of incoherently averaged (unpolarised) field map.

    Averages ``enhancement`` across all polarisations using arithmetic mean
    (FDTD-style incoherent average for intensity enhancement).

    Parameters
    ----------
    fields_by_pol : list of dict
        One dict per polarisation, each with ``'enhancement'``, ``'x_grid'``,
        ``'y_grid'``, ``'z_grid'``, ``'wavelength'``.
    save : str, optional
    title : str
    dpi : int

    Returns
    -------
    str  saved path (or '' if *save* is None)
    """
    if not fields_by_pol:
        return ''

    enhs = [np.abs(np.asarray(fd['enhancement'], dtype=complex)).real for fd in fields_by_pol]
    unpol_enh = np.mean(enhs, axis = 0)
    ref = fields_by_pol[0]
    x_grid = np.asarray(ref['x_grid'])
    y_grid = np.asarray(ref['y_grid'])
    z_grid = np.asarray(ref['z_grid'])
    wavelength = float(ref.get('wavelength', 0))
    n_pol = len(fields_by_pol)

    plane_type, extent, xlabel, ylabel = _determine_plane(x_grid, y_grid, z_grid)
    enh_m = np.ma.masked_invalid(unpol_enh)
    valid = enh_m.compressed()
    vmin_lin = float(np.percentile(valid, 1)) if valid.size else 0.0
    vmax_lin = float(np.percentile(valid, 99)) if valid.size else 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))

    im1 = ax1.imshow(enh_m, extent = extent, origin = 'lower', cmap = 'hot',
                     aspect = 'auto', vmin = vmin_lin, vmax = vmax_lin)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title('Unpolarised (linear)\nlambda={:.1f} nm  n_pol={}'.format(wavelength, n_pol))
    cb1 = fig.colorbar(im1, ax = ax1)
    cb1.set_label('|E|^2/|E0|^2')

    from matplotlib.colors import LogNorm
    pos_vals = valid[valid > 0]
    if pos_vals.size > 0:
        vmin_log = max(float(np.percentile(pos_vals, 1)), 1e-12)
        vmax_log = float(np.percentile(pos_vals, 99))
        if vmin_log >= vmax_log:
            vmin_log = vmax_log / 1000
        norm = LogNorm(vmin = vmin_log, vmax = vmax_log)
        im2 = ax2.imshow(enh_m, extent = extent, origin = 'lower', cmap = 'hot',
                         aspect = 'auto', norm = norm)
    else:
        im2 = ax2.imshow(enh_m, extent = extent, origin = 'lower', cmap = 'hot', aspect = 'auto')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title('Unpolarised (log)\nlambda={:.1f} nm  n_pol={}'.format(wavelength, n_pol))
    cb2 = fig.colorbar(im2, ax = ax2)
    cb2.set_label('|E|^2/|E0|^2 (log)')

    if title:
        fig.suptitle(title, fontweight = 'bold')
    fig.tight_layout()

    path = ''
    if save is not None:
        ensure_dir(os.path.dirname(save) or '.')
        fig.savefig(save, dpi = dpi)
        print_info('saved <{}>'.format(save))
        path = save
    plt.close(fig)
    return path
