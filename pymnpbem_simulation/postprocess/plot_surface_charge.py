import os

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..util import ensure_dir, print_info


# ---------------------------------------------------------------------------
# Public entry points (Phase 2 deliverables)
# ---------------------------------------------------------------------------


def plot_all_surface_charge(out_dir: str,
        sc: Dict[str, Any],
        plot_format: Optional[List[str]] = None,
        dpi: int = 200,
        polarization_labels: Optional[List[str]] = None,
        verbose: bool = False) -> List[str]:
    """Iterate over (wavelength x polarization) and emit 3D + 8-view + phase plots.

    Args:
        out_dir: simulation result directory. plots go under <out_dir>/surface_charge/
        sc: dict with keys wavelengths, sig2, verts, faces, centroids, normals,
            polarizations (see PlaneWaveRetRunner._stack_surface_charge +
            _extract_mesh_info).
        plot_format: list of extensions, e.g. ['png'] or ['png', 'pdf'].
        dpi: figure dpi.
        polarization_labels: optional pretty labels per polarization.
        verbose: print progress.

    Returns:
        List of all saved file paths.
    """

    sc_dir = os.path.join(out_dir, 'surface_charge')
    phase_dir = os.path.join(out_dir, 'surface_charge_phase')
    ensure_dir(sc_dir)
    ensure_dir(phase_dir)

    wavelengths = np.asarray(sc['wavelengths'])
    sig2 = np.asarray(sc['sig2'])
    verts = np.asarray(sc['verts'])
    faces = np.asarray(sc['faces'])
    centroids = np.asarray(sc['centroids'])
    normals = np.asarray(sc.get('normals'))
    polarizations = np.asarray(sc.get('polarizations',
            [[1, 0, 0], [0, 1, 0]]))

    if plot_format is None:
        plot_format = ['png']

    n_wl = len(wavelengths)
    n_pol = sig2.shape[2]

    saved = []

    for i_wl in range(n_wl):
        wl = float(wavelengths[i_wl])

        for i_pol in range(n_pol):
            sigma = sig2[i_wl, :, i_pol]

            if polarization_labels is not None and i_pol < len(polarization_labels):
                pol_label = polarization_labels[i_pol]
            else:
                pol_label = _format_pol_vec(polarizations[i_pol]
                        if i_pol < len(polarizations) else None)

            if verbose:
                print_info('surface_charge plot: wl={:.1f} nm, pol={}'.format(
                        wl, pol_label))

            f = plot_surface_charge_3d(
                    sc_dir, sigma, verts, faces, wl, i_pol, pol_label,
                    plot_format = plot_format, dpi = dpi)
            saved.extend(f)

            f = plot_surface_charge_2d_8views(
                    sc_dir, sigma, centroids, normals, wl, i_pol, pol_label,
                    plot_format = plot_format, dpi = dpi)
            saved.extend(f)

            f = plot_surface_charge_phase(
                    phase_dir, sigma, verts, faces, centroids, normals,
                    wl, i_pol, pol_label,
                    plot_format = plot_format, dpi = dpi)
            saved.extend(f)

    return saved


def plot_surface_charge_3d(out_dir: str,
        sigma: np.ndarray,
        verts: np.ndarray,
        faces: np.ndarray,
        wavelength: float,
        polarization_idx: int,
        polarization_label: str = '',
        norm_method: str = 'percentile',
        plot_format: Optional[List[str]] = None,
        dpi: int = 200) -> List[str]:
    """3D surface mesh with charge colormap (matplotlib Poly3DCollection).

    sigma : (nfaces,) complex-valued surface charge (sig2 from BEMRet).
    Real part is plotted with diverging colormap.
    """

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if plot_format is None:
        plot_format = ['png']

    ensure_dir(out_dir)

    sigma_real = np.real(np.asarray(sigma))

    faces_clean = _process_faces_for_plotting(faces)
    verts_tri = verts[faces_clean]

    charge_per_tri = _replicate_charge_for_split_faces(faces, sigma_real, faces_clean)
    charge_norm, vmin, vmax = _normalize_charge(charge_per_tri, norm_method)

    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111, projection = '3d')

    poly = Poly3DCollection(verts_tri, alpha = 0.9,
            edgecolor = 'k', linewidth = 0.2)
    poly.set_array(charge_norm)
    poly.set_cmap('RdBu_r')
    poly.set_clim(vmin, vmax)
    ax.add_collection3d(poly)

    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())

    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_zlabel('z (nm)')
    ax.set_title('Surface Charge (Re[sigma], norm={})\n'
            'lambda = {:.1f} nm, pol = {}'.format(
                    norm_method, wavelength, polarization_label))

    cbar = plt.colorbar(poly, ax = ax, pad = 0.1, shrink = 0.7)
    cbar.set_label('Normalized Re[sigma]')

    x_range = verts[:, 0].max() - verts[:, 0].min()
    y_range = verts[:, 1].max() - verts[:, 1].min()
    z_range = verts[:, 2].max() - verts[:, 2].min()
    max_range = max(x_range, y_range, z_range, 1e-10)
    ax.set_box_aspect([x_range / max_range, y_range / max_range, z_range / max_range])

    fig.tight_layout()

    base = 'surface_charge_3d_pol{}_lambda{:.0f}nm_{}'.format(
            polarization_idx + 1, wavelength, norm_method)

    saved = _save_figure(fig, out_dir, base, plot_format, dpi)
    plt.close(fig)
    return saved


def plot_surface_charge_2d_8views(out_dir: str,
        sigma: np.ndarray,
        centroids: np.ndarray,
        normals: Optional[np.ndarray],
        wavelength: float,
        polarization_idx: int,
        polarization_label: str = '',
        norm_method: str = 'percentile',
        plot_format: Optional[List[str]] = None,
        dpi: int = 200) -> List[str]:
    """Eight viewpoint 2D scatter projections (+x, -x, +y, -y, +z, -z, gap+, gap-).

    Reproduces MATLAB visualizer._plot_surface_charge_2d_8views.
    """

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if plot_format is None:
        plot_format = ['png']

    ensure_dir(out_dir)

    sigma_real = np.real(np.asarray(sigma))

    charge_norm, vmin, vmax = _normalize_charge(sigma_real, norm_method)

    fig, axes = plt.subplots(2, 4, figsize = (24, 12))

    standard_views = [
            ('+X view', 'x+', (1, 2), 'y (nm)', 'z (nm)'),
            ('-X view', 'x-', (1, 2), 'y (nm)', 'z (nm)'),
            ('+Y view', 'y+', (0, 2), 'x (nm)', 'z (nm)'),
            ('-Y view', 'y-', (0, 2), 'x (nm)', 'z (nm)'),
            ('+Z view', 'z+', (0, 1), 'x (nm)', 'y (nm)'),
            ('-Z view', 'z-', (0, 1), 'x (nm)', 'y (nm)')]

    scatter = None

    for idx, (view_name, direction, axes_idx, xlabel, ylabel) in enumerate(standard_views):
        ax = axes.flat[idx]

        outer_indices = _detect_outer_surface_faces(centroids, normals, direction)

        if len(outer_indices) > 0:
            xs = centroids[outer_indices, axes_idx[0]]
            ys = centroids[outer_indices, axes_idx[1]]
            cs = charge_norm[outer_indices]
            scatter = ax.scatter(xs, ys, c = cs, cmap = 'RdBu_r',
                    s = 50, vmin = vmin, vmax = vmax,
                    edgecolors = 'k', linewidth = 0.3)
        else:
            ax.text(0.5, 0.5, 'No outer surface\nfaces detected',
                    ha = 'center', va = 'center',
                    transform = ax.transAxes, fontsize = 12, color = 'gray')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(view_name, fontweight = 'bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha = 0.3)

    gap_faces = _detect_gap_facing_faces(centroids, normals)

    gap_view_configs = [
            ('Gap+ (P1->Gap)', gap_faces.get('particle1', []),
                    (1, 2), 'y (nm)', 'z (nm)'),
            ('Gap- (P2->Gap)', gap_faces.get('particle2', []),
                    (1, 2), 'y (nm)', 'z (nm)')]

    for idx, (view_name, face_indices, axes_idx, xlabel, ylabel) in enumerate(gap_view_configs):
        ax = axes.flat[6 + idx]

        face_indices = np.asarray(face_indices, dtype = int)

        if len(face_indices) > 0:
            xs = centroids[face_indices, axes_idx[0]]
            ys = centroids[face_indices, axes_idx[1]]
            cs = charge_norm[face_indices]
            scatter = ax.scatter(xs, ys, c = cs, cmap = 'RdBu_r',
                    s = 80, vmin = vmin, vmax = vmax,
                    edgecolors = 'k', linewidth = 0.3)
        else:
            ax.text(0.5, 0.5, 'No gap faces\ndetected',
                    ha = 'center', va = 'center',
                    transform = ax.transAxes, fontsize = 12, color = 'gray')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(view_name, fontweight = 'bold', color = 'darkred')
        ax.set_aspect('equal')
        ax.grid(True, alpha = 0.3)

    if scatter is not None:
        fig.subplots_adjust(right = 0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(scatter, cax = cbar_ax)
        cbar.set_label('Normalized Re[sigma]')

    fig.suptitle('Surface Charge Distribution - 8 Views\n'
            'lambda = {:.1f} nm, pol = {}, norm = {}'.format(
                    wavelength, polarization_label, norm_method),
            fontsize = 13, fontweight = 'bold')

    fig.tight_layout(rect = [0, 0.02, 0.92, 0.96])

    base = 'surface_charge_8views_pol{}_lambda{:.0f}nm_{}'.format(
            polarization_idx + 1, wavelength, norm_method)

    saved = _save_figure(fig, out_dir, base, plot_format, dpi)
    plt.close(fig)
    return saved


def plot_surface_charge_phase(out_dir: str,
        sigma: np.ndarray,
        verts: np.ndarray,
        faces: np.ndarray,
        centroids: np.ndarray,
        normals: Optional[np.ndarray],
        wavelength: float,
        polarization_idx: int,
        polarization_label: str = '',
        plot_format: Optional[List[str]] = None,
        dpi: int = 200) -> List[str]:
    """Re/Im/Amplitude/Phase 8-view scatter plots for Fano analysis.

    Reproduces MATLAB visualizer.plot_surface_charge_phase_analysis (lite).
    """

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if plot_format is None:
        plot_format = ['png']

    ensure_dir(out_dir)

    sigma_complex = np.asarray(sigma, dtype = complex)

    components = [
            ('real', 'Re[sigma]', 'RdBu_r', True),
            ('imag', 'Im[sigma]', 'RdBu_r', True),
            ('amplitude', '|sigma|', 'hot', False),
            ('phase', 'arg(sigma)', 'twilight', False)]

    saved = []

    gap_faces = _detect_gap_facing_faces(centroids, normals)

    comp_arrs = {
            'real': np.real(sigma_complex),
            'imag': np.imag(sigma_complex),
            'amplitude': np.abs(sigma_complex),
            'phase': np.angle(sigma_complex)}

    for comp_key, comp_label, cmap, symmetric in components:
        comp_data = comp_arrs[comp_key]

        if comp_key == 'phase':
            plot_data = comp_data
            vmin, vmax = -np.pi, np.pi
            cbar_label = 'Phase (rad)'
        elif symmetric:
            p95 = np.percentile(np.abs(comp_data), 95)
            if p95 < 1e-30:
                p95 = float(np.max(np.abs(comp_data))) + 1e-30
            plot_data = np.clip(comp_data / p95, -1, 1)
            vmin, vmax = -1, 1
            cbar_label = 'Normalized {}'.format(comp_label)
        else:
            p95 = np.percentile(comp_data, 95)
            if p95 < 1e-30:
                p95 = float(np.max(comp_data)) + 1e-30
            plot_data = np.clip(comp_data / p95, 0, 1)
            vmin, vmax = 0, 1
            cbar_label = 'Normalized {}'.format(comp_label)

        fig, axes = plt.subplots(2, 4, figsize = (24, 12))

        standard_views = [
                ('+X view', 'x+', (1, 2), 'y (nm)', 'z (nm)'),
                ('-X view', 'x-', (1, 2), 'y (nm)', 'z (nm)'),
                ('+Y view', 'y+', (0, 2), 'x (nm)', 'z (nm)'),
                ('-Y view', 'y-', (0, 2), 'x (nm)', 'z (nm)'),
                ('+Z view', 'z+', (0, 1), 'x (nm)', 'y (nm)'),
                ('-Z view', 'z-', (0, 1), 'x (nm)', 'y (nm)')]

        scatter = None

        for idx, (view_name, direction, axes_idx, xlabel, ylabel) in enumerate(standard_views):
            ax = axes.flat[idx]
            outer_indices = _detect_outer_surface_faces(centroids, normals, direction)

            if len(outer_indices) > 0:
                xs = centroids[outer_indices, axes_idx[0]]
                ys = centroids[outer_indices, axes_idx[1]]
                cs = plot_data[outer_indices]
                scatter = ax.scatter(xs, ys, c = cs, cmap = cmap,
                        s = 50, vmin = vmin, vmax = vmax,
                        edgecolors = 'k', linewidth = 0.3)
            else:
                ax.text(0.5, 0.5, 'No outer surface\nfaces detected',
                        ha = 'center', va = 'center',
                        transform = ax.transAxes, fontsize = 12, color = 'gray')

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(view_name, fontweight = 'bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha = 0.3)

        gap_view_configs = [
                ('Gap+ (P1->Gap)', gap_faces.get('particle1', []),
                        (1, 2), 'y (nm)', 'z (nm)'),
                ('Gap- (P2->Gap)', gap_faces.get('particle2', []),
                        (1, 2), 'y (nm)', 'z (nm)')]

        for idx, (view_name, face_indices, axes_idx, xlabel, ylabel) in enumerate(gap_view_configs):
            ax = axes.flat[6 + idx]
            face_indices = np.asarray(face_indices, dtype = int)

            if len(face_indices) > 0:
                xs = centroids[face_indices, axes_idx[0]]
                ys = centroids[face_indices, axes_idx[1]]
                cs = plot_data[face_indices]
                scatter = ax.scatter(xs, ys, c = cs, cmap = cmap,
                        s = 80, vmin = vmin, vmax = vmax,
                        edgecolors = 'k', linewidth = 0.3)
            else:
                ax.text(0.5, 0.5, 'No gap faces\ndetected',
                        ha = 'center', va = 'center',
                        transform = ax.transAxes, fontsize = 12, color = 'gray')

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(view_name, fontweight = 'bold', color = 'darkred')
            ax.set_aspect('equal')
            ax.grid(True, alpha = 0.3)

        if scatter is not None:
            fig.subplots_adjust(right = 0.92)
            cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
            cbar = fig.colorbar(scatter, cax = cbar_ax)
            cbar.set_label(cbar_label)

        fig.suptitle('Surface Charge {} - 8 Views\n'
                'lambda = {:.1f} nm, pol = {}'.format(
                        comp_label, wavelength, polarization_label),
                fontsize = 13, fontweight = 'bold')

        fig.tight_layout(rect = [0, 0.02, 0.92, 0.96])

        base = 'charge_{}_8views_pol{}_lambda{:.0f}nm'.format(
                comp_key, polarization_idx + 1, wavelength)

        files = _save_figure(fig, out_dir, base, plot_format, dpi)
        saved.extend(files)
        plt.close(fig)

    return saved


# ---------------------------------------------------------------------------
# helpers (mirror of MATLAB visualizer.py utilities)
# ---------------------------------------------------------------------------


def _format_pol_vec(vec: Any) -> str:

    if vec is None:
        return ''

    arr = np.asarray(vec).flatten()

    if arr.size < 3:
        return str(arr.tolist())

    return '[{:.0f} {:.0f} {:.0f}]'.format(
            float(arr[0]), float(arr[1]), float(arr[2]))


def _save_figure(fig: Any,
        out_dir: str,
        base_filename: str,
        plot_format: List[str],
        dpi: int) -> List[str]:

    saved = []

    for fmt in plot_format:
        path = os.path.join(out_dir, '{}.{}'.format(base_filename, fmt))
        fig.savefig(path, dpi = dpi, bbox_inches = 'tight')
        saved.append(path)

    return saved


def _normalize_charge(charge: np.ndarray,
        method: str = 'percentile') -> Tuple[np.ndarray, float, float]:

    if method == 'linear':
        cmax = float(np.max(np.abs(charge)))
        norm = charge / (cmax + 1e-30)
        return norm, -1.0, 1.0

    if method == 'percentile':
        p95 = float(np.percentile(np.abs(charge), 95))
        if p95 < 1e-30:
            p95 = float(np.max(np.abs(charge))) + 1e-30
        norm = np.clip(charge / p95, -1, 1)
        return norm, -1.0, 1.0

    if method == 'power':
        cmax = float(np.max(np.abs(charge)))
        base = charge / (cmax + 1e-30)
        gamma = 0.2
        norm = np.sign(base) * np.abs(base) ** gamma
        return norm, -1.0, 1.0

    raise ValueError('[error] Unknown norm_method <{}>!'.format(method))


def _process_faces_for_plotting(faces: np.ndarray) -> np.ndarray:
    """Convert MNPBEM faces (triangle or quad with NaN) to triangles.

    pymnpbem faces are 0-indexed (Particle.__init__ already converts).
    """

    faces_arr = np.asarray(faces)
    triangles = []

    if faces_arr.ndim == 1 or faces_arr.shape[1] == 3:

        for face in faces_arr:
            triangles.append([int(face[0]), int(face[1]), int(face[2])])

        return np.asarray(triangles, dtype = int)

    for face in faces_arr:

        if faces_arr.shape[1] == 4 and not np.isnan(face[3]):
            triangles.append([int(face[0]), int(face[1]), int(face[2])])
            triangles.append([int(face[0]), int(face[2]), int(face[3])])
        else:
            triangles.append([int(face[0]), int(face[1]), int(face[2])])

    return np.asarray(triangles, dtype = int)


def _replicate_charge_for_split_faces(faces: np.ndarray,
        charge: np.ndarray,
        faces_clean: np.ndarray) -> np.ndarray:

    if len(faces_clean) == len(charge):
        return charge

    out = []

    for i, face in enumerate(faces):
        out.append(charge[i])

        if faces.shape[1] == 4 and not np.isnan(face[3]):
            out.append(charge[i])

    return np.asarray(out)


def _detect_gap_facing_faces(centroids: np.ndarray,
        normals: Optional[np.ndarray]) -> Dict[str, np.ndarray]:

    if normals is None or normals.size == 0:
        return {'particle1': np.asarray([], dtype = int),
                'particle2': np.asarray([], dtype = int)}

    x = centroids[:, 0]
    gap_x = 0.5 * (x.min() + x.max())

    n_norm = normals / (np.linalg.norm(normals, axis = 1, keepdims = True) + 1e-30)

    p1_dir = np.array([1.0, 0.0, 0.0])
    p2_dir = np.array([-1.0, 0.0, 0.0])

    p1_mask = (x < gap_x) & (n_norm @ p1_dir > 0.5)
    p2_mask = (x > gap_x) & (n_norm @ p2_dir > 0.5)

    return {'particle1': np.where(p1_mask)[0],
            'particle2': np.where(p2_mask)[0]}


def _detect_outer_surface_faces(centroids: np.ndarray,
        normals: Optional[np.ndarray],
        direction: str,
        position_threshold: Optional[float] = None) -> np.ndarray:

    if position_threshold is None:
        extent = centroids.max(axis = 0) - centroids.min(axis = 0)
        position_threshold = max(float(np.min(extent)) * 0.15, 1.0)

    direction_config = {
            'x+': (0, np.array([1.0, 0.0, 0.0]), 1),
            'x-': (0, np.array([-1.0, 0.0, 0.0]), -1),
            'y+': (1, np.array([0.0, 1.0, 0.0]), 1),
            'y-': (1, np.array([0.0, -1.0, 0.0]), -1),
            'z+': (2, np.array([0.0, 0.0, 1.0]), 1),
            'z-': (2, np.array([0.0, 0.0, -1.0]), -1)}

    if direction not in direction_config:
        raise ValueError('[error] Invalid direction <{}>!'.format(direction))

    axis, normal_dir, sign = direction_config[direction]
    coords = centroids[:, axis]

    if sign > 0:
        position_mask = coords >= (coords.max() - position_threshold)
    else:
        position_mask = coords <= (coords.min() + position_threshold)

    if normals is None or normals.size == 0:
        return np.where(position_mask)[0]

    n_norm = normals / (np.linalg.norm(normals, axis = 1, keepdims = True) + 1e-30)

    normal_mask = (n_norm @ normal_dir) > 0.5

    return np.where(position_mask & normal_mask)[0]


# ---------------------------------------------------------------------------
# loader (used by reanalyze CLI)
# ---------------------------------------------------------------------------


def load_surface_charge_from_npz(npz: Any) -> Optional[Dict[str, Any]]:
    """Load surface_charge dict from numpy npz handle.

    Returns None if the npz has no surface_charge keys.
    """

    if 'surface_charge_sig2' not in npz.files:
        return None

    return {
            'wavelengths': np.asarray(npz['surface_charge_wavelengths']),
            'wl_indices': np.asarray(npz['surface_charge_wl_indices']),
            'sig2': np.asarray(npz['surface_charge_sig2']),
            'sig1': np.asarray(npz['surface_charge_sig1']),
            'verts': np.asarray(npz['surface_charge_verts']),
            'faces': np.asarray(npz['surface_charge_faces']),
            'centroids': np.asarray(npz['surface_charge_centroids']),
            'normals': np.asarray(npz['surface_charge_normals']),
            'areas': np.asarray(npz['surface_charge_areas']),
            'polarizations': np.asarray(npz['surface_charge_polarizations'])}
