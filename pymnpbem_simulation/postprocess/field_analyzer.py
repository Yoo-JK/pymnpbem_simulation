import os
import sys

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from box import Box

from .edge_filter import find_edge_artifacts
from .geometry_cross_section import GeometryCrossSection


def hotspot_location(field_result: Any,
        threshold_quantile: float = 0.99) -> Box:

    e = np.asarray(field_result['e'] if isinstance(field_result, dict) else field_result.e)
    pos = np.asarray(field_result['pos'] if isinstance(field_result, dict) else field_result.pos)

    e2 = np.sum(np.abs(e) ** 2, axis = -1)

    while e2.ndim > 1:
        e2 = e2.mean(axis = -1)

    if e2.shape[0] != pos.shape[0]:
        e2 = e2.reshape(pos.shape[0], -1).mean(axis = 1)

    threshold = float(np.quantile(e2, threshold_quantile))
    mask = e2 >= threshold

    max_idx = int(np.argmax(e2))

    return Box({
            'positions': pos[mask],
            'intensities': e2[mask],
            'max_pos': pos[max_idx],
            'max_intensity': float(e2[max_idx]),
            'threshold': threshold,
            'n_hotspots': int(mask.sum())})


def field_enhancement(field_result: Any,
        e_inc: Any) -> np.ndarray:

    e = np.asarray(field_result['e'] if isinstance(field_result, dict) else field_result.e)
    e_inc_arr = np.asarray(e_inc)

    e2 = np.sum(np.abs(e) ** 2, axis = -1)
    e_inc2 = float(np.sum(np.abs(e_inc_arr) ** 2))

    if e_inc2 < 1e-30:
        e_inc2 = 1.0

    return e2 / e_inc2


def near_field_decay(field_result: Any,
        surface_pos: np.ndarray) -> Box:

    pos = np.asarray(field_result['pos'] if isinstance(field_result, dict) else field_result.pos)
    e = np.asarray(field_result['e'] if isinstance(field_result, dict) else field_result.e)
    surface_pos = np.asarray(surface_pos)

    diff = pos[:, None, :] - surface_pos[None, :, :]
    distances = np.linalg.norm(diff, axis = 2).min(axis = 1)

    e2 = np.sum(np.abs(e) ** 2, axis = -1)
    while e2.ndim > 1:
        e2 = e2.mean(axis = -1)

    order = np.argsort(distances)

    return Box({
            'distances': distances[order],
            'e2': e2[order]})


def integrated_field_intensity(field_result: Any) -> float:

    e = np.asarray(field_result['e'] if isinstance(field_result, dict) else field_result.e)
    e2 = np.sum(np.abs(e) ** 2, axis = -1)
    return float(e2.sum())


def hotspot_summary(field_result: Any,
        threshold_quantile: float = 0.99) -> Dict[str, Any]:

    hot = hotspot_location(field_result, threshold_quantile = threshold_quantile)

    return {
            'n_hotspots': int(hot.n_hotspots),
            'max_intensity': float(hot.max_intensity),
            'max_pos': hot.max_pos.tolist() if hasattr(hot.max_pos, 'tolist') else list(hot.max_pos),
            'threshold': float(hot.threshold),
            'threshold_quantile': float(threshold_quantile)}


def field_statistics(field_result: Any) -> Dict[str, float]:
    """Compute summary statistics over |E|^2 (or enhancement when supplied).

    Returns dict with: max / min / mean / median / std / percentiles 90/95/99.
    """
    e = np.asarray(field_result['e'] if isinstance(field_result, dict) else field_result.e)
    e2 = np.sum(np.abs(e) ** 2, axis = -1)
    flat = e2.flatten()
    flat = flat[np.isfinite(flat)]

    if flat.size == 0:
        return {
                'max': 0.0, 'min': 0.0, 'mean': 0.0, 'median': 0.0, 'std': 0.0,
                'percentile_90': 0.0, 'percentile_95': 0.0, 'percentile_99': 0.0,
                'n_points': 0}

    return {
            'max': float(np.max(flat)),
            'min': float(np.min(flat)),
            'mean': float(np.mean(flat)),
            'median': float(np.median(flat)),
            'std': float(np.std(flat)),
            'percentile_90': float(np.percentile(flat, 90)),
            'percentile_95': float(np.percentile(flat, 95)),
            'percentile_99': float(np.percentile(flat, 99)),
            'n_points': int(flat.size)}


def high_field_regions(field_result: Any,
        thresholds: Optional[list] = None,
        e_inc: Optional[Any] = None) -> Dict[str, Any]:
    """Count points and estimate area/volume above each enhancement threshold.

    Args:
        field_result: dict-like with 'e' and 'pos' arrays.
        thresholds: list of enhancement thresholds (default [2,5,10,20,50,100]).
        e_inc: incident field amplitude vector. If None, uses |E|^2 directly
            (no normalization).

    Returns:
        dict {
            'thresholds': [...],
            'regions': {threshold -> {'n_points', 'fraction', 'area_nm2' or 'volume_nm3'}},
            'detected_dim': 2|3|None,
            'spacing': [dx, dy, dz] or [dx, dy] (None if non-uniform)
        }
    """
    if thresholds is None:
        thresholds = [2, 5, 10, 20, 50, 100]

    e = np.asarray(field_result['e'] if isinstance(field_result, dict) else field_result.e)
    pos = np.asarray(field_result['pos'] if isinstance(field_result, dict) else field_result.pos)

    e2 = np.sum(np.abs(e) ** 2, axis = -1)
    while e2.ndim > 1:
        e2 = e2.mean(axis = -1)

    if e_inc is not None:
        e_inc_arr = np.asarray(e_inc)
        e_inc2 = float(np.sum(np.abs(e_inc_arr) ** 2))
        if e_inc2 < 1e-30:
            e_inc2 = 1.0
        enhancement = e2 / e_inc2
    else:
        enhancement = e2

    flat = enhancement.flatten()
    flat = flat[np.isfinite(flat)]
    total = flat.size

    # Try to detect uniform-grid spacing from pos.
    detected_dim = None
    spacing = None
    element_size = None

    if pos.ndim == 2 and pos.shape[1] >= 3 and pos.shape[0] > 1:
        # Count unique values per axis (within tolerance).
        u_per_axis = []
        for ax in range(3):
            u_per_axis.append(np.unique(np.round(pos[:, ax], 6)).size)

        # 2D slice: 1 axis collapsed (all values equal).
        n_var_axes = sum(1 for u in u_per_axis if u > 1)

        if n_var_axes == 2:
            detected_dim = 2
            spacing = []
            for ax, n_u in enumerate(u_per_axis):
                if n_u > 1:
                    coords = np.unique(np.round(pos[:, ax], 6))
                    diffs = np.diff(coords)
                    if diffs.size > 0:
                        spacing.append(float(np.median(diffs)))
                    else:
                        spacing.append(1.0)
            if len(spacing) == 2:
                element_size = float(spacing[0] * spacing[1])
        elif n_var_axes == 3:
            detected_dim = 3
            spacing = []
            for ax in range(3):
                coords = np.unique(np.round(pos[:, ax], 6))
                diffs = np.diff(coords)
                if diffs.size > 0:
                    spacing.append(float(np.median(diffs)))
                else:
                    spacing.append(1.0)
            element_size = float(spacing[0] * spacing[1] * spacing[2])

    out = {
            'thresholds': list(thresholds),
            'regions': dict(),
            'detected_dim': detected_dim,
            'spacing': spacing}

    for thr in thresholds:
        mask = flat > float(thr)
        n_pts = int(np.sum(mask))
        entry = {
                'n_points': n_pts,
                'fraction': float(n_pts / total) if total > 0 else 0.0}

        if element_size is not None:
            if detected_dim == 2:
                entry['area_nm2'] = float(n_pts * element_size)
            elif detected_dim == 3:
                entry['volume_nm3'] = float(n_pts * element_size)

        out['regions']['enhancement_above_{}'.format(thr)] = entry

    return out


# ---------------------------------------------------------------------------
# Near-field integration  (ported from OLD post_utils/field_analyzer.py)
# ---------------------------------------------------------------------------

_DEFAULT_DEPTHS = [2.0, 15.0]
_SUPPORTED_STRUCTURES = {'sphere', 'sphere_cluster', 'sphere_cluster_aggregate',
                         'dimer_sphere', 'dimer'}


def calculate_near_field_integration(field_data_list: List[Dict],
        config: Dict,
        geometry: Optional[GeometryCrossSection] = None,
        center_only: bool = False,
        depths: Optional[List[float]] = None,
        verbose: bool = False) -> Optional[Dict]:
    """Integrate |E|²/|E0|² inside particle volumes at several interior depths.

    Applies a hybrid edge-artifact filter (see :mod:`edge_filter`) to remove
    BEM boundary spikes before summing.

    Parameters
    ----------
    field_data_list : list of dict
        Each dict must contain keys ``enhancement`` (2-D ndarray), ``x_grid``,
        ``y_grid``, ``z_grid``, ``wavelength``, and optionally
        ``intensity``, ``e_sq``, ``e0_sq``, ``e_sq_int``,
        ``polarization_idx``.
    config : dict
        Simulation configuration — must have ``'structure'`` and geometry params.
    geometry : GeometryCrossSection, optional
        Pre-built geometry object.  Created from *config* when not supplied.
    center_only : bool
        Cluster structures only — integrate the centre sphere alone.
    depths : list of float, optional
        Interior integration depths in nm (default ``[2.0, 15.0]``).
    verbose : bool

    Returns
    -------
    dict or None
        ``{wavelength: {pol_key: {'depths': {depth: {'strict': ...,
        'conservative': ..., 'n_spheres': int, 'n_artifacts_removed': int}},
        'grid_info': ...}}}``
        Returns *None* if the structure is not supported.
    """
    if depths is None:
        depths = list(_DEFAULT_DEPTHS)

    structure = config.get('structure', 'unknown')
    if structure not in _SUPPORTED_STRUCTURES:
        return None

    if geometry is None:
        geometry = GeometryCrossSection(config, verbose=verbose)

    results = {}

    for field_data in field_data_list:
        wl = float(field_data['wavelength'])
        pol_idx = field_data.get('polarization_idx')

        integration = _integrate_single_field(
            field_data, config, geometry, center_only=center_only,
            depths=depths, verbose=verbose)

        if wl not in results:
            results[wl] = {}

        key = 'polarization_{}'.format(pol_idx + 1) if pol_idx is not None else 'unpolarized'
        results[wl][key] = integration

    return results


def _integrate_single_field(field_data: Dict,
        config: Dict,
        geometry: GeometryCrossSection,
        center_only: bool,
        depths: List[float],
        verbose: bool) -> Dict:
    """Integrate one wavelength/polarisation at each depth."""
    enhancement = np.asarray(field_data['enhancement'], dtype=float)
    intensity = field_data.get('intensity')
    if intensity is not None:
        intensity = np.asarray(intensity, dtype=float)
    x_grid = np.asarray(field_data['x_grid'], dtype=float)
    y_grid = np.asarray(field_data['y_grid'], dtype=float)
    z_grid = np.asarray(field_data['z_grid'], dtype=float)

    e_sq = field_data.get('e_sq')
    e0_sq = field_data.get('e0_sq')
    e_sq_int = field_data.get('e_sq_int')
    if e_sq is not None:
        e_sq = np.asarray(e_sq, dtype=float)
    if e0_sq is not None:
        e0_sq = np.asarray(e0_sq, dtype=float)
    if e_sq_int is not None:
        e_sq_int = np.asarray(e_sq_int, dtype=float)

    # Take abs of complex arrays
    if np.iscomplexobj(enhancement):
        enhancement = np.abs(enhancement)
    if intensity is not None and np.iscomplexobj(intensity):
        intensity = np.abs(intensity)

    spheres = _get_sphere_boundaries(config, geometry, center_only=center_only, verbose=verbose)
    if not spheres:
        return {'depths': {}, 'grid_info': {}}

    n_spheres = len(spheres)
    results_by_depth = {}

    for depth in depths:
        dist_mask = _distance_mask(x_grid, y_grid, z_grid, spheres, depth)

        artifact_mask, n_art = find_edge_artifacts(
            enhancement, x_grid, y_grid, z_grid, spheres,
            mask=dist_mask, edge_threshold=1.0, isolation_ratio=1.3,
            verbose=verbose)

        strict = _filter_and_sum(enhancement, intensity, dist_mask, n_spheres,
                                 e_sq=e_sq, e0_sq=e0_sq, e_sq_int=e_sq_int,
                                 method='strict', artifact_mask=artifact_mask)
        cons = _filter_and_sum(enhancement, intensity, dist_mask, n_spheres,
                               e_sq=e_sq, e0_sq=e0_sq, e_sq_int=e_sq_int,
                               method='conservative', artifact_mask=artifact_mask)

        results_by_depth[depth] = {
            'strict': strict,
            'conservative': cons,
            'n_spheres': n_spheres,
            'n_artifacts_removed': n_art,
        }

    grid_info = {
        'total_points': int(enhancement.size),
        'valid_points': int(np.sum(np.isfinite(enhancement))),
    }
    return {'depths': results_by_depth, 'grid_info': grid_info}


def _distance_mask(x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        spheres: list,
        depth: float) -> np.ndarray:
    """Boolean mask: inside any sphere AND within *depth* nm from its surface."""
    mask = np.zeros(x_grid.shape, dtype=bool)
    for cx, cy, cz, radius in spheres:
        dist_from_center = np.sqrt(
            (x_grid - cx) ** 2 + (y_grid - cy) ** 2 + (z_grid - cz) ** 2)
        dist_from_surface = dist_from_center - radius
        mask |= (dist_from_surface <= 0) & (dist_from_surface >= -depth)
    return mask


def _filter_and_sum(enhancement: np.ndarray,
        intensity: Optional[np.ndarray],
        distance_mask: np.ndarray,
        n_spheres: int,
        *,
        e_sq: Optional[np.ndarray],
        e0_sq: Optional[np.ndarray],
        e_sq_int: Optional[np.ndarray],
        method: str,
        artifact_mask: Optional[np.ndarray]) -> Dict:
    """Sum enhancement/intensity inside *distance_mask* with specified filtering."""
    final_mask = distance_mask.copy()
    final_mask &= np.isfinite(enhancement)

    excluded_outliers = 0
    if method == 'conservative':
        valid = final_mask & np.isfinite(enhancement)
        if valid.any():
            threshold = float(np.percentile(enhancement[valid], 99.9))
            outlier_mask = enhancement <= threshold * 10
            excluded_outliers = int(np.sum(final_mask & ~outlier_mask))
            final_mask &= outlier_mask

    excluded_artifacts = 0
    if artifact_mask is not None:
        excluded_artifacts = int(np.sum(final_mask & artifact_mask))
        final_mask &= ~artifact_mask

    enh_vals = enhancement[final_mask]
    if enh_vals.size == 0:
        result = {
            'enhancement_sum': 0.0, 'enhancement_mean': 0.0,
            'enhancement_per_sphere': 0.0,
            'intensity_sum': 0.0, 'intensity_mean': 0.0,
            'intensity_per_sphere': 0.0,
            'e_sq_sum': None, 'e0_sq_sum': None,
            'energy_ratio': None, 'energy_ratio_per_sphere': None,
            'valid_points': 0, 'excluded_artifacts': excluded_artifacts,
        }
        if method == 'conservative':
            result['excluded_outliers'] = excluded_outliers
        return result

    enh_sum = float(enh_vals.sum())
    enh_mean = float(enh_vals.mean())
    enh_per_sphere = enh_sum / n_spheres if n_spheres > 0 else 0.0

    result = {
        'enhancement_sum': enh_sum,
        'enhancement_mean': enh_mean,
        'enhancement_per_sphere': enh_per_sphere,
        'valid_points': int(final_mask.sum()),
        'excluded_artifacts': excluded_artifacts,
    }

    if intensity is not None:
        int_vals = intensity[final_mask]
        int_sum = float(int_vals.sum())
        result.update({
            'intensity_sum': int_sum,
            'intensity_mean': float(int_vals.mean()),
            'intensity_per_sphere': int_sum / n_spheres if n_spheres > 0 else 0.0,
        })
    else:
        result.update({'intensity_sum': 0.0, 'intensity_mean': 0.0, 'intensity_per_sphere': 0.0})

    # Energy ratio: Σ|E|² / Σ|E0|²
    e_sq_use = e_sq_int if e_sq_int is not None else e_sq
    if e_sq_use is not None and e0_sq is not None:
        es_vals = e_sq_use[final_mask]
        e0_vals = e0_sq[final_mask]
        both_valid = np.isfinite(es_vals) & np.isfinite(e0_vals)
        if both_valid.any():
            es_sum = float(es_vals[both_valid].sum())
            e0_sum = float(e0_vals[both_valid].sum())
            ratio = es_sum / e0_sum if e0_sum > 1e-20 else None
            result.update({
                'e_sq_sum': es_sum, 'e0_sq_sum': e0_sum,
                'energy_ratio': ratio,
                'energy_ratio_per_sphere': ratio / n_spheres if (ratio is not None and n_spheres > 0) else None,
            })
        else:
            result.update({'e_sq_sum': None, 'e0_sq_sum': None,
                           'energy_ratio': None, 'energy_ratio_per_sphere': None})
    else:
        result.update({'e_sq_sum': None, 'e0_sq_sum': None,
                       'energy_ratio': None, 'energy_ratio_per_sphere': None})

    if method == 'conservative':
        result['excluded_outliers'] = excluded_outliers

    return result


def _get_sphere_boundaries(config: Dict,
        geometry: GeometryCrossSection,
        center_only: bool,
        verbose: bool) -> list:
    """Return list of (cx, cy, cz, radius) from config."""
    structure = config.get('structure', 'unknown')

    if structure in ('sphere_cluster_aggregate', 'sphere_cluster'):
        n = int(config.get('n_spheres', 1))
        diameter = float(config.get('diameter', 50.0))
        gap = float(config.get('gap', -0.1))
        radius = diameter / 2
        spacing = diameter + gap
        positions = geometry._calculate_cluster_positions(n, spacing)
        spheres = [(p[0], p[1], p[2], radius) for p in positions]
        if center_only:
            spheres = [spheres[0]]
        return spheres

    elif structure == 'sphere':
        radius = float(config.get('radius', config.get('diameter', 20.0) / 2))
        center = config.get('center', [0, 0, 0])
        return [(center[0], center[1], center[2], radius)]

    elif structure in ('dimer_sphere', 'dimer'):
        radius = float(config.get('radius', config.get('diameter', 20.0) / 2))
        gap = float(config.get('gap', 5.0))
        spacing = radius * 2 + gap
        offset = spacing / 2
        return [(-offset, 0.0, 0.0, radius), (offset, 0.0, 0.0, radius)]

    if verbose:
        print('[field_analyzer] _get_sphere_boundaries: unsupported structure "{}"'.format(structure))
    return []


def save_near_field_results(results: Dict,
        config: Dict,
        output_path: str,
        center_only: bool = False,
        depths: Optional[List[float]] = None) -> None:
    """Write near-field integration results to a text file.

    Parameters
    ----------
    results : dict
        Output of :func:`calculate_near_field_integration`.
    config : dict
        Simulation configuration (for header information).
    output_path : str
        Destination file path.
    center_only : bool
        If True, annotate header as "centre sphere only".
    depths : list of float, optional
        Integration depths used (for header; inferred from results if None).
    """
    if depths is None:
        # infer from first result entry
        try:
            first_wl = next(iter(results.values()))
            first_pol = next(iter(first_wl.values()))
            depths = sorted(first_pol.get('depths', {}).keys())
        except StopIteration:
            depths = list(_DEFAULT_DEPTHS)

    import os
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w') as f:
        # Header
        title_parts = ['Near-Field Integration Analysis (INTERIOR)']
        if center_only:
            title_parts.append('CENTER SPHERE ONLY')
        title_parts.append('HYBRID EDGE FILTER')
        f.write('=' * 80 + '\n')
        f.write(' - '.join(title_parts) + '\n')
        f.write('=' * 80 + '\n\n')
        f.write('Configuration:\n')
        f.write('  Integration depths: {} nm (interior)\n'.format(
            ', '.join('{:.1f}'.format(d) for d in depths)))
        f.write('  Structure: {}\n'.format(config.get('structure', 'unknown')))
        f.write('\n' + '=' * 80 + '\n\n')

        # Per-wavelength results
        for wl in sorted(results.keys()):
            f.write('Results at wavelength = {:.1f} nm:\n'.format(wl))
            f.write('-' * 80 + '\n')
            for pol_key in sorted(results[wl].keys()):
                pol_data = results[wl][pol_key]
                pol_label = 'Unpolarized' if pol_key == 'unpolarized' else 'Polarization {}'.format(pol_key.split('_')[1])
                f.write('{}\n'.format(pol_label))
                f.write('-' * 80 + '\n')
                if 'grid_info' in pol_data:
                    gi = pol_data['grid_info']
                    f.write('  Total grid points:   {}\n'.format(gi.get('total_points', '?')))
                    f.write('  Valid points:        {}\n\n'.format(gi.get('valid_points', '?')))
                for depth in sorted(pol_data.get('depths', {}).keys()):
                    dd = pol_data['depths'][depth]
                    f.write('  Depth: {:.1f} nm | spheres: {}\n'.format(
                        depth, dd.get('n_spheres', '?')))
                    for flt in ('strict', 'conservative'):
                        r = dd[flt]
                        f.write('    [{}]  enh_sum={:.3f}  per_sphere={:.3f}  '
                                'pts={}  energy_ratio={}\n'.format(
                                    flt, r['enhancement_sum'], r['enhancement_per_sphere'],
                                    r['valid_points'],
                                    '{:.6f}'.format(r['energy_ratio']) if r.get('energy_ratio') is not None else 'N/A'))
                    f.write('\n')

        # Summary table
        f.write('=' * 100 + '\n')
        f.write('Summary (Strict Filtering)\n')
        f.write('=' * 100 + '\n')
        f.write('{:<12} {:<15} {:<8} {:<15} {:<15} {:<15} {:<10}\n'.format(
            'Wavelength', 'Polarization', 'Depth', 'Enh.Sum', 'Int.Sum',
            'Energy Ratio', 'Points'))
        f.write('-' * 100 + '\n')
        for wl in sorted(results.keys()):
            for pol_key in sorted(results[wl].keys()):
                pol_data = results[wl][pol_key]
                pol_str = 'unpolarized' if pol_key == 'unpolarized' else 'pol{}'.format(pol_key.split('_')[1])
                for depth in sorted(pol_data.get('depths', {}).keys()):
                    r = pol_data['depths'][depth]['strict']
                    er_str = '{:.6f}'.format(r['energy_ratio']) if r.get('energy_ratio') is not None else 'N/A'
                    f.write('{:<12} {:<15} {:<8} {:<15.3f} {:<15.3f} {:<15} {:<10d}\n'.format(
                        '{:.1f} nm'.format(wl), pol_str, '{:.1f}nm'.format(depth),
                        r['enhancement_sum'],
                        r.get('intensity_sum') or 0.0,
                        er_str,
                        r['valid_points']))
        f.write('\n' + '=' * 100 + '\n')
