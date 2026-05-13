import os
import sys

from typing import Any, Dict, Optional

import numpy as np
from box import Box


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
