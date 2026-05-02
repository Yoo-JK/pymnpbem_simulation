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
