"""
Hybrid Edge + Isolation Filter for BEM Artifact Removal

Removes numerical artifacts at particle boundaries (BEM edge singularities)
while preserving physically meaningful features like gap hotspots.

Filter conditions (AND):
  1. Edge zone: pixel is within edge_threshold nm from nearest sphere surface
  2. Spatially isolated: pixel value > isolation_ratio * local_median (in-mask neighbors)

Ported from OLD post_utils/edge_filter.py.
"""

import numpy as np


def get_sphere_boundaries_from_config(config: dict) -> list:
    """Extract 3-D sphere centre positions and radii from a simulation config.

    Supports: sphere, dimer_sphere, dimer, sphere_cluster, sphere_cluster_aggregate.

    Parameters
    ----------
    config : dict
        Simulation configuration with 'structure', 'diameter'/'radius', etc.

    Returns
    -------
    list of (cx, cy, cz, radius) or None if structure not supported.
    """
    structure = config.get('structure', 'unknown')
    radius = _get_radius(config)

    if structure in ('sphere_cluster_aggregate', 'sphere_cluster'):
        n_spheres = config.get('n_spheres', 1)
        diameter = radius * 2
        gap = config.get('gap', -0.1)
        spacing = diameter + gap
        positions = _cluster_positions(n_spheres, spacing)
        return [(p[0], p[1], p[2], radius) for p in positions]

    elif structure == 'sphere':
        center = config.get('center', [0, 0, 0])
        return [(center[0], center[1], center[2], radius)]

    elif structure in ('dimer_sphere', 'dimer'):
        gap = config.get('gap', 2.0)
        center = config.get('center', [0, 0, 0])
        dimer_axis = config.get('dimer_axis', 'x')
        offset = radius + gap / 2

        if dimer_axis == 'x':
            return [(center[0] - offset, center[1], center[2], radius),
                    (center[0] + offset, center[1], center[2], radius)]
        elif dimer_axis == 'y':
            return [(center[0], center[1] - offset, center[2], radius),
                    (center[0], center[1] + offset, center[2], radius)]
        elif dimer_axis == 'z':
            return [(center[0], center[1], center[2] - offset, radius),
                    (center[0], center[1], center[2] + offset, radius)]
        else:
            return [(center[0] - offset, center[1], center[2], radius),
                    (center[0] + offset, center[1], center[2], radius)]
    else:
        return None


def compute_min_surface_distance(x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        spheres: list) -> np.ndarray:
    """Compute minimum absolute distance from the nearest sphere surface.

    Parameters
    ----------
    x_grid, y_grid, z_grid : ndarray
        Coordinate grids (same shape as field data).
    spheres : list of (cx, cy, cz, radius)

    Returns
    -------
    ndarray
        Absolute distance from nearest sphere surface for each grid point.
        0 = on surface, positive = near surface (inside or outside).
    """
    min_abs_dist = np.full(x_grid.shape, np.inf)

    for cx, cy, cz, radius in spheres:
        dist_from_center = np.sqrt(
            (x_grid - cx) ** 2 + (y_grid - cy) ** 2 + (z_grid - cz) ** 2)
        abs_dist = np.abs(dist_from_center - radius)
        min_abs_dist = np.minimum(min_abs_dist, abs_dist)

    return min_abs_dist


def find_edge_artifacts(data: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        spheres: list,
        mask: np.ndarray = None,
        edge_threshold: float = 1.0,
        isolation_ratio: float = 1.3,
        kernel_radius: int = 2,
        verbose: bool = False):
    """Find BEM artifact pixels using a hybrid edge + spatial isolation filter.

    A pixel is identified as an artifact if ALL conditions are met:
      1. Edge zone: within ``edge_threshold`` nm from nearest sphere surface.
      2. Spatially isolated: value > ``isolation_ratio`` * local median (in-mask
         neighbours only).

    Only edge-zone pixels (typically <5 % of the grid) are iterated — fast.

    Parameters
    ----------
    data : ndarray (2-D)
        Field data (enhancement, intensity, etc.).
    x_grid, y_grid, z_grid : ndarray
        Coordinate grids (same shape as *data*).
    spheres : list of (cx, cy, cz, radius)
        Sphere boundaries.
    mask : ndarray (bool), optional
        Region mask (True = valid pixels to consider).  If None, uses all
        non-NaN finite pixels.
    edge_threshold : float
        Distance from surface in nm (default 1.0).
    isolation_ratio : float
        Threshold for spatial isolation (default 1.3).
    kernel_radius : int
        Half-size of neighbourhood window (default 2 → 5x5 window).
    verbose : bool

    Returns
    -------
    artifact_mask : ndarray (bool)
        True for pixels identified as artifacts.
    n_removed : int
        Number of artifacts found.
    """
    if data is None or spheres is None or len(spheres) == 0:
        return (np.zeros_like(data, dtype=bool) if data is not None else None), 0

    if mask is None:
        mask = ~np.isnan(data) & np.isfinite(data)

    # 1. Edge zone
    min_dist = compute_min_surface_distance(x_grid, y_grid, z_grid, spheres)
    is_edge = (min_dist <= edge_threshold) & mask
    edge_indices = np.argwhere(is_edge)

    if verbose:
        print('      Edge filter: {} edge-zone pixels (within {:.1f} nm of surface)'.format(
            len(edge_indices), edge_threshold))

    if len(edge_indices) == 0:
        return np.zeros_like(data, dtype=bool), 0

    # 2. Spatial isolation check — only for edge-zone pixels
    data_masked = np.full_like(data, np.nan, dtype=float)
    data_masked[mask] = data[mask]

    artifact_mask = np.zeros_like(data, dtype=bool)
    rows, cols = data.shape

    for idx in edge_indices:
        r, c = int(idx[0]), int(idx[1])
        val = data_masked[r, c]

        if not np.isfinite(val) or val <= 0:
            continue

        r_min = max(0, r - kernel_radius)
        r_max = min(rows, r + kernel_radius + 1)
        c_min = max(0, c - kernel_radius)
        c_max = min(cols, c + kernel_radius + 1)

        neighbourhood = data_masked[r_min:r_max, c_min:c_max]
        valid_neighbours = neighbourhood[~np.isnan(neighbourhood)]

        if len(valid_neighbours) < 3:
            continue

        local_median = float(np.median(valid_neighbours))

        if local_median > 0 and val > isolation_ratio * local_median:
            artifact_mask[r, c] = True

    n_removed = int(np.sum(artifact_mask))

    if verbose:
        print('      Edge filter: {} artifacts removed '
              '(isolated spikes within {:.1f} nm edge zone)'.format(n_removed, edge_threshold))

    return artifact_mask, n_removed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_radius(config: dict) -> float:
    """Return sphere radius from config, handling 'radius' and 'diameter' keys."""
    if 'diameter' in config:
        return float(config['diameter']) / 2
    elif 'radius' in config:
        return float(config['radius'])
    return 25.0


def _cluster_positions(n: int, spacing: float) -> list:
    """Calculate hexagonal cluster positions (matches MATLAB geometry).

    Parameters
    ----------
    n : int
        Number of spheres (1-7).
    spacing : float
        Centre-to-centre spacing (diameter + gap).

    Returns
    -------
    list of (x, y, z) tuples
    """
    dy60 = spacing * 0.866025404  # sin(60°)

    hex_pos = []
    for i in range(6):
        angle = i * 60 * np.pi / 180
        hex_pos.append((spacing * np.cos(angle), spacing * np.sin(angle), 0.0))

    layouts = {
        1: [(0.0, 0.0, 0.0)],
        2: [(-spacing / 2, 0.0, 0.0), (spacing / 2, 0.0, 0.0)],
        3: [(-spacing / 2, 0.0, 0.0), (spacing / 2, 0.0, 0.0), (0.0, dy60, 0.0)],
        4: [(0.0, 0.0, 0.0)] + hex_pos[:3],
        5: [(0.0, 0.0, 0.0)] + hex_pos[:4],
        6: [(0.0, 0.0, 0.0)] + hex_pos[:5],
        7: [(0.0, 0.0, 0.0)] + hex_pos[:6],
    }

    if n not in layouts:
        raise ValueError('n_spheres must be 1-7, got {}'.format(n))

    return layouts[n]
