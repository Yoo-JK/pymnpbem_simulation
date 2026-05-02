"""Regression: field calculation (M3).

grid_builder + FieldCalculator + field_analyzer.

Markers:
  fast: grid_builder + analyzer (no BEM solve)
  slow: FieldCalculator small dimer single-wavelength
"""
from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.fast
def test_grid_builder_rectangular():
    from pymnpbem_simulation.simulation import grid_builder

    xx, yy, zz, pts = grid_builder.make_rectangular_grid(
            [-10, 10], [0, 0], [-10, 10], [3, 1, 3])

    assert pts.shape == (9, 3)
    assert xx.shape == (3, 1, 3)


@pytest.mark.fast
def test_grid_builder_spherical():
    from pymnpbem_simulation.simulation import grid_builder

    xx, yy, zz, pts = grid_builder.make_spherical_grid(
            [10.0, 20.0], [0.0, np.pi], [0.0, 2 * np.pi], [2, 4, 4])

    assert pts.shape == (32, 3)
    r_calc = np.linalg.norm(pts, axis = 1)
    assert r_calc.min() > 9.0 and r_calc.max() < 21.0


@pytest.mark.fast
def test_grid_builder_custom():
    from pymnpbem_simulation.simulation import grid_builder

    user_pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype = np.float64)
    _, _, _, pts = grid_builder.make_custom_points(user_pts)
    assert pts.shape == (3, 3)
    np.testing.assert_array_equal(pts, user_pts)


@pytest.mark.fast
def test_meshfield_import():
    from mnpbem.simulation import MeshField
    assert MeshField is not None


@pytest.mark.fast
def test_field_analyzer_smoke():
    from pymnpbem_simulation.postprocess import (
            hotspot_location,
            field_enhancement,
            near_field_decay,
            integrated_field_intensity)

    n_pts = 50
    pos = np.random.RandomState(0).uniform(-30, 30, (n_pts, 3))
    e = (np.random.RandomState(1).randn(n_pts, 3)
            + 1j * np.random.RandomState(2).randn(n_pts, 3))

    field_result = {'e': e, 'pos': pos}

    hot = hotspot_location(field_result, threshold_quantile = 0.9)
    assert hot.n_hotspots > 0
    assert hot.max_intensity > 0

    enh = field_enhancement(field_result, np.array([1.0, 0.0, 0.0]))
    assert enh.shape == (n_pts,)

    surf = pos[:5]
    decay = near_field_decay(field_result, surf)
    assert decay.distances.shape == (n_pts,)
    assert decay.e2.shape == (n_pts,)

    total = integrated_field_intensity(field_result)
    assert total > 0


@pytest.mark.slow
def test_field_calculator_dimer_smoke(repo_root):
    """Small dimer field eval at single wavelength."""
    from mnpbem.geometry import tricube, ComParticle
    from mnpbem.materials import EpsConst
    from pymnpbem_simulation.simulation import FieldCalculator

    eps_medium = EpsConst(1.33 ** 2)
    eps_particle = EpsConst(-10 + 1j)
    epstab = [eps_medium, eps_particle]

    edge = 30.0
    gap = 1.0
    n_per_edge = 8
    refine = 2
    half = edge / 2 + gap / 2

    cube1 = tricube(n_per_edge, edge, e = 0.2, refine = refine)
    cube1.shift([-half, 0, 0])
    cube2 = tricube(n_per_edge, edge, e = 0.2, refine = refine)
    cube2.shift([+half, 0, 0])

    p = ComParticle(epstab, [cube1, cube2], [[2, 1], [2, 1]],
            interp = 'curv', refine = refine)

    cfg = {
            'structure': {'type': 'dimer_cube', 'edge': 30.0, 'gap': 1.0,
                    'n_per_edge': 8, 'refine': 2},
            'simulation': {
                    'type': 'ret',
                    'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]],
                    'propagation_dirs': [[0, 0, 1]],
                    'grid': {
                            'type': 'rectangular',
                            'x_range': [0, 0],
                            'y_range': [-40, 40],
                            'z_range': [-40, 40],
                            'n_points': [1, 5, 5]},
                    'mindist': 2.0,
                    'nmax': None,
                    'inout': 2,
                    'fmm': False},
            'materials': {'medium': 'water', 'particle': 'gold'}}

    fc = FieldCalculator(cfg, p, epstab)

    assert fc.grid_points.shape == (25, 3)

    enei = np.array([620.0])
    result = fc.run(enei)

    assert 'e' in result
    assert result['e'].shape == (1, 25, 3, 1)
    assert np.iscomplexobj(result['e'])

    finite = np.isfinite(result['e'].real) & np.isfinite(result['e'].imag)
    assert finite.mean() >= 0.8

    e_mag = float(np.nanmax(np.abs(result['e'])))
    assert e_mag > 0
