import os
import sys
import time

from pathlib import Path
from typing import Any, Dict

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_small_dimer():
    from mnpbem.geometry import tricube, ComParticle
    from mnpbem.materials import EpsConst

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

    return p, epstab


def _make_cfg() -> Dict[str, Any]:
    return {
            'structure': {
                'type': 'dimer_cube',
                'edge': 30.0,
                'gap': 1.0,
                'n_per_edge': 8,
                'refine': 2},
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
            'materials': {
                'medium': 'water',
                'particle': 'gold'}}


def test_grid_builder_rectangular():
    from pymnpbem_simulation.simulation import grid_builder

    xx, yy, zz, pts = grid_builder.make_rectangular_grid(
            [-10, 10], [0, 0], [-10, 10], [3, 1, 3])

    assert pts.shape == (9, 3), '[error] rectangular grid pts shape <{}>'.format(pts.shape)
    assert xx.shape == (3, 1, 3)
    print('[test] grid_builder rectangular: OK pts={}'.format(pts.shape))


def test_grid_builder_spherical():
    from pymnpbem_simulation.simulation import grid_builder

    xx, yy, zz, pts = grid_builder.make_spherical_grid(
            [10.0, 20.0], [0.0, np.pi], [0.0, 2 * np.pi], [2, 4, 4])

    assert pts.shape == (32, 3), '[error] spherical grid pts shape <{}>'.format(pts.shape)
    r_calc = np.linalg.norm(pts, axis = 1)
    assert r_calc.min() > 9.0 and r_calc.max() < 21.0
    print('[test] grid_builder spherical: OK pts={}'.format(pts.shape))


def test_grid_builder_custom():
    from pymnpbem_simulation.simulation import grid_builder

    user_pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype = np.float64)
    xx, yy, zz, pts = grid_builder.make_custom_points(user_pts)

    assert pts.shape == (3, 3)
    np.testing.assert_array_equal(pts, user_pts)
    print('[test] grid_builder custom: OK')


def test_meshfield_import():
    from mnpbem.simulation import MeshField

    assert MeshField is not None
    print('[test] mnpbem.simulation.MeshField import: OK')


def test_field_calculator_dimer_smoke():
    from pymnpbem_simulation.simulation import FieldCalculator

    print('[test] building dimer particle (small)...')
    t0 = time.time()
    p, epstab = _build_small_dimer()
    print('[test]   built in {:.1f}s, n_face={}'.format(time.time() - t0, p.pos.shape[0]))

    cfg = _make_cfg()

    fc = FieldCalculator(cfg, p, epstab)

    assert fc.grid_points.shape == (25, 3), \
            '[error] grid_points shape <{}>'.format(fc.grid_points.shape)
    assert fc.grid_x.shape == (1, 5, 5)

    print('[test] running 1-wavelength field eval at enei=620 nm...')
    t1 = time.time()

    enei = np.array([620.0])
    result = fc.run(enei)

    elapsed = time.time() - t1
    print('[test]   field run elapsed: {:.1f}s'.format(elapsed))

    assert 'e' in result
    assert result['e'].shape == (1, 25, 3, 1), \
            '[error] e shape <{}>'.format(result['e'].shape)
    assert np.iscomplexobj(result['e'])

    finite_mask = np.isfinite(result['e'].real) & np.isfinite(result['e'].imag)
    finite_frac = float(finite_mask.mean())
    assert finite_frac >= 0.8, \
            '[error] only {:.1%} finite (mindist filter too aggressive)!'.format(finite_frac)
    print('[test] finite fraction = {:.1%}'.format(finite_frac))

    e_mag = float(np.nanmax(np.abs(result['e'])))
    assert e_mag > 0, '[error] zero E field!'
    print('[test] field max |E| = {:.3e}'.format(e_mag))

    return result


def test_field_analyzer_smoke():
    from pymnpbem_simulation.postprocess import (
            hotspot_location,
            field_enhancement,
            near_field_decay,
            integrated_field_intensity)

    n_pts = 50
    pos = np.random.RandomState(0).uniform(-30, 30, (n_pts, 3))
    e = np.random.RandomState(1).randn(n_pts, 3) + 1j * np.random.RandomState(2).randn(n_pts, 3)

    field_result = {'e': e, 'pos': pos}

    hot = hotspot_location(field_result, threshold_quantile = 0.9)
    assert hot.n_hotspots > 0
    assert hot.max_intensity > 0
    print('[test] hotspot_location: n_hot={}  max_I={:.3e}'.format(hot.n_hotspots, hot.max_intensity))

    enh = field_enhancement(field_result, np.array([1.0, 0.0, 0.0]))
    assert enh.shape == (n_pts,)
    print('[test] field_enhancement: shape={}  range=[{:.3e}, {:.3e}]'.format(
            enh.shape, enh.min(), enh.max()))

    surf = pos[:5]
    decay = near_field_decay(field_result, surf)
    assert decay.distances.shape == (n_pts,)
    assert decay.e2.shape == (n_pts,)
    assert np.all(np.diff(decay.distances) >= -1e-12)
    print('[test] near_field_decay: dmin={:.2f}  dmax={:.2f}'.format(
            decay.distances.min(), decay.distances.max()))

    total = integrated_field_intensity(field_result)
    assert total > 0
    print('[test] integrated_field_intensity = {:.3e}'.format(total))


def test_plot_field_smoke():
    from pymnpbem_simulation.postprocess import plot_field_2d_slice, plot_near_field_decay

    out_dir = REPO_ROOT / 'results' / 'test_field_smoke'
    out_dir.mkdir(parents = True, exist_ok = True)

    n_pts = 100
    rng = np.random.RandomState(42)
    pos = np.zeros((n_pts, 3), dtype = np.float64)
    pos[:, 0] = rng.uniform(-30, 30, n_pts)
    pos[:, 2] = rng.uniform(-30, 30, n_pts)
    e = rng.randn(n_pts, 3) + 1j * rng.randn(n_pts, 3)

    field_result = {'e': e, 'pos': pos}

    save_path = str(out_dir / 'field_slice.png')
    plot_field_2d_slice(field_result, axis = 'y', value = 0.0, log_scale = True, save = save_path)
    assert os.path.exists(save_path)
    print('[test] plot_field_2d_slice: saved <{}>'.format(save_path))

    decay_save = str(out_dir / 'decay.png')
    decay_result = {
            'distances': np.linspace(1, 50, 50),
            'e2': np.exp(-np.linspace(0, 5, 50))}
    plot_near_field_decay(decay_result, save = decay_save)
    assert os.path.exists(decay_save)
    print('[test] plot_near_field_decay: saved <{}>'.format(decay_save))


if __name__ == '__main__':
    print('[test] === test_field_dimer ===')
    test_grid_builder_rectangular()
    test_grid_builder_spherical()
    test_grid_builder_custom()
    test_meshfield_import()
    test_field_analyzer_smoke()
    test_plot_field_smoke()
    test_field_calculator_dimer_smoke()
    print('[test] === all field tests passed ===')
