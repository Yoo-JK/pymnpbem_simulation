import os
import sys
import tempfile
import shutil

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))


def _build_dimer_mesh() -> dict:
    """Build a tiny synthetic dimer mesh for surface charge plot smoke tests."""
    rng = np.random.default_rng(42)

    # Two spheres on x-axis: centers at -10 and +10, radius 5
    nfaces_per = 32
    verts_list = []
    faces_list = []
    centroids = []
    normals = []
    areas = []

    vert_offset = 0

    for cx in [-10.0, 10.0]:
        # Random points on sphere surface
        phi = rng.uniform(0, np.pi, nfaces_per)
        theta = rng.uniform(0, 2 * np.pi, nfaces_per)
        r = 5.0
        cx_arr = np.full(nfaces_per, cx)

        x = cx_arr + r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        face_centroids = np.stack([x, y, z], axis = 1)
        face_normals = face_centroids - np.array([cx, 0.0, 0.0])
        face_normals /= np.linalg.norm(face_normals, axis = 1, keepdims = True)

        # Triangle vertices: build a tiny tri around each centroid
        tri_size = 0.5
        tri_dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        for i in range(nfaces_per):
            v0 = face_centroids[i] + tri_size * tri_dirs[0]
            v1 = face_centroids[i] + tri_size * tri_dirs[1]
            v2 = face_centroids[i] + tri_size * tri_dirs[2]
            verts_list.extend([v0, v1, v2])
            faces_list.append([vert_offset, vert_offset + 1, vert_offset + 2,
                    np.nan])
            vert_offset += 3

        centroids.append(face_centroids)
        normals.append(face_normals)
        areas.append(np.full(nfaces_per, tri_size ** 2 * 0.5))

    return {
            'verts': np.asarray(verts_list, dtype = float),
            'faces': np.asarray(faces_list, dtype = float),
            'centroids': np.concatenate(centroids, axis = 0),
            'normals': np.concatenate(normals, axis = 0),
            'areas': np.concatenate(areas, axis = 0),
            'polarizations': np.array([[1, 0, 0], [0, 1, 0]], dtype = float)}


def test_plot_surface_charge_3d_smoke():
    from pymnpbem_simulation.postprocess import plot_surface_charge_3d

    mesh = _build_dimer_mesh()
    nfaces = mesh['centroids'].shape[0]
    rng = np.random.default_rng(7)
    sigma = (rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)) * 1e-3

    with tempfile.TemporaryDirectory() as tmpdir:
        files = plot_surface_charge_3d(tmpdir, sigma, mesh['verts'],
                mesh['faces'], 600.0, 0, '[1 0 0]')
        assert len(files) >= 1, '[error] expected at least one file'
        for f in files:
            assert os.path.exists(f), '[error] missing <{}>'.format(f)
            assert os.path.getsize(f) > 1000, '[error] empty figure <{}>'.format(f)
    print('[test] plot_surface_charge_3d smoke OK')


def test_plot_surface_charge_2d_8views_smoke():
    from pymnpbem_simulation.postprocess import plot_surface_charge_2d_8views

    mesh = _build_dimer_mesh()
    nfaces = mesh['centroids'].shape[0]
    rng = np.random.default_rng(11)
    sigma = (rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)) * 1e-3

    with tempfile.TemporaryDirectory() as tmpdir:
        files = plot_surface_charge_2d_8views(tmpdir, sigma,
                mesh['centroids'], mesh['normals'],
                600.0, 0, '[1 0 0]')
        assert len(files) >= 1, '[error] expected at least one file'
        for f in files:
            assert os.path.exists(f), '[error] missing <{}>'.format(f)
            assert os.path.getsize(f) > 1000, '[error] empty figure <{}>'.format(f)
    print('[test] plot_surface_charge_2d_8views smoke OK')


def test_plot_surface_charge_phase_smoke():
    from pymnpbem_simulation.postprocess import plot_surface_charge_phase

    mesh = _build_dimer_mesh()
    nfaces = mesh['centroids'].shape[0]
    rng = np.random.default_rng(13)
    sigma = (rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)) * 1e-3

    with tempfile.TemporaryDirectory() as tmpdir:
        files = plot_surface_charge_phase(tmpdir, sigma,
                mesh['verts'], mesh['faces'],
                mesh['centroids'], mesh['normals'],
                600.0, 0, '[1 0 0]')
        # 4 components produced
        assert len(files) >= 4, '[error] expected at least 4 phase plots, got {}'.format(
                len(files))
        for f in files:
            assert os.path.exists(f), '[error] missing <{}>'.format(f)
            assert os.path.getsize(f) > 1000, '[error] empty figure <{}>'.format(f)
    print('[test] plot_surface_charge_phase smoke OK')


def test_plot_all_surface_charge_smoke():
    from pymnpbem_simulation.postprocess import plot_all_surface_charge

    mesh = _build_dimer_mesh()
    nfaces = mesh['centroids'].shape[0]
    rng = np.random.default_rng(17)

    n_wl = 2
    n_pol = 2

    sig2 = (rng.standard_normal((n_wl, nfaces, n_pol))
            + 1j * rng.standard_normal((n_wl, nfaces, n_pol))) * 1e-3

    sc = dict(mesh)
    sc['wavelengths'] = np.array([600.0, 700.0])
    sc['wl_indices'] = np.array([5, 10])
    sc['sig2'] = sig2
    sc['sig1'] = sig2.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        files = plot_all_surface_charge(tmpdir, sc, plot_format = ['png'])
        # n_wl x n_pol x (1 3D + 1 8-view + 4 phase) = 2 x 2 x 6 = 24
        assert len(files) == 24, '[error] expected 24 files, got {}'.format(
                len(files))
        for f in files:
            assert os.path.exists(f), '[error] missing <{}>'.format(f)
            assert os.path.getsize(f) > 1000, '[error] empty figure <{}>'.format(f)
    print('[test] plot_all_surface_charge smoke OK ({} files)'.format(len(files)))


def test_planewave_ret_sigma_storage_smoke():
    """End-to-end mini test: run a tiny PlaneWaveRet with field_wavelength_idx."""
    from mnpbem.materials import EpsConst, EpsTable
    from mnpbem.geometry import trisphere, ComParticle
    from pymnpbem_simulation.simulation.planewave_ret import PlaneWaveRetRunner

    eps_medium = EpsConst(1.0)
    eps_au = EpsTable('gold.dat')
    epstab = [eps_medium, eps_au]

    sphere = trisphere(60, 20.0)  # small mesh
    p = ComParticle(epstab, [sphere], [[2, 1]],
            interp = 'curv', refine = 0)

    cfg = {
            'simulation': {
                    'polarizations': [[1, 0, 0], [0, 1, 0]],
                    'propagation_dirs': [[0, 0, 1], [0, 0, 1]],
                    'field_wavelength_idx': [550.0],
                    'calculate_surface_charge': True,
                    'surface_charge_wavelength_tol': 30.0}}

    runner = PlaneWaveRetRunner(cfg, p, epstab)
    enei = np.linspace(500.0, 600.0, 3)

    result = runner.run(enei)

    assert 'surface_charge' in result, '[error] surface_charge missing'
    sc = result['surface_charge']
    assert sc['sig2'].shape[0] >= 1, '[error] no sigma stored'
    assert sc['sig2'].shape[1] == p.nfaces, '[error] nfaces mismatch'
    assert sc['sig2'].shape[2] == 2, '[error] n_pol mismatch'
    assert sc['verts'].ndim == 2, '[error] verts not 2D'
    assert sc['centroids'].shape[0] == p.nfaces, '[error] centroids mismatch'

    print('[test] planewave_ret sigma storage smoke OK: {} wls, sig2={}'.format(
            sc['sig2'].shape[0], sc['sig2'].shape))


if __name__ == '__main__':
    test_plot_surface_charge_3d_smoke()
    test_plot_surface_charge_2d_8views_smoke()
    test_plot_surface_charge_phase_smoke()
    test_plot_all_surface_charge_smoke()
    test_planewave_ret_sigma_storage_smoke()
    print('\n[OK] All surface_charge smoke tests passed.')
