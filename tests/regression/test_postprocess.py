"""Regression: postprocess (eigenmode + Fano + multipole + export) — M8.

Markers:
  fast: all (no BEM solve required, test sphere small)
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest


def _build_test_sphere(diameter: float = 50.0,
        mesh_density: int = 144,
        refine: int = 2):
    from mnpbem.materials import EpsConst, EpsTable
    from mnpbem.geometry import trisphere, ComParticle

    eps_medium = EpsConst(1.0)
    eps_au = EpsTable('gold.dat')
    epstab = [eps_medium, eps_au]

    sphere = trisphere(mesh_density, diameter)
    p = ComParticle(epstab, [sphere], [[2, 1]], interp = 'curv', refine = refine)
    return p, epstab


@pytest.mark.fast
def test_eigenmode_smoke():
    from pymnpbem_simulation.postprocess import qs_eigenmodes

    p, _ = _build_test_sphere(mesh_density = 144)
    result = qs_eigenmodes(p, n_modes = 5)

    assert result['n_modes'] == 5
    assert result['eigenvalues'].shape == (5,)
    assert result['eigenvectors_r'].shape[1] == 5
    assert result['eigenvectors_l'].shape[0] == 5


@pytest.mark.fast
def test_svd_decomposition():
    from pymnpbem_simulation.postprocess import svd_decomposition

    rng = np.random.default_rng(42)
    sig_matrix = rng.standard_normal((100, 30))
    result = svd_decomposition(sig_matrix)

    assert result['u'].shape[0] == 100
    assert result['vh'].shape[1] == 30
    assert len(result['singular_values']) == 30
    assert result['rank_eff'] > 0


@pytest.mark.fast
def test_fano_fit_synthetic():
    from pymnpbem_simulation.postprocess import fano_fit, fano_lineshape

    enei = np.linspace(500, 700, 200)
    true_amp, true_x0, true_gamma, true_q, true_c = 2.0, 600.0, 30.0, 1.5, 0.1

    spectrum_clean = fano_lineshape(enei, true_amp, true_x0, true_gamma,
            true_q, true_c)
    rng = np.random.default_rng(123)
    spectrum = (spectrum_clean
            + 0.02 * np.std(spectrum_clean) * rng.standard_normal(len(enei)))

    result = fano_fit(enei, spectrum)
    assert result['success']

    bf = result['best_fit']
    assert abs(bf['x0'] - true_x0) / abs(true_x0) < 0.05
    assert abs(bf['gamma'] - true_gamma) / abs(true_gamma) < 0.10
    assert abs(abs(bf['q']) - abs(true_q)) / abs(true_q) < 0.20


@pytest.mark.fast
def test_multipole_sphere():
    from pymnpbem_simulation.postprocess import (
            multipole_decomposition, dominant_l)

    p, _ = _build_test_sphere(diameter = 50.0, mesh_density = 144)

    pos = np.asarray(p.pos)
    rel_z = pos[:, 2] - pos[:, 2].mean()
    sig = rel_z / max(np.abs(rel_z).max(), 1e-12)

    result = multipole_decomposition(sig, p, max_l = 4)
    power_l = np.asarray(result['power_l'])

    assert dominant_l(result) == 1
    assert power_l[1] > 10 * power_l[2]


@pytest.mark.fast
def test_export_roundtrip():
    from pymnpbem_simulation.postprocess import (
            export_npz, export_h5, export_csv, export_json)

    rng = np.random.default_rng(7)
    data = {
            'wavelength': np.linspace(400, 800, 50),
            'ext': rng.standard_normal((50, 2)),
            'meta_n_pol': 2,
            'meta_label': 'test'}

    with tempfile.TemporaryDirectory() as tmp:
        npz_path = os.path.join(tmp, 'out.npz')
        h5_path = os.path.join(tmp, 'out.h5')
        csv_path = os.path.join(tmp, 'out.csv')
        json_path = os.path.join(tmp, 'out.json')

        export_npz(data, npz_path)
        export_h5(data, h5_path)
        export_csv(data, csv_path)
        export_json(data, json_path)

        npz = np.load(npz_path)
        assert np.allclose(npz['wavelength'], data['wavelength'])
        assert np.allclose(npz['ext'], data['ext'])

        import h5py
        with h5py.File(h5_path, 'r') as f:
            assert np.allclose(f['wavelength'][:], data['wavelength'])
            assert np.allclose(f['ext'][:], data['ext'])

        with open(json_path) as f:
            j = json.load(f)
        assert np.allclose(np.asarray(j['wavelength']), data['wavelength'])
        assert j['meta_label'] == 'test'
