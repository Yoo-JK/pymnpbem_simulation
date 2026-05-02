import os
import sys
import json
import tempfile
import subprocess

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))


def _build_test_sphere(diameter: float = 50.0,
        mesh_density: int = 144,
        refine: int = 2):

    from mnpbem.materials import EpsConst, EpsTable
    from mnpbem.geometry import trisphere, ComParticle

    eps_medium = EpsConst(1.0)
    eps_au = EpsTable('gold.dat')
    epstab = [eps_medium, eps_au]

    sphere = trisphere(mesh_density, diameter)
    p = ComParticle(epstab, [sphere], [[2, 1]],
            interp = 'curv', refine = refine)
    return p, epstab


def test_eigenmode_smoke():
    from pymnpbem_simulation.postprocess import qs_eigenmodes

    p, _ = _build_test_sphere(mesh_density = 144)

    result = qs_eigenmodes(p, n_modes = 5)

    assert result['n_modes'] == 5, '[error] n_modes mismatch'
    assert result['eigenvalues'].shape == (5,), '[error] eigenvalues shape'
    assert result['eigenvectors_r'].shape[1] == 5, '[error] eigvec_r shape'
    assert result['eigenvectors_l'].shape[0] == 5, '[error] eigvec_l shape'

    print('[test] eigenmode smoke: ene[0..4] = {}'.format(
            result['eigenvalues'].real.tolist()))
    print('[test] eigenmode smoke: inv_eps_p[0..4] = {}'.format(
            np.real(result['inv_eps_p']).tolist()))

    return {
        'n_modes': int(result['n_modes']),
        'eigenvalues_real': result['eigenvalues'].real.tolist(),
        'inv_eps_p_real': np.real(result['inv_eps_p']).tolist()}


def test_svd_decomposition():
    from pymnpbem_simulation.postprocess import svd_decomposition

    rng = np.random.default_rng(42)
    n_pts, n_wl = 100, 30
    sig_matrix = rng.standard_normal((n_pts, n_wl))

    result = svd_decomposition(sig_matrix)

    assert result['u'].shape[0] == n_pts, '[error] U rows'
    assert result['vh'].shape[1] == n_wl, '[error] Vh cols'
    assert len(result['singular_values']) == min(n_pts, n_wl), '[error] s len'
    assert result['rank_eff'] > 0, '[error] rank_eff'

    print('[test] svd: rank_eff = {}, s[0] = {:.3f}'.format(
            result['rank_eff'], result['singular_values'][0]))

    return result


def test_fano_fit_synthetic():
    from pymnpbem_simulation.postprocess import fano_fit, fano_lineshape

    enei = np.linspace(500, 700, 200)

    true_amp = 2.0
    true_x0 = 600.0
    true_gamma = 30.0
    true_q = 1.5
    true_c = 0.1

    spectrum_clean = fano_lineshape(enei, true_amp, true_x0, true_gamma, true_q, true_c)
    rng = np.random.default_rng(123)
    spectrum = spectrum_clean + 0.02 * np.std(spectrum_clean) * rng.standard_normal(len(enei))

    result = fano_fit(enei, spectrum)

    assert result['success'], '[error] fano fit not converged'

    bf = result['best_fit']
    err_x0 = abs(bf['x0'] - true_x0) / abs(true_x0)
    err_q = abs(abs(bf['q']) - abs(true_q)) / abs(true_q)
    err_gamma = abs(bf['gamma'] - true_gamma) / abs(true_gamma)

    print('[test] fano: x0 fit={:.2f} (true={:.2f}, err={:.2e})'.format(
            bf['x0'], true_x0, err_x0))
    print('[test] fano: gamma fit={:.2f} (true={:.2f}, err={:.2e})'.format(
            bf['gamma'], true_gamma, err_gamma))
    print('[test] fano: q fit={:.3f} (true={:.3f}, err={:.2e})'.format(
            bf['q'], true_q, err_q))

    assert err_x0 < 0.05, '[error] x0 recovery error {:.2e} > 5%'.format(err_x0)
    assert err_gamma < 0.10, '[error] gamma recovery error {:.2e} > 10%'.format(err_gamma)
    assert err_q < 0.20, '[error] q recovery error {:.2e} > 20%'.format(err_q)

    return {
        'true': {'amp': true_amp, 'x0': true_x0, 'gamma': true_gamma, 'q': true_q},
        'fit': bf}


def test_multipole_sphere():
    from pymnpbem_simulation.postprocess import (
            multipole_decomposition, dominant_l)

    p, _ = _build_test_sphere(diameter = 50.0, mesh_density = 144)

    pos = np.asarray(p.pos)
    # Z-direction dipole-like surface charge: rho ∝ z
    rel_z = pos[:, 2] - pos[:, 2].mean()
    sig = rel_z / max(np.abs(rel_z).max(), 1e-12)

    result = multipole_decomposition(sig, p, max_l = 4)

    power_l = np.asarray(result['power_l'])

    print('[test] multipole sphere: power_l = {}'.format(
            ['{:.3e}'.format(v) for v in power_l]))
    print('[test] multipole sphere: dominant_l = {}'.format(dominant_l(result)))

    # l=1 should dominate
    assert dominant_l(result) == 1, \
            '[error] expected l=1 dominant, got l={}'.format(dominant_l(result))
    assert power_l[1] > 10 * power_l[2], \
            '[error] l=1 not >> l=2: {:.2e} vs {:.2e}'.format(power_l[1], power_l[2])

    return {
        'power_l': power_l.tolist(),
        'dominant_l': int(dominant_l(result))}


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

        # NPZ round-trip
        npz = np.load(npz_path)
        assert np.allclose(npz['wavelength'], data['wavelength']), '[error] npz wavelength'
        assert np.allclose(npz['ext'], data['ext']), '[error] npz ext'

        # H5 round-trip
        import h5py
        with h5py.File(h5_path, 'r') as f:
            wl_h5 = f['wavelength'][:]
            ext_h5 = f['ext'][:]
            assert np.allclose(wl_h5, data['wavelength']), '[error] h5 wavelength'
            assert np.allclose(ext_h5, data['ext']), '[error] h5 ext'

        # CSV: read back
        csv_data = np.loadtxt(csv_path, delimiter = ',', skiprows = 1)
        assert csv_data.shape[0] == 50, '[error] csv rows'
        assert np.allclose(csv_data[:, 0], data['wavelength']), '[error] csv wl'

        # JSON
        with open(json_path) as f:
            j = json.load(f)
        assert np.allclose(np.asarray(j['wavelength']), data['wavelength']), '[error] json wl'
        assert j['meta_label'] == 'test', '[error] json meta_label'

        print('[test] export round-trip: all 4 formats OK')

    return True


def main():
    print('=' * 60)
    print('test_eigenmode_smoke')
    print('=' * 60)
    eig_out = test_eigenmode_smoke()

    print('=' * 60)
    print('test_svd_decomposition')
    print('=' * 60)
    test_svd_decomposition()

    print('=' * 60)
    print('test_fano_fit_synthetic')
    print('=' * 60)
    fano_out = test_fano_fit_synthetic()

    print('=' * 60)
    print('test_multipole_sphere')
    print('=' * 60)
    mp_out = test_multipole_sphere()

    print('=' * 60)
    print('test_export_roundtrip')
    print('=' * 60)
    test_export_roundtrip()

    print('=' * 60)
    print('ALL TESTS PASSED')
    print('=' * 60)

    return {
        'eigenmode': eig_out,
        'fano': fano_out,
        'multipole': mp_out}


if __name__ == '__main__':
    main()
