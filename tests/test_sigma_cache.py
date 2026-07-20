import sys

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from pymnpbem_simulation import sigma_cache as _sc


def _append_manifest(case_dir: str, wavelength_nm: float) -> str:
    return _sc.update_manifest_append(
            case_dir,
            n_faces = 12,
            solver_type = 'retarded',
            structure_hash = 'struct-hash',
            eps_hash = 'eps-hash',
            polarizations = [[1, 0, 0]],
            propagation_dirs = [[0, 0, 1]],
            wavelength_nm = wavelength_nm)


def test_update_manifest_append_serializes_concurrent_writers(tmp_path):
    case_dir = str(tmp_path / 'case')
    wavelengths = [300.0 + 7.5 * idx for idx in range(12)]

        # Multiple spectrum workers append to one manifest during cache saves.
    with ThreadPoolExecutor(max_workers = 6) as pool:
        list(pool.map(lambda wl: _append_manifest(case_dir, wl), wavelengths))

    manifest = _sc.read_manifest(case_dir)
    assert manifest is not None
    assert manifest['solver_type'] == 'retarded'
    assert manifest['n_faces'] == 12
    assert manifest['excitations'] == [{'pol': [1, 0, 0], 'prop_dir': [0, 0, 1]}]
    assert manifest['wavelengths_nm'] == sorted(round(wl, 4) for wl in wavelengths)


def test_update_manifest_append_recovers_from_corrupt_manifest(tmp_path):
    case_dir = str(tmp_path / 'case')
    sigma_dir = Path(_sc.ensure_sigma_dir(case_dir))

    cached_wl = 550.0
    np.savez(
            sigma_dir / _sc.make_filename(cached_wl, [1, 0, 0], [0, 0, 1]),
            sig = np.array([1.0 + 0.0j], dtype = np.complex128),
            wavelength_nm = cached_wl,
            pol = np.array([1, 0, 0], dtype = np.int8),
            prop_dir = np.array([0, 0, 1], dtype = np.int8),
            solver_type = 'quasistatic')

    manifest_path = Path(_sc.manifest_path(case_dir))
    manifest_path.write_text(
            '{"broken": true}\n{"duplicate": true}\n',
            encoding = 'utf-8')

    _sc.update_manifest_append(
            case_dir,
            n_faces = 7,
            solver_type = 'quasistatic',
            structure_hash = 'struct-hash',
            eps_hash = 'eps-hash',
            polarizations = [[1, 0, 0]],
            propagation_dirs = [[0, 0, 1]],
            wavelength_nm = 600.0)

    manifest = _sc.read_manifest(case_dir)
    assert manifest is not None
    assert manifest['solver_type'] == 'quasistatic'
    assert manifest['n_faces'] == 7
    assert manifest['excitations'] == [{'pol': [1, 0, 0], 'prop_dir': [0, 0, 1]}]
    assert manifest['wavelengths_nm'] == [550.0, 600.0]