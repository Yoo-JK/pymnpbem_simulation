import os
import sys
import json
import time
import shutil
import subprocess

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SUBSTRATE_YAML = REPO_ROOT / 'examples' / 'sphere_substrate.yaml'


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _run_cli(name: str,
        config_path: Path,
        n_wl: int = 2) -> Path:

    out_root = REPO_ROOT / 'results'
    out_dir = out_root / name
    _clear_dir(out_dir)

    cmd = [
        sys.executable,
        str(REPO_ROOT / 'run_simulation.py'),
        '--config', str(config_path),
        '--simulation-name', name,
        '--n-wavelengths', str(n_wl),
        '--n-workers', '1',
        '--n-threads', '4',
        '--n-gpus-per-worker', '0']

    print('[test] running: {}'.format(' '.join(cmd)))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output = True, text = True, cwd = REPO_ROOT)
    elapsed = time.time() - t0

    print('[test] stdout (last 30 lines):')
    for line in proc.stdout.splitlines()[-30:]:
        print('  {}'.format(line))

    if proc.returncode != 0:
        print('[test] stderr:')
        for line in proc.stderr.splitlines()[-30:]:
            print('  {}'.format(line))
        raise AssertionError(
            '[test] simulation failed (rc={}, t={:.1f}s)'.format(
                proc.returncode, elapsed))

    print('[test] simulation OK in {:.1f}s'.format(elapsed))
    return out_dir


def _load_spectrum(out_dir: Path) -> dict:
    npz_path = out_dir / 'spectrum.npz'
    assert npz_path.exists(), '[test] missing <{}>'.format(npz_path)
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def test_sphere_substrate_smoke():
    """End-to-end smoke: 20nm Au sphere on glass, 2 wavelengths."""

    out_dir = _run_cli('sphere_substrate_smoke', SUBSTRATE_YAML, n_wl = 2)

    spec = _load_spectrum(out_dir)

    assert 'wavelength' in spec, '[test] no wavelength array in spectrum'
    assert 'ext' in spec, '[test] no ext array in spectrum'
    assert 'sca' in spec, '[test] no sca array in spectrum'
    assert 'abs' in spec, '[test] no abs array in spectrum'

    assert spec['wavelength'].shape == (2,), \
        '[test] wavelength shape {} != (2,)'.format(spec['wavelength'].shape)

    assert spec['ext'].shape[0] == 2, \
        '[test] ext shape {} (n_wl mismatch)'.format(spec['ext'].shape)

    assert np.all(np.isfinite(spec['ext'])), '[test] ext contains non-finite'
    assert np.all(np.isfinite(spec['sca'])), '[test] sca contains non-finite'

    # extinction must be strictly positive for a metal sphere in this band
    assert np.all(spec['ext'] > 0), \
        '[test] ext not strictly positive: {}'.format(spec['ext'])

    print('[test] PASS: sphere_substrate smoke (ext shape {})'.format(
        spec['ext'].shape))


def test_substrate_changes_result():
    """Substrate가 ext 결과를 free-space 와 다르게 만드는지 확인.

    free-space sphere 와 sphere-on-substrate 비교. 같은 wavelength 격자.
    동일하면 substrate 가 BEM 매트릭스에 반영되지 않은 것이므로 fail.
    """

    free_space_yaml = REPO_ROOT / 'examples' / 'sphere_freespace_for_compare.yaml'

    free_space_cfg = '''structure:
  type: sphere
  diameter: 20.0
  mesh_density: 144
  refine: 2
  interp: curv

simulation:
  type: ret
  excitation: planewave
  enei_min: 450
  enei_max: 750
  n_wavelengths: 2
  polarizations:
    - [1, 0, 0]
  propagation_dirs:
    - [0, 0, -1]

materials:
  medium: vacuum
  particle: gold

compute:
  n_workers: 1
  n_threads: 4
  n_gpus_per_worker: 0
  multi_node: false

output:
  dir: ./results
  name: sphere_freespace_compare
  formats:
    - npz
  save_plots: false
'''

    free_space_yaml.write_text(free_space_cfg)

    try:
        out_free = _run_cli('sphere_freespace_compare', free_space_yaml, n_wl = 2)
        out_sub = _run_cli('sphere_substrate_compare', SUBSTRATE_YAML, n_wl = 2)

        spec_free = _load_spectrum(out_free)
        spec_sub = _load_spectrum(out_sub)

        # Compare ext at the same wavelengths (both [450, 750])
        ext_free = spec_free['ext'][:, 0]
        ext_sub = spec_sub['ext'][:, 0]

        rel_diff = np.max(np.abs(ext_free - ext_sub) / np.maximum(np.abs(ext_free), 1e-30))

        print('[test] free ext: {}'.format(ext_free))
        print('[test] sub  ext: {}'.format(ext_sub))
        print('[test] max relative diff: {:.3e}'.format(rel_diff))

        # substrate 효과는 보통 5% 이상의 ext 변화. 1% 이상이면 substrate 가 작동했다고 본다.
        assert rel_diff > 1e-2, \
            '[test] substrate had no measurable effect (rel diff {:.3e} <= 1%)'.format(
                rel_diff)

        print('[test] PASS: substrate measurably alters ext spectrum')

    finally:
        if free_space_yaml.exists():
            free_space_yaml.unlink()


if __name__ == '__main__':
    test_sphere_substrate_smoke()
    test_substrate_changes_result()
