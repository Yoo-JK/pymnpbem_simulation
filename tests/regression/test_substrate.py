"""Regression: substrate / layer simulation (M6).

Markers:
  fast: substrate REGISTRY check
  slow: sphere on glass substrate, 2 wl smoke
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.fast
def test_substrate_registry():
    """ret_layer simulation entries must register."""
    from pymnpbem_simulation.simulation import REGISTRY

    layer_keys = [k for k in REGISTRY.keys() if 'ret_layer' in str(k)]
    assert len(layer_keys) >= 1, 'no ret_layer entries in REGISTRY'


@pytest.mark.fast
def test_with_substrate_builder():
    """with_substrate composite structure builder must be registered."""
    from pymnpbem_simulation.structures import REGISTRY
    assert 'with_substrate' in REGISTRY


@pytest.mark.slow
def test_sphere_substrate_smoke(repo_root, reference_results):
    """Wave 2 smoke 4: sphere on glass, 2 wl, peak ext_x = 76.543 (gap=0.001 touching)."""
    from .runners.compute_grade import compute_grade

    yaml_path = repo_root / 'examples' / 'sphere_substrate.yaml'
    if not yaml_path.exists():
        pytest.skip('sphere_substrate.yaml missing')

    name = 'reg_sphere_substrate_2wl'
    out_dir = repo_root / 'results' / name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
            sys.executable,
            str(repo_root / 'run_simulation.py'),
            '--config', str(yaml_path),
            '--simulation-name', name,
            '--n-wavelengths', '2',
            '--n-workers', '1',
            '--n-threads', '4',
            '--n-gpus-per-worker', '0']

    print('[reg] cmd: {}'.format(' '.join(cmd)))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd = repo_root, capture_output = True, text = True)
    elapsed = time.time() - t0
    print('[reg] elapsed: {:.1f}s'.format(elapsed))

    if proc.returncode != 0:
        for line in proc.stderr.splitlines()[-20:]:
            print('  {}'.format(line))
        raise AssertionError('CLI rc={}'.format(proc.returncode))

    summary_path = out_dir / 'spectrum.json'
    if not summary_path.exists():
        # fall back to npz if spectrum.json missing
        npz_path = out_dir / 'spectrum.npz'
        assert npz_path.exists(), 'no spectrum output'
        data = np.load(npz_path)
        ext = data['ext']
        peak = float(np.max(ext))
    else:
        with open(summary_path) as f:
            summary = json.load(f)
        peak = summary.get('peak_ext_x', 0.0)

    ref = reference_results['sphere_substrate_2wl']
    expected = ref['peak_ext_x']
    grade = compute_grade(peak, expected)

    print('[reg] substrate peak_ext_x measured={:.3f} ref={:.3f} grade={}'.format(
            peak, expected, grade))

    assert grade != 'BAD', 'BAD grade'
