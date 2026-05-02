"""Regression: 6 excitation types (M5).

planewave/dipole/eels × ret/stat = 6 modes.

Markers:
  fast: skipped (CLI-based smokes are slow, needs build)
  slow: each 2-wavelength smoke (~30s each)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest


def _run_yaml(repo_root: Path,
        yaml_name: str,
        sim_name: str,
        n_wl: int = 2) -> Tuple[Dict[str, Any], int, float]:

    yaml_path = repo_root / 'examples' / yaml_name
    if not yaml_path.exists():
        pytest.skip('{} missing'.format(yaml_name))

    out_dir = repo_root / 'results' / sim_name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
            sys.executable,
            str(repo_root / 'run_simulation.py'),
            '--config', str(yaml_path),
            '--simulation-name', sim_name,
            '--n-wavelengths', str(n_wl),
            '--n-workers', '1',
            '--n-threads', '2',
            '--n-gpus-per-worker', '0']

    print('[reg] cmd: {}'.format(' '.join(cmd)))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd = repo_root, capture_output = True, text = True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print('[reg] stderr (last 20):')
        for line in proc.stderr.splitlines()[-20:]:
            print('  {}'.format(line))

    summary = {}
    summary_path = out_dir / 'spectrum.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)

    return summary, proc.returncode, elapsed


@pytest.mark.slow
def test_planewave_ret(repo_root):
    summary, rc, _ = _run_yaml(repo_root, 'sphere_planewave_stat.yaml',
            'reg_pw_stat', n_wl = 2)
    assert rc == 0
    assert summary.get('n_wavelengths') == 2
    assert summary.get('peak_ext_x', 0.0) > 0


@pytest.mark.slow
def test_dipole_ret(repo_root, reference_results):
    """Wave 2 smoke 3: sphere dipole peak decay = 25.221."""
    from .runners.compute_grade import compute_grade

    summary, rc, _ = _run_yaml(repo_root, 'sphere_dipole.yaml',
            'reg_dipole_ret', n_wl = 2)
    assert rc == 0

    ref = reference_results['sphere_dipole_2wl']
    measured = summary.get('peak_ext_x', 0.0)
    expected = ref['peak_decay_total']
    grade = compute_grade(measured, expected)

    print('[reg] dipole_ret peak measured={:.3f} ref={:.3f} grade={}'.format(
            measured, expected, grade))

    assert grade != 'BAD'


@pytest.mark.slow
def test_dipole_stat(repo_root):
    summary, rc, _ = _run_yaml(repo_root, 'sphere_dipole_stat.yaml',
            'reg_dipole_stat', n_wl = 2)
    assert rc == 0
    assert summary.get('peak_ext_x', 0.0) > 0


@pytest.mark.slow
def test_eels_ret(repo_root):
    summary, rc, _ = _run_yaml(repo_root, 'sphere_eels.yaml',
            'reg_eels_ret', n_wl = 2)
    assert rc == 0
    assert summary.get('peak_ext_x', 0.0) > 0


@pytest.mark.slow
def test_eels_stat(repo_root):
    summary, rc, _ = _run_yaml(repo_root, 'sphere_eels_stat.yaml',
            'reg_eels_stat', n_wl = 2)
    assert rc == 0
    assert summary.get('peak_ext_x', 0.0) > 0
