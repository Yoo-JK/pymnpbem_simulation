"""Regression: dimer baseline (Wave 1 회귀, 6336 face × 2/10 wavelength).

Markers:
  fast: skip (dimer too heavy)
  slow: 2 wavelength smoke (~5 min on 4-thread CPU)
  long: 10 wavelength full (~25 min)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

from pathlib import Path

import pytest

from .runners.compute_grade import compute_grade


def _run_baseline(repo_root: Path,
        sim_name: str,
        n_wl: int,
        n_workers: int = 1,
        n_threads: int = 4) -> dict:

    out_dir = repo_root / 'results' / sim_name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
            sys.executable,
            str(repo_root / 'run_simulation.py'),
            '--config', str(repo_root / 'examples' / 'dimer_baseline.yaml'),
            '--simulation-name', sim_name,
            '--n-wavelengths', str(n_wl),
            '--n-workers', str(n_workers),
            '--n-threads', str(n_threads),
            '--n-gpus-per-worker', '0']

    print('[reg] cmd: {}'.format(' '.join(cmd)))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd = repo_root, capture_output = True, text = True)
    elapsed = time.time() - t0
    print('[reg] elapsed: {:.1f}s'.format(elapsed))

    if proc.returncode != 0:
        print('[reg] stderr (last 30 lines):')
        for line in proc.stderr.splitlines()[-30:]:
            print('  {}'.format(line))
        raise AssertionError('CLI returned non-zero rc={}'.format(proc.returncode))

    summary_path = out_dir / 'spectrum.json'
    assert summary_path.exists(), 'missing spectrum.json'

    with open(summary_path, 'r') as f:
        return json.load(f)


@pytest.mark.slow
def test_dimer_baseline_2wl(repo_root, reference_results):
    """Wave 2 smoke 1: dimer 6336 face × 2 wl, expected peak ext_x = 8744.331."""
    summary = _run_baseline(repo_root, 'reg_dimer_baseline_2wl', n_wl = 2)

    ref = reference_results['dimer_baseline_2wl']

    measured = summary['peak_ext_x']
    expected = ref['peak_ext_x']
    grade = compute_grade(measured, expected)

    print('[reg] dimer 2wl peak_ext_x measured={:.3f} ref={:.3f} grade={}'.format(
            measured, expected, grade))

    assert grade != 'BAD', 'BAD grade: rel diff > 1e-3'
    assert summary['n_wavelengths'] == 2


@pytest.mark.long
def test_dimer_baseline_10wl(repo_root, reference_results):
    """Long: 10 wavelength run, validates spectrum coverage."""
    summary = _run_baseline(repo_root, 'reg_dimer_baseline_10wl', n_wl = 10)

    assert summary['n_wavelengths'] == 10
    assert summary['peak_ext_x'] > 0

    print('[reg] dimer 10wl peak_ext_x = {:.3f} at {:.1f} nm'.format(
            summary['peak_ext_x'], summary.get('peak_wl_nm', 0.0)))


@pytest.mark.slow
def test_dimer_baseline_cpu_pool(repo_root, reference_results):
    """Wave 2 smoke 6: dimer × 2 wl × 2 workers (cpu_pool dispatch)."""
    summary = _run_baseline(repo_root, 'reg_dimer_baseline_pool',
            n_wl = 2, n_workers = 2)

    ref = reference_results['dimer_baseline_cpu_pool_2wl']

    measured = summary['peak_ext_x']
    expected = ref['peak_ext_x']
    grade = compute_grade(measured, expected)

    print('[reg] dimer pool peak_ext_x measured={:.3f} ref={:.3f} grade={}'.format(
            measured, expected, grade))

    assert grade != 'BAD', 'BAD grade: rel diff > 1e-3'
    assert summary['n_wavelengths'] == 2
