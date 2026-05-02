"""Regression: 14 structure builders + REGISTRY (M4).

Markers:
  fast: build-only smoke for all 14 structures
  slow: small CLI run for cube (peak ext_x check)
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


STRUCTURE_CFG = {
    'sphere':                   {'type': 'sphere', 'diameter': 30, 'mesh_density': 60},
    'cube':                     {'type': 'cube', 'size': 20, 'n_per_edge': 6, 'e': 0.25},
    'rod':                      {'type': 'rod', 'diameter': 10, 'height': 30, 'mesh_density': 6.0},
    'ellipsoid':                {'type': 'ellipsoid', 'axes': [10, 12, 15], 'n_verts': 60},
    'triangle':                 {'type': 'triangle', 'side_length': 30, 'thickness': 5, 'nz': 5},
    'dimer_sphere':             {'type': 'dimer_sphere', 'diameter': 30, 'gap': 5, 'n_verts': 60},
    'dimer_cube':               {'type': 'dimer_cube', 'edge': 20, 'gap': 5, 'n_per_edge': 6, 'e': 0.25},
    'core_shell_sphere':        {'type': 'core_shell_sphere', 'core_diameter': 20,
                                 'shell_thickness': 5, 'mesh_density': 60},
    'core_shell_cube':          {'type': 'core_shell_cube', 'core_size': 20,
                                 'shell_thickness': 5, 'n_per_edge': 6, 'e': 0.25},
    'core_shell_rod':           {'type': 'core_shell_rod', 'core_diameter': 10,
                                 'shell_thickness': 3, 'height': 40, 'mesh_density': 6.0},
    'dimer_core_shell_cube':    {'type': 'dimer_core_shell_cube', 'core_size': 20,
                                 'shell_thickness': 5, 'gap': 3, 'n_per_edge': 6},
    'advanced_monomer_cube':    {'type': 'advanced_monomer_cube', 'core_size': 20,
                                 'shell_layers': [5], 'materials': ['gold', 'silver'],
                                 'n_per_edge': 6},
    'advanced_dimer_cube':      {'type': 'advanced_dimer_cube', 'core_size': 20,
                                 'shell_layers': [5], 'materials': ['gold', 'silver'],
                                 'gap': 3, 'tilt_angle': 10, 'rotation_angle': 5,
                                 'n_per_edge': 6},
    'connected_dimer_cube':     {'type': 'connected_dimer_cube', 'core_size': 15,
                                 'gap': 0.0, 'n_per_edge': 5, 'e': 0.25},
    'sphere_cluster_aggregate': {'type': 'sphere_cluster_aggregate', 'n_spheres': 3,
                                 'diameter': 30, 'gap': -0.1, 'n_verts': 60}}


@pytest.mark.fast
@pytest.mark.parametrize('stype', sorted(STRUCTURE_CFG.keys()))
def test_structure_builds(stype: str) -> None:
    """All 14 structures must register and build without exceptions."""
    from pymnpbem_simulation.structures import build_structure, REGISTRY

    assert stype in REGISTRY, 'missing <{}> in REGISTRY'.format(stype)

    cfg_struct = STRUCTURE_CFG[stype]
    cfg_mat = {'medium': 'water', 'particle': 'gold',
            'core': 'gold', 'shell': 'silver'}

    p, epstab, nfaces = build_structure(cfg_struct, cfg_mat)

    assert nfaces > 0, '<{}> built with 0 faces'.format(stype)
    assert isinstance(epstab, list)
    assert len(epstab) >= 2


@pytest.mark.fast
def test_registry_completeness() -> None:
    """REGISTRY must contain all expected structure keys."""
    from pymnpbem_simulation.structures import REGISTRY

    expected = {
            'sphere', 'cube', 'rod', 'ellipsoid', 'triangle',
            'dimer_sphere', 'dimer_cube',
            'core_shell_sphere', 'core_shell_cube', 'core_shell_rod',
            'dimer_core_shell_cube',
            'advanced_monomer_cube', 'advanced_dimer_cube',
            'connected_dimer_cube',
            'sphere_cluster_aggregate',
            'from_shape'}

    missing = expected - set(REGISTRY.keys())
    assert not missing, 'missing keys: {}'.format(missing)


@pytest.mark.fast
def test_unknown_structure_raises() -> None:
    from pymnpbem_simulation.structures import build_structure

    with pytest.raises(ValueError):
        build_structure({'type': 'no_such_structure'},
                {'medium': 'water', 'particle': 'gold'})


@pytest.mark.slow
def test_cube_cli_smoke(repo_root, reference_results):
    """Wave 2 smoke 2: cube CLI run, expected peak ext_x = 1678.476.

    Uses examples/cube.yaml directly.
    """
    from .runners.compute_grade import compute_grade

    yaml_path = repo_root / 'examples' / 'cube.yaml'
    if not yaml_path.exists():
        pytest.skip('cube.yaml not in examples')

    name = 'reg_cube_2wl'
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
            '--n-threads', '2',
            '--n-gpus-per-worker', '0']

    print('[reg] cmd: {}'.format(' '.join(cmd)))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd = repo_root, capture_output = True, text = True)
    elapsed = time.time() - t0
    print('[reg] elapsed: {:.1f}s'.format(elapsed))

    if proc.returncode != 0:
        print('[reg] stderr (last 20 lines):')
        for line in proc.stderr.splitlines()[-20:]:
            print('  {}'.format(line))
        raise AssertionError('CLI rc={}'.format(proc.returncode))

    summary_path = out_dir / 'spectrum.json'
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)

    ref = reference_results['cube_2wl']
    measured = summary['peak_ext_x']
    expected = ref['peak_ext_x']
    grade = compute_grade(measured, expected)

    print('[reg] cube 2wl peak_ext_x measured={:.3f} ref={:.3f} grade={}'.format(
            measured, expected, grade))

    assert grade != 'BAD', 'BAD grade: rel > 1e-3'
