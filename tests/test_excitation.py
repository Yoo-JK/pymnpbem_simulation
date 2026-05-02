import os
import sys
import json
import time
import subprocess

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / 'examples'


def _run_cli(yaml_name: str,
        sim_name: str,
        n_wl: int = 2) -> Tuple[Dict[str, Any], int]:

    out_root = REPO_ROOT / 'results'
    out_dir = out_root / sim_name

    if out_dir.exists():
        for f in out_dir.iterdir():

            if f.is_file():
                f.unlink()

        out_dir.rmdir()

    cmd = [
            sys.executable,
            str(REPO_ROOT / 'run_simulation.py'),
            '--config', str(EXAMPLES_DIR / yaml_name),
            '--simulation-name', sim_name,
            '--n-wavelengths', str(n_wl),
            '--n-workers', '1',
            '--n-threads', '2',
            '--n-gpus-per-worker', '0']

    print('[test] running:', ' '.join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, cwd = REPO_ROOT, capture_output = True, text = True)
    elapsed = time.time() - t0

    print('[test] elapsed: {:.1f}s'.format(elapsed))
    print('[test] stdout tail:\n', res.stdout[-2000:])

    if res.returncode != 0:
        print('[test] stderr tail:\n', res.stderr[-2000:])

    summary_path = out_dir / 'spectrum.json'

    summary = dict()

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return summary, res.returncode


def test_planewave_stat_smoke():
    yaml_path = EXAMPLES_DIR / 'sphere_planewave_stat.yaml'

    if not yaml_path.exists():
        _write_planewave_stat_yaml(yaml_path)

    summary, rc = _run_cli('sphere_planewave_stat.yaml',
            'smoke_planewave_stat', n_wl = 2)

    assert rc == 0, '[error] CLI exit code = {}'.format(rc)
    assert summary.get('n_wavelengths') == 2
    assert summary.get('peak_ext_x', 0.0) > 0


def test_dipole_ret_smoke():
    summary, rc = _run_cli('sphere_dipole.yaml',
            'smoke_dipole_ret', n_wl = 2)

    assert rc == 0, '[error] CLI exit code = {}'.format(rc)
    assert summary.get('n_wavelengths') == 2

    peak_decay = summary.get('peak_ext_x', 0.0)
    print('[test] dipole peak decay_total = {:.3f}'.format(peak_decay))
    assert peak_decay > 0, '[error] dipole decay rate must be > 0, got {}'.format(peak_decay)


def test_dipole_stat_smoke():
    yaml_path = EXAMPLES_DIR / 'sphere_dipole_stat.yaml'

    if not yaml_path.exists():
        _write_dipole_stat_yaml(yaml_path)

    summary, rc = _run_cli('sphere_dipole_stat.yaml',
            'smoke_dipole_stat', n_wl = 2)

    assert rc == 0, '[error] CLI exit code = {}'.format(rc)
    assert summary.get('n_wavelengths') == 2

    peak_decay = summary.get('peak_ext_x', 0.0)
    print('[test] dipole_stat peak decay_total = {:.3f}'.format(peak_decay))
    assert peak_decay > 0


def test_eels_ret_smoke():
    summary, rc = _run_cli('sphere_eels.yaml',
            'smoke_eels_ret', n_wl = 2)

    assert rc == 0, '[error] CLI exit code = {}'.format(rc)
    assert summary.get('n_wavelengths') == 2

    peak_psurf = summary.get('peak_ext_x', 0.0)
    print('[test] eels_ret peak psurf = {:.3e}'.format(peak_psurf))
    assert peak_psurf > 0, '[error] eels loss probability must be > 0, got {}'.format(peak_psurf)


def test_eels_stat_smoke():
    yaml_path = EXAMPLES_DIR / 'sphere_eels_stat.yaml'

    if not yaml_path.exists():
        _write_eels_stat_yaml(yaml_path)

    summary, rc = _run_cli('sphere_eels_stat.yaml',
            'smoke_eels_stat', n_wl = 2)

    assert rc == 0, '[error] CLI exit code = {}'.format(rc)
    assert summary.get('n_wavelengths') == 2

    peak_psurf = summary.get('peak_ext_x', 0.0)
    print('[test] eels_stat peak psurf = {:.3e}'.format(peak_psurf))
    assert peak_psurf > 0


def _write_planewave_stat_yaml(path: Path) -> None:
    content = (
            'structure:\n'
            '  type: sphere\n'
            '  diameter: 30.0\n'
            '  mesh_density: 144\n'
            '  refine: 2\n'
            '  interp: curv\n'
            '\n'
            'simulation:\n'
            '  type: stat\n'
            '  excitation: planewave\n'
            '  enei_min: 500\n'
            '  enei_max: 700\n'
            '  n_wavelengths: 5\n'
            '  polarizations:\n'
            '    - [1, 0, 0]\n'
            '    - [0, 1, 0]\n'
            '\n'
            'materials:\n'
            '  medium: water\n'
            '  particle: gold\n'
            '\n'
            'compute:\n'
            '  n_workers: 1\n'
            '  n_threads: 2\n'
            '  n_gpus_per_worker: 0\n'
            '  multi_node: false\n'
            '  hmode: dense\n'
            '\n'
            'output:\n'
            '  dir: ./results\n'
            '  name: sphere_planewave_stat\n'
            '  formats:\n'
            '    - npz\n'
            '    - json\n'
            '  save_plots: false\n'
            '\n'
            'postprocess:\n'
            '  run_eigenmode_analysis: false\n')

    with open(path, 'w') as f:
        f.write(content)


def _write_dipole_stat_yaml(path: Path) -> None:
    content = (
            'structure:\n'
            '  type: sphere\n'
            '  diameter: 30.0\n'
            '  mesh_density: 144\n'
            '  refine: 2\n'
            '  interp: curv\n'
            '\n'
            'simulation:\n'
            '  type: stat\n'
            '  excitation: dipole\n'
            '  enei_min: 500\n'
            '  enei_max: 700\n'
            '  n_wavelengths: 5\n'
            '  dipole:\n'
            '    position: [25.0, 0.0, 0.0]\n'
            '    orientation: [1, 0, 0]\n'
            '\n'
            'materials:\n'
            '  medium: water\n'
            '  particle: gold\n'
            '\n'
            'compute:\n'
            '  n_workers: 1\n'
            '  n_threads: 2\n'
            '  n_gpus_per_worker: 0\n'
            '  multi_node: false\n'
            '  hmode: dense\n'
            '\n'
            'output:\n'
            '  dir: ./results\n'
            '  name: sphere_dipole_stat\n'
            '  formats:\n'
            '    - npz\n'
            '    - json\n'
            '  save_plots: false\n'
            '\n'
            'postprocess:\n'
            '  run_eigenmode_analysis: false\n')

    with open(path, 'w') as f:
        f.write(content)


def _write_eels_stat_yaml(path: Path) -> None:
    content = (
            'structure:\n'
            '  type: sphere\n'
            '  diameter: 30.0\n'
            '  mesh_density: 144\n'
            '  refine: 2\n'
            '  interp: curv\n'
            '\n'
            'simulation:\n'
            '  type: stat\n'
            '  excitation: eels\n'
            '  enei_min: 400\n'
            '  enei_max: 700\n'
            '  n_wavelengths: 5\n'
            '  electron:\n'
            '    impact: [[20.0, 0.0]]\n'
            '    energy_kev: 200.0\n'
            '    width: 0.5\n'
            '\n'
            'materials:\n'
            '  medium: vacuum\n'
            '  particle: gold\n'
            '\n'
            'compute:\n'
            '  n_workers: 1\n'
            '  n_threads: 2\n'
            '  n_gpus_per_worker: 0\n'
            '  multi_node: false\n'
            '  hmode: dense\n'
            '\n'
            'output:\n'
            '  dir: ./results\n'
            '  name: sphere_eels_stat\n'
            '  formats:\n'
            '    - npz\n'
            '    - json\n'
            '  save_plots: false\n'
            '\n'
            'postprocess:\n'
            '  run_eigenmode_analysis: false\n')

    with open(path, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    test_planewave_stat_smoke()
    test_dipole_ret_smoke()
    test_dipole_stat_smoke()
    test_eels_ret_smoke()
    test_eels_stat_smoke()
    print('[test] all smoke tests passed')
