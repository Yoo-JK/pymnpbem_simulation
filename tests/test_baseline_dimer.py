import os
import sys
import json
import time
import subprocess

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_YAML = REPO_ROOT / 'examples' / 'dimer_baseline.yaml'
REFERENCE_JSON = Path('/home/yoojk20/scratch/pymnpbem_sanity_test/lane_results/baseline_cpu.json')


def grade_diff(rel: float) -> str:
    if rel < 1e-12:
        return 'machine'
    if rel < 1e-9:
        return 'OK'
    if rel < 1e-6:
        return 'good'
    if rel < 1e-3:
        return 'warn'
    return 'BAD'


def test_dimer_baseline_10wl():
    out_root = REPO_ROOT / 'results'
    name = 'dimer_baseline_10wl_test'
    out_dir = out_root / name

    if out_dir.exists():
        for f in out_dir.iterdir():
            f.unlink()
        out_dir.rmdir()

    cmd = [
        sys.executable,
        str(REPO_ROOT / 'run_simulation.py'),
        '--config', str(EXAMPLE_YAML),
        '--simulation-name', name,
        '--n-wavelengths', '10',
        '--n-workers', '1',
        '--n-threads', '4',
        '--n-gpus-per-worker', '0']

    print('[test] running:', ' '.join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, cwd = REPO_ROOT, capture_output = True, text = True)
    elapsed = time.time() - t0

    print('[test] stdout:\n', res.stdout[-2000:])
    if res.returncode != 0:
        print('[test] stderr:\n', res.stderr[-2000:])
        raise AssertionError('[error] CLI exit code = {}'.format(res.returncode))

    print('[test] CLI elapsed: {:.1f}s'.format(elapsed))

    summary_path = out_dir / 'spectrum.json'
    assert summary_path.exists(), '[error] missing <{}>'.format(summary_path)

    with open(summary_path) as f:
        summary = json.load(f)

    assert summary['n_wavelengths'] == 10
    assert summary['n_pol'] == 2
    assert summary['peak_ext_x'] > 0

    print('[test] peak_ext_x = {:.3f} at {:.2f} nm'.format(
        summary['peak_ext_x'], summary['peak_wl_nm']))
    print('[test] wall = {:.2f} min'.format(summary['wall_min']))

    if REFERENCE_JSON.exists():
        with open(REFERENCE_JSON) as f:
            ref = json.load(f)

        ref_peak = ref['peak_wl_636_36']['ext_x']
        my_peak = summary['peak_ext_x']

        rel = abs(my_peak - ref_peak) / abs(ref_peak)
        grade = grade_diff(rel)
        print('[test] peak_ext_x: my={:.3f}  ref={:.3f}  rel={:.3e}  grade=<{}>'.format(
            my_peak, ref_peak, rel, grade))

        assert grade in {'machine', 'OK', 'good', 'warn'}, \
            '[error] BAD precision: {}'.format(grade)

    return summary


if __name__ == '__main__':
    test_dimer_baseline_10wl()
