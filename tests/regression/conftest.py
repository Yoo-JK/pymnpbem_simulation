import os
import json

from pathlib import Path
from typing import Any, Dict

import pytest


REGRESSION_DIR = Path(__file__).resolve().parent
DATA_DIR = REGRESSION_DIR / 'data'
REPO_ROOT = REGRESSION_DIR.parent.parent


def pytest_configure(config):
    config.addinivalue_line('markers', 'fast: < 1 min, runs every commit')
    config.addinivalue_line('markers', 'slow: 5-30 min, daily nightly')
    config.addinivalue_line('markers', 'long: > 30 min, weekly')
    config.addinivalue_line('markers', 'gpu: requires GPU (CUDA + cupy)')
    config.addinivalue_line('markers', 'multinode: requires SLURM/MPI')

    if os.environ.get('PYMNPBEM_REGRESSION_FAST', '0') == '1':
        config.option.markexpr = (config.option.markexpr or '') + ' and fast'


@pytest.fixture(scope = 'session')
def regression_data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope = 'session')
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope = 'session')
def reference_results() -> Dict[str, Any]:
    path = DATA_DIR / 'reference_results.json'
    with open(path, 'r') as f:
        return json.load(f)


@pytest.fixture(scope = 'session')
def gpu_available() -> bool:
    try:
        import cupy
        try:
            cupy.cuda.runtime.getDeviceCount()
            return True
        except Exception:
            return False
    except ImportError:
        return False


@pytest.fixture(scope = 'session')
def slurm_available() -> bool:
    import shutil as _sh
    return _sh.which('srun') is not None


def compute_grade(measured: float, reference: float) -> str:
    """Grade measured vs reference relative error.

    Grades:
      machine precision: rel < 1e-12
      OK:                rel < 1e-9
      good:              rel < 1e-6
      warn:              rel < 1e-3
      BAD:               rel >= 1e-3
    """
    if reference == 0.0:
        return 'machine' if measured == 0.0 else 'BAD'

    rel = abs(measured - reference) / abs(reference)

    if rel < 1e-12:
        return 'machine'
    if rel < 1e-9:
        return 'OK'
    if rel < 1e-6:
        return 'good'
    if rel < 1e-3:
        return 'warn'
    return 'BAD'


@pytest.fixture(scope = 'session')
def grade_func():
    return compute_grade
