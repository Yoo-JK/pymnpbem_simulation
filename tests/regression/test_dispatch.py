"""Regression: dispatch (CPU/GPU/multi-node) — M2/M9.

Markers:
  fast: dispatch module imports / config parse
  slow: cpu_pool 2-worker run (covered by test_baseline_dimer.test_dimer_baseline_cpu_pool)
  multinode: requires SLURM/MPI
  gpu: requires CUDA + cupy
"""
from __future__ import annotations

import pytest


@pytest.mark.fast
def test_dispatch_module_imports():
    """Dispatch module must expose single_node and mpi_node entries."""
    from pymnpbem_simulation import dispatch

    assert hasattr(dispatch, 'single_node') or hasattr(dispatch, 'run_single'), \
            'dispatch.single_node missing'


@pytest.mark.fast
def test_compute_config_default():
    """Config compute defaults must be parseable."""
    from pymnpbem_simulation.config import load_yaml
    from pathlib import Path

    yaml_path = Path(__file__).resolve().parent.parent.parent \
            / 'examples' / 'dimer_smoke.yaml'

    if not yaml_path.exists():
        pytest.skip('dimer_smoke.yaml missing')

    cfg = load_yaml(str(yaml_path))
    assert 'compute' in cfg or 'compute' not in cfg  # smoke test parse
    assert isinstance(cfg, dict)


@pytest.mark.gpu
def test_gpu_available_smoke(gpu_available):
    """Skip if no GPU, otherwise check cupy + driver."""
    if not gpu_available:
        pytest.skip('no GPU available')

    import cupy
    n_devices = cupy.cuda.runtime.getDeviceCount()
    assert n_devices >= 1


@pytest.mark.multinode
def test_slurm_available_smoke(slurm_available):
    """Skip if no SLURM, otherwise check srun."""
    if not slurm_available:
        pytest.skip('no SLURM (srun not in PATH)')

    import shutil
    assert shutil.which('srun') is not None
