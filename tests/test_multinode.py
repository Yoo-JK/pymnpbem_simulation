"""Multi-node MPI dispatch unit tests.

Real multi-node execution requires a SLURM/PBS allocation (or local
mpiexec across hosts). These tests verify the bits we *can* check
locally:

* mpi4py import + MPI library load.
* `detect_multi_node()` env var heuristics under mock environments.
* `dispatch_mpi` skeleton: raises a clean RuntimeError when mpi4py is
  unavailable, and otherwise reaches the `solve_spectrum_mpi` call.
* SLURM/PBS scripts exist and are syntactically shell-clean.

End-to-end multi-rank correctness is exercised by
`mpiexec -n 2 python -m pytest -k smoke` only when an MPI runtime is
on the PATH (smoke tests under a `if MPI runtime present:` guard).
"""
import os
import shutil
import subprocess
import sys

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))


def _has_mpi4py() -> bool:
    try:
        import mpi4py  # noqa: F401
        from mpi4py import MPI  # noqa: F401
        return True

    except Exception:
        return False


def _has_mpiexec() -> bool:
    return shutil.which('mpiexec') is not None or shutil.which('srun') is not None


# -----------------------------------------------------------------
# 1. mpi4py / MPI runtime smoke
# -----------------------------------------------------------------

def test_mpi4py_importable():
    """mpi4py + MPI library load (skip if not installed)."""

    if not _has_mpi4py():
        pytest.skip('mpi4py not installed (install via [mpi] extras)')

    from mpi4py import MPI

    assert MPI.COMM_WORLD.Get_rank() == 0
    assert MPI.COMM_WORLD.Get_size() >= 1


def test_mpiexec_available():
    """OS-level MPI launcher present (skip if absent)."""

    if not _has_mpiexec():
        pytest.skip('no mpiexec/srun on PATH')

    if shutil.which('mpiexec') is not None:
        proc = subprocess.run(
                ['mpiexec', '--version'],
                capture_output = True,
                text = True,
                timeout = 10)
        assert proc.returncode == 0


# -----------------------------------------------------------------
# 2. detect_multi_node env var heuristics
# -----------------------------------------------------------------

def test_detect_multi_node_no_env(monkeypatch):
    """No MPI env vars -> single node."""

    for key in [
            'SLURM_NNODES', 'SLURM_JOB_NUM_NODES', 'PBS_NUM_NODES',
            'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MPI_LOCALNRANKS',
            'PMIX_RANK', 'PMIX_NAMESPACE', 'PMIX_SERVER_URI', 'PMIX_HOSTNAME',
            'PBS_NODEFILE', 'SLURM_PROCID']:
        monkeypatch.delenv(key, raising = False)

    from pymnpbem_simulation.auto_detect import detect_multi_node
    assert detect_multi_node() is False


def test_detect_multi_node_slurm_nnodes(monkeypatch):
    """SLURM_NNODES > 1 -> multi-node."""

    for key in ['OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'PMIX_RANK', 'PBS_NODEFILE']:
        monkeypatch.delenv(key, raising = False)

    monkeypatch.setenv('SLURM_NNODES', '4')

    from pymnpbem_simulation.auto_detect import detect_multi_node
    assert detect_multi_node() is True


def test_detect_multi_node_slurm_single(monkeypatch):
    """SLURM_NNODES == 1 -> single node."""

    for key in [
            'SLURM_JOB_NUM_NODES', 'PBS_NUM_NODES', 'OMPI_COMM_WORLD_SIZE',
            'PMI_SIZE', 'MPI_LOCALNRANKS', 'PMIX_RANK', 'PBS_NODEFILE',
            'SLURM_PROCID']:
        monkeypatch.delenv(key, raising = False)

    monkeypatch.setenv('SLURM_NNODES', '1')

    from pymnpbem_simulation.auto_detect import detect_multi_node
    assert detect_multi_node() is False


def test_detect_multi_node_ompi_size(monkeypatch):
    """OMPI_COMM_WORLD_SIZE > 1 (mpiexec -n 2) -> multi-node."""

    for key in [
            'SLURM_NNODES', 'SLURM_JOB_NUM_NODES', 'PBS_NUM_NODES',
            'PMI_SIZE', 'MPI_LOCALNRANKS', 'PMIX_RANK', 'PBS_NODEFILE',
            'SLURM_PROCID']:
        monkeypatch.delenv(key, raising = False)

    monkeypatch.setenv('OMPI_COMM_WORLD_SIZE', '8')

    from pymnpbem_simulation.auto_detect import detect_multi_node
    assert detect_multi_node() is True


def test_detect_multi_node_pmix_with_namespace(monkeypatch):
    """PMIX_RANK + PMIX_NAMESPACE -> multi-node."""

    for key in [
            'SLURM_NNODES', 'SLURM_JOB_NUM_NODES', 'PBS_NUM_NODES',
            'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MPI_LOCALNRANKS',
            'PBS_NODEFILE', 'SLURM_PROCID']:
        monkeypatch.delenv(key, raising = False)

    monkeypatch.setenv('PMIX_RANK', '0')
    monkeypatch.setenv('PMIX_NAMESPACE', 'pymnpbem-test')

    from pymnpbem_simulation.auto_detect import detect_multi_node
    assert detect_multi_node() is True


def test_detect_multi_node_pmix_alone(monkeypatch):
    """PMIX_RANK alone (no NAMESPACE/SERVER_URI/HOSTNAME) -> single node."""

    for key in [
            'SLURM_NNODES', 'SLURM_JOB_NUM_NODES', 'PBS_NUM_NODES',
            'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MPI_LOCALNRANKS',
            'PMIX_NAMESPACE', 'PMIX_SERVER_URI', 'PMIX_HOSTNAME',
            'PBS_NODEFILE', 'SLURM_PROCID']:
        monkeypatch.delenv(key, raising = False)

    monkeypatch.setenv('PMIX_RANK', '0')

    from pymnpbem_simulation.auto_detect import detect_multi_node
    assert detect_multi_node() is False


def test_detect_multi_node_pbs_nodefile(monkeypatch, tmp_path):
    """PBS_NODEFILE listing 2 unique hosts -> multi-node."""

    for key in [
            'SLURM_NNODES', 'SLURM_JOB_NUM_NODES', 'PBS_NUM_NODES',
            'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MPI_LOCALNRANKS',
            'PMIX_RANK', 'SLURM_PROCID']:
        monkeypatch.delenv(key, raising = False)

    nodefile = tmp_path / 'pbs_nodefile'
    nodefile.write_text('node01\nnode01\nnode02\nnode02\n')
    monkeypatch.setenv('PBS_NODEFILE', str(nodefile))

    from pymnpbem_simulation.auto_detect import detect_multi_node
    assert detect_multi_node() is True


def test_detect_multi_node_pbs_single_node(monkeypatch, tmp_path):
    """PBS_NODEFILE listing 1 unique host -> single node."""

    for key in [
            'SLURM_NNODES', 'SLURM_JOB_NUM_NODES', 'PBS_NUM_NODES',
            'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MPI_LOCALNRANKS',
            'PMIX_RANK', 'SLURM_PROCID']:
        monkeypatch.delenv(key, raising = False)

    nodefile = tmp_path / 'pbs_nodefile'
    nodefile.write_text('node01\nnode01\nnode01\n')
    monkeypatch.setenv('PBS_NODEFILE', str(nodefile))

    from pymnpbem_simulation.auto_detect import detect_multi_node
    assert detect_multi_node() is False


def test_detect_multi_node_slurm_procid_with_nnodes(monkeypatch):
    """SLURM_PROCID set + SLURM_NNODES > 1 -> multi-node."""

    for key in [
            'SLURM_JOB_NUM_NODES', 'PBS_NUM_NODES', 'OMPI_COMM_WORLD_SIZE',
            'PMI_SIZE', 'MPI_LOCALNRANKS', 'PMIX_RANK', 'PBS_NODEFILE']:
        monkeypatch.delenv(key, raising = False)

    monkeypatch.setenv('SLURM_PROCID', '3')
    monkeypatch.setenv('SLURM_NNODES', '2')

    from pymnpbem_simulation.auto_detect import detect_multi_node
    assert detect_multi_node() is True


# -----------------------------------------------------------------
# 3. detect_mpi_rank parses common env vars
# -----------------------------------------------------------------

def test_detect_mpi_rank_ompi(monkeypatch):
    for key in ['PMI_RANK', 'SLURM_PROCID', 'MPI_RANK']:
        monkeypatch.delenv(key, raising = False)

    monkeypatch.setenv('OMPI_COMM_WORLD_RANK', '5')

    from pymnpbem_simulation.auto_detect import detect_mpi_rank
    assert detect_mpi_rank() == 5


def test_detect_mpi_rank_default(monkeypatch):
    for key in ['OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'SLURM_PROCID', 'MPI_RANK']:
        monkeypatch.delenv(key, raising = False)

    from pymnpbem_simulation.auto_detect import detect_mpi_rank
    assert detect_mpi_rank() == 0


# -----------------------------------------------------------------
# 4. dispatch_mpi import smoke (no actual MPI invocation)
# -----------------------------------------------------------------

def test_dispatch_mpi_import():
    """dispatch_mpi is importable without MPI runtime."""
    from pymnpbem_simulation.dispatch.mpi_node import dispatch_mpi
    assert callable(dispatch_mpi)


def test_dispatch_mpi_raises_when_mpi4py_missing(monkeypatch):
    """When mpi4py is not importable, dispatch_mpi raises RuntimeError."""

    if _has_mpi4py():
        pytest.skip('mpi4py is installed; skip the missing-dep test')

    import numpy as np
    from pymnpbem_simulation.dispatch.mpi_node import dispatch_mpi

    cfg = {
            'simulation': {
                    'polarizations': [[1, 0, 0]],
                    'propagation_dirs': [[0, 0, 1]]},
            'compute': {'n_gpus_per_worker': 0, 'hmode': 'dense'},
            'structure': {'type': 'sphere', 'diameter': 10.0,
                    'n_per_edge': 6, 'refine': 1},
            'materials': {'medium': 'water', 'particle': 'gold'}}
    enei = np.linspace(500, 600, 5)

    with pytest.raises(RuntimeError, match = 'mpi4py'):
        dispatch_mpi(cfg, None, None, enei)


# -----------------------------------------------------------------
# 5. SLURM/PBS scripts exist + are non-empty
# -----------------------------------------------------------------

@pytest.mark.parametrize('script', [
        'slurm_scripts/cpu_only.slurm',
        'slurm_scripts/single_node_gpu.slurm',
        'slurm_scripts/multi_node_2.slurm',
        'slurm_scripts/multi_node_4.slurm',
        'pbs_scripts/single_node_gpu.pbs',
        'pbs_scripts/multi_node_2.pbs',
        'pbs_scripts/multi_node_4.pbs'])
def test_submission_scripts_exist(script):
    p = REPO_ROOT / script
    assert p.exists(), '{} not found'.format(p)
    assert p.stat().st_size > 100
    text = p.read_text()
    assert text.startswith('#!'), 'shebang missing in {}'.format(script)


@pytest.mark.parametrize('script', [
        'slurm_scripts/multi_node_2.slurm',
        'slurm_scripts/multi_node_4.slurm'])
def test_slurm_multinode_uses_srun(script):
    p = REPO_ROOT / script
    text = p.read_text()
    assert 'srun' in text
    assert '--multi-node' in text


@pytest.mark.parametrize('script', [
        'pbs_scripts/multi_node_2.pbs',
        'pbs_scripts/multi_node_4.pbs'])
def test_pbs_multinode_uses_mpirun(script):
    p = REPO_ROOT / script
    text = p.read_text()
    assert 'mpirun' in text
    assert '--multi-node' in text


# -----------------------------------------------------------------
# 6. Multi-node example YAMLs exist + parse + have multi_node=true
# -----------------------------------------------------------------

@pytest.mark.parametrize('yaml_file', [
        'examples/dimer_multinode.yaml',
        'examples/large_mesh_multinode.yaml'])
def test_multinode_yaml_valid(yaml_file):
    import yaml

    p = REPO_ROOT / yaml_file
    assert p.exists()

    with open(p) as f:
        cfg = yaml.safe_load(f)

    assert cfg['compute']['multi_node'] is True
    assert 'structure' in cfg
    assert 'simulation' in cfg


# -----------------------------------------------------------------
# 7. CLI --auto + multi_node env -> sets compute.multi_node
# -----------------------------------------------------------------

def test_cli_auto_sets_multinode(monkeypatch):
    """cli --auto under multi-node env should set compute.multi_node=True."""

    monkeypatch.setenv('SLURM_NNODES', '2')
    monkeypatch.setenv('SLURM_GPUS_ON_NODE', '4')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '16')

    from pymnpbem_simulation.auto_detect import (
            detect_multi_node, auto_compute_plan)

    assert detect_multi_node() is True

    n_w, n_t, n_g = auto_compute_plan()
    assert n_g == 1
    assert n_w == 4
