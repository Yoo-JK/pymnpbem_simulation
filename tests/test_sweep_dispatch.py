"""Tests for sweep dispatch (--sweep-conf).

Unit tests cover sweep YAML parsing, case expansion (formats A and B),
worker plan resolution, env/cmd construction. Integration tests that
actually launch worker subprocesses are gated behind @pytest.mark.gpu.
"""

import os
import sys
import copy

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from pymnpbem_simulation.cli import (
        _build_parser, _has_required_inputs, _has_consistent_inputs)
from pymnpbem_simulation.dispatch.sweep import (
        _load_sweep_yaml, _expand_cases, _expand_grid, _build_plan,
        _build_worker_env, _build_worker_cmd, _resolve_path,
        _name_from_paths, _detect_visible_gpus, _resolve_gpu_ids)


# ----------------------------- CLI parsing -----------------------------

def test_parser_accepts_sweep_conf():
    parser = _build_parser()
    args = parser.parse_args(['--sweep-conf', 'sweep.yaml'])
    assert args.sweep_conf == 'sweep.yaml'
    assert args.str_conf is None
    assert args.sim_conf is None


def test_has_required_inputs_sweep_conf():
    import argparse
    args = argparse.Namespace(
            str_conf = None, sim_conf = None,
            config = None, sweep_conf = 'x.yaml')
    assert _has_required_inputs(args) is True


def test_has_consistent_inputs_sweep_alone():
    import argparse
    args = argparse.Namespace(
            str_conf = None, sim_conf = None,
            config = None, sweep_conf = 'x.yaml')
    assert _has_consistent_inputs(args) is True


def test_has_consistent_inputs_sweep_plus_strconf_rejected():
    import argparse
    args = argparse.Namespace(
            str_conf = 'a.py', sim_conf = 'b.py',
            config = None, sweep_conf = 'x.yaml')
    assert _has_consistent_inputs(args) is False


def test_has_consistent_inputs_sweep_plus_yaml_rejected():
    import argparse
    args = argparse.Namespace(
            str_conf = None, sim_conf = None,
            config = 'cfg.yaml', sweep_conf = 'x.yaml')
    assert _has_consistent_inputs(args) is False


def test_has_consistent_inputs_legacy_paths_unchanged():
    import argparse
    a = argparse.Namespace(
            str_conf = 'a.py', sim_conf = 'b.py',
            config = None, sweep_conf = None)
    assert _has_consistent_inputs(a) is True

    b = argparse.Namespace(
            str_conf = None, sim_conf = None,
            config = 'c.yaml', sweep_conf = None)
    assert _has_consistent_inputs(b) is True


# ------------------------- sweep YAML parsing --------------------------

def _write_yaml(path, data):
    with open(str(path), 'w') as f:
        yaml.safe_dump(data, f)


def _write_py(path, args_dict):
    with open(str(path), 'w') as f:
        f.write('args = {}\n'.format(repr(args_dict)))


def test_load_sweep_yaml_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_sweep_yaml(str(tmp_path / 'nope.yaml'))


def test_load_sweep_yaml_root_must_be_mapping(tmp_path):
    p = tmp_path / 'bad.yaml'
    with open(str(p), 'w') as f:
        f.write('- a\n- b\n')
    with pytest.raises(ValueError, match = 'mapping'):
        _load_sweep_yaml(str(p))


def test_expand_cases_format_A_str_confs(tmp_path):
    sim_path = tmp_path / 'sim.py'
    _write_py(sim_path, {'simulation_type': 'ret'})

    str0 = tmp_path / 'str0.py'
    str1 = tmp_path / 'str1.py'
    _write_py(str0, {'structure': 'sphere', 'diameter': 50})
    _write_py(str1, {'structure': 'sphere', 'diameter': 60})

    sweep_yaml = tmp_path / 'sweep.yaml'
    _write_yaml(sweep_yaml, {
            'sim_conf': 'sim.py',
            'str_confs': ['str0.py', 'str1.py']})

    cfg = _load_sweep_yaml(str(sweep_yaml))
    cases = _expand_cases(cfg, str(sweep_yaml))

    assert len(cases) == 2
    assert cases[0]['idx'] == 0
    assert cases[0]['str_conf'].endswith('str0.py')
    assert cases[0]['sim_conf'].endswith('sim.py')
    assert cases[1]['str_conf'].endswith('str1.py')
    assert cases[0]['name'] == 'str0'
    assert cases[1]['name'] == 'str1'


def test_expand_cases_format_A_explicit_cases(tmp_path):
    str0 = tmp_path / 's0.py'
    sim0 = tmp_path / 'm0.py'
    str1 = tmp_path / 's1.py'
    sim1 = tmp_path / 'm1.py'
    for p in (str0, sim0, str1, sim1):
        _write_py(p, {'foo': 1})

    sweep_yaml = tmp_path / 'sweep.yaml'
    _write_yaml(sweep_yaml, {
            'cases': [
                {'str_conf': 's0.py', 'sim_conf': 'm0.py', 'name': 'caseA'},
                {'str_conf': 's1.py', 'sim_conf': 'm1.py'}]})

    cfg = _load_sweep_yaml(str(sweep_yaml))
    cases = _expand_cases(cfg, str(sweep_yaml))

    assert len(cases) == 2
    assert cases[0]['name'] == 'caseA'
    assert cases[1]['name'] == 's1'  # auto-derived


def test_expand_cases_missing_required_raises(tmp_path):
    sweep_yaml = tmp_path / 'sweep.yaml'
    _write_yaml(sweep_yaml, {'unknown_key': 'x'})

    cfg = _load_sweep_yaml(str(sweep_yaml))
    with pytest.raises(ValueError, match = 'cases|str_confs|base_str_conf'):
        _expand_cases(cfg, str(sweep_yaml))


def test_expand_cases_str_confs_without_sim_conf(tmp_path):
    sweep_yaml = tmp_path / 'sweep.yaml'
    _write_yaml(sweep_yaml, {
            'str_confs': ['x.py']})

    cfg = _load_sweep_yaml(str(sweep_yaml))
    with pytest.raises(ValueError, match = 'sim_conf'):
        _expand_cases(cfg, str(sweep_yaml))


def test_expand_grid_format_B(tmp_path):
    base = tmp_path / 'base.py'
    _write_py(base, {
            'structure': 'advanced_dimer_cube',
            'core_size': 47,
            'gap': 1.0,
            'materials': ['gold', 'silver']})

    sim = tmp_path / 'sim.py'
    _write_py(sim, {'simulation_type': 'ret'})

    sweep_yaml = tmp_path / 'sweep.yaml'
    _write_yaml(sweep_yaml, {
            'base_str_conf': 'base.py',
            'sim_conf': 'sim.py',
            'overrides': {
                'gap': [0.6, 1.0, 2.0]}})

    cfg = _load_sweep_yaml(str(sweep_yaml))
    cases = _expand_cases(cfg, str(sweep_yaml))

    assert len(cases) == 3
    for i, expected_gap in enumerate([0.6, 1.0, 2.0]):
        assert os.path.exists(cases[i]['str_conf'])

        with open(cases[i]['str_conf']) as f:
            src = f.read()
        # the generated .py contains the override value
        assert 'gap' in src
        assert str(expected_gap) in src

        # name embeds the override slug
        assert 'gap' in cases[i]['name']


def test_expand_grid_cartesian_product(tmp_path):
    base = tmp_path / 'base.py'
    _write_py(base, {'structure': 'sphere', 'diameter': 50, 'materials': ['gold']})
    sim = tmp_path / 'sim.py'
    _write_py(sim, {'simulation_type': 'stat'})

    sweep_yaml = tmp_path / 'sweep.yaml'
    _write_yaml(sweep_yaml, {
            'base_str_conf': 'base.py',
            'sim_conf': 'sim.py',
            'overrides': {
                'diameter': [50, 60],
                'mesh_density': [2, 3]}})

    cfg = _load_sweep_yaml(str(sweep_yaml))
    cases = _expand_cases(cfg, str(sweep_yaml))

    assert len(cases) == 4  # 2 x 2 grid


# --------------------------- build_plan --------------------------------

def test_build_plan_defaults_with_explicit_n_workers(tmp_path):
    cfg = {'n_workers': 2, 'gpus_per_worker': 1, 'threads_per_worker': 4}
    plan = _build_plan(cfg, n_cases = 4)

    assert plan['n_workers'] == 2
    assert plan['gpus_per_worker'] == 1
    assert plan['threads_per_worker'] == 4


def test_build_plan_caps_n_workers_to_n_cases():
    cfg = {'n_workers': 8, 'gpus_per_worker': 1}
    plan = _build_plan(cfg, n_cases = 3)
    assert plan['n_workers'] == 3


def test_build_plan_threads_per_worker_auto():
    cfg = {'n_workers': 2, 'gpus_per_worker': 1}
    plan = _build_plan(cfg, n_cases = 2)
    cpus = os.cpu_count() or 1
    assert plan['threads_per_worker'] == max(1, cpus // 2)


def test_build_plan_default_n_workers_is_n_cases():
    cfg = {'gpus_per_worker': 1}
    plan = _build_plan(cfg, n_cases = 4)
    assert plan['n_workers'] == 4


# ----------------------- env + cmd construction ------------------------

def test_build_worker_env_pins_gpu():
    env = _build_worker_env([2], threads_per_worker = 4)
    assert env['CUDA_VISIBLE_DEVICES'] == '2'
    assert env['MNPBEM_GPU'] == '1'
    assert env['OMP_NUM_THREADS'] == '4'
    assert env['MKL_NUM_THREADS'] == '4'
    assert env['NUMBA_NUM_THREADS'] == '4'
    assert 'MNPBEM_VRAM_SHARE' not in env
    assert 'MNPBEM_VRAM_SHARE_GPUS' not in env


def test_build_worker_env_strips_inherited_vram_share(monkeypatch):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE', '1')
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '4')
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')

    env = _build_worker_env([0], threads_per_worker = 1)

    assert 'MNPBEM_VRAM_SHARE' not in env
    assert 'MNPBEM_VRAM_SHARE_GPUS' not in env
    assert 'MNPBEM_VRAM_SHARE_BACKEND' not in env


def test_build_worker_env_cpu_only():
    env = _build_worker_env([], threads_per_worker = 8)
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    assert env['MNPBEM_GPU'] == '0'


def test_build_worker_env_multi_gpu():
    env = _build_worker_env([2, 3], threads_per_worker = 1)
    assert env['CUDA_VISIBLE_DEVICES'] == '2,3'
    assert env['MNPBEM_GPU'] == '1'


def test_build_worker_cmd_basic():
    case = {
            'idx': 1,
            'str_conf': '/path/str.py',
            'sim_conf': '/path/sim.py',
            'name': 'foo'}
    plan = {
            'n_workers': 2,
            'gpus_per_worker': 1,
            'threads_per_worker': 4,
            'output_dir': '/out',
            'output_subdir_pattern': '{idx:02d}_{name}'}

    cmd = _build_worker_cmd(case, plan, verbose = False,
            n_wavelengths_override = None)

    assert '--str-conf' in cmd
    assert '/path/str.py' in cmd
    assert '--sim-conf' in cmd
    assert '/path/sim.py' in cmd
    assert '--n-workers' in cmd
    assert '1' in cmd  # n-workers value
    assert '--n-threads' in cmd
    assert '4' in cmd
    assert '--n-gpus-per-worker' in cmd
    assert '--simulation-name' in cmd
    assert '01_foo' in cmd
    assert '--output-dir' in cmd
    assert '/out' in cmd
    assert '--verbose' not in cmd


def test_build_worker_cmd_with_verbose_and_nwl():
    case = {
            'idx': 0,
            'str_conf': 'a.py',
            'sim_conf': 'b.py',
            'name': 'c'}
    plan = {
            'n_workers': 1,
            'gpus_per_worker': 0,
            'threads_per_worker': 1,
            'output_dir': None,
            'output_subdir_pattern': '{idx}_{name}'}

    cmd = _build_worker_cmd(case, plan, verbose = True,
            n_wavelengths_override = 8)

    assert '--verbose' in cmd
    assert '--n-wavelengths' in cmd
    assert '8' in cmd
    assert '--output-dir' not in cmd  # no output_dir in plan


# --------------------------- gpu id resolver ---------------------------

def test_resolve_gpu_ids_explicit_flat_list():
    cfg = {'gpu_ids': [0, 2, 4]}
    ids = _resolve_gpu_ids(cfg, n_workers = 3, gpus_per_worker = 1)
    assert ids == [[0], [2], [4]]


def test_resolve_gpu_ids_explicit_nested():
    cfg = {'gpu_ids': [[0, 1], [2, 3]]}
    ids = _resolve_gpu_ids(cfg, n_workers = 2, gpus_per_worker = 2)
    assert ids == [[0, 1], [2, 3]]


def test_resolve_gpu_ids_from_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    cfg = dict()
    ids = _resolve_gpu_ids(cfg, n_workers = 4, gpus_per_worker = 1)
    assert ids == [[0], [1], [2], [3]]


def test_resolve_gpu_ids_round_robin(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    cfg = dict()
    ids = _resolve_gpu_ids(cfg, n_workers = 4, gpus_per_worker = 1)
    # round-robin across 2 GPUs for 4 workers
    assert ids == [[0], [1], [0], [1]]


def test_resolve_gpu_ids_no_gpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    # if CUDA_VISIBLE_DEVICES='', _detect_visible_gpus returns 0
    # but _resolve_gpu_ids takes the cvd path first when set
    cfg = dict()
    ids = _resolve_gpu_ids(cfg, n_workers = 2, gpus_per_worker = 1)
    # with empty CVD, no CPU pinning (empty lists)
    assert ids == [[], []]


# ----------------------------- helpers ---------------------------------

def test_resolve_path_absolute(tmp_path):
    abs_p = '/etc/foo.yaml'
    assert _resolve_path(abs_p, str(tmp_path)) == abs_p


def test_resolve_path_relative(tmp_path):
    rel = 'sub/cfg.py'
    out = _resolve_path(rel, str(tmp_path))
    assert out.startswith(str(tmp_path))
    assert out.endswith('sub/cfg.py')


def test_name_from_paths_uses_str_basename():
    n = _name_from_paths('/x/y/auag_g0.6.py', '/x/y/sim.py', 0)
    assert n == 'auag_g0.6'


# -------------------- end-to-end (subprocess, fast) --------------------
#
# These tests intercept the worker subprocess invocation by replacing
# pymnpbem_simulation.dispatch.sweep.subprocess.run with a stub that
# records the call (env + cmd) and writes a fake spectrum.npz to mimic a
# successful CLI run. This lets us verify the dispatch mechanism (worker
# spawn, env pinning, GPU id assignment, output collision avoidance,
# return-code aggregation) without depending on the heavyweight `mnpbem`
# kernel package at all.

class _StubProc:
    def __init__(self, returncode, stdout = b''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = b''


def test_sweep_smoke_with_stub_subprocess(tmp_path, monkeypatch):
    """End-to-end smoke: dispatch_sweep with stubbed subprocess.run."""
    from pymnpbem_simulation.dispatch import sweep as sweep_mod

    str0_args = {'structure': 'sphere', 'diameter': 30, 'materials': ['gold']}
    str1_args = {'structure': 'sphere', 'diameter': 35, 'materials': ['gold']}
    sim_args = {
            'simulation_type': 'stat',
            'excitation_type': 'planewave',
            'wavelength_range': [400, 600, 4]}

    str0 = tmp_path / 'str0.py'
    str1 = tmp_path / 'str1.py'
    sim = tmp_path / 'sim.py'
    _write_py(str0, str0_args)
    _write_py(str1, str1_args)
    _write_py(sim, sim_args)

    sweep_yaml = tmp_path / 'sweep.yaml'
    out_dir = tmp_path / 'out'
    _write_yaml(sweep_yaml, {
            'sim_conf': 'sim.py',
            'str_confs': ['str0.py', 'str1.py'],
            'n_workers': 2,
            'gpus_per_worker': 1,
            'threads_per_worker': 4,
            'gpu_ids': [0, 1],
            'output_dir': str(out_dir)})

    calls = []

    def fake_run(cmd, env = None, **kwargs):
        # Filter out auto-detection probes (e.g. nvidia-smi).
        is_worker = any('pymnpbem_simulation.cli' in str(t) for t in cmd)

        if not is_worker:
            return _StubProc(0)

        # Record what each worker received.
        calls.append({
                'cmd': list(cmd),
                'cuda_visible': env.get('CUDA_VISIBLE_DEVICES'),
                'omp_threads': env.get('OMP_NUM_THREADS'),
                'mnpbem_gpu': env.get('MNPBEM_GPU'),
                'vram_share': env.get('MNPBEM_VRAM_SHARE')})

        # Write a placeholder spectrum into the output dir indicated by
        # --simulation-name so we can assert on collision-free folders.
        sim_name = None
        out_root = None
        for i, tok in enumerate(cmd):
            if tok == '--simulation-name':
                sim_name = cmd[i + 1]
            elif tok == '--output-dir':
                out_root = cmd[i + 1]
        if sim_name and out_root:
            d = Path(out_root) / sim_name
            d.mkdir(parents = True, exist_ok = True)
            (d / 'spectrum.npz').write_bytes(b'\x00')

        return _StubProc(0)

    monkeypatch.setenv('MNPBEM_SWEEP_INLINE', '1')
    monkeypatch.setattr(sweep_mod.subprocess, 'run', fake_run)

    from pymnpbem_simulation.dispatch.sweep import dispatch_sweep
    rc = dispatch_sweep(str(sweep_yaml))
    assert rc == 0
    assert len(calls) == 2

    # Each worker got a *different* GPU.
    cvds = sorted([c['cuda_visible'] for c in calls])
    assert cvds == ['0', '1']

    # Thread limits propagated.
    for c in calls:
        assert c['omp_threads'] == '4'
        assert c['mnpbem_gpu'] == '1'
        # VRAM share must be cleaned (None means key not present).
        assert c['vram_share'] is None

    # Output sub-dirs are unique per case.
    out_subs = sorted(d.name for d in out_dir.iterdir() if d.is_dir())
    assert out_subs == ['00_str0', '01_str1']
    for d in out_dir.iterdir():
        assert (d / 'spectrum.npz').exists()


def test_sweep_partial_failure_returns_nonzero(tmp_path, monkeypatch):
    from pymnpbem_simulation.dispatch import sweep as sweep_mod

    str0 = tmp_path / 's0.py'
    str1 = tmp_path / 's1.py'
    sim = tmp_path / 'sim.py'
    for p in (str0, str1, sim):
        _write_py(p, {'foo': 1})

    sweep_yaml = tmp_path / 'sweep.yaml'
    _write_yaml(sweep_yaml, {
            'sim_conf': 'sim.py',
            'str_confs': ['s0.py', 's1.py'],
            'n_workers': 2,
            'gpus_per_worker': 0,
            'output_dir': str(tmp_path / 'out')})

    seen = []

    def fake_run(cmd, env = None, **kwargs):
        is_worker = any('pymnpbem_simulation.cli' in str(t) for t in cmd)
        if not is_worker:
            return _StubProc(0)
        seen.append(cmd)
        # 2nd worker fails
        rc = 0 if len(seen) == 1 else 7
        return _StubProc(rc)

    monkeypatch.setenv('MNPBEM_SWEEP_INLINE', '1')
    monkeypatch.setattr(sweep_mod.subprocess, 'run', fake_run)

    from pymnpbem_simulation.dispatch.sweep import dispatch_sweep
    rc = dispatch_sweep(str(sweep_yaml))
    assert rc != 0


def test_sweep_smoke_fixture_loadable(monkeypatch):
    """Verify the shipped tests/fixtures/sweep_smoke.yaml resolves into
    well-formed worker commands. Uses inline mode + stub run.
    """
    from pymnpbem_simulation.dispatch import sweep as sweep_mod

    fixture = REPO_ROOT / 'tests' / 'fixtures' / 'sweep_smoke.yaml'
    assert fixture.exists()

    seen = []

    def fake_run(cmd, env = None, **kwargs):
        if any('pymnpbem_simulation.cli' in str(t) for t in cmd):
            seen.append({'cmd': cmd, 'env': dict(env or {})})
        return _StubProc(0)

    monkeypatch.setenv('MNPBEM_SWEEP_INLINE', '1')
    monkeypatch.setattr(sweep_mod.subprocess, 'run', fake_run)

    from pymnpbem_simulation.dispatch.sweep import dispatch_sweep
    rc = dispatch_sweep(str(fixture))

    assert rc == 0
    assert len(seen) == 2

    # both workers got n-gpus-per-worker = 0 (CPU smoke fixture)
    for s in seen:
        idx = s['cmd'].index('--n-gpus-per-worker')
        assert s['cmd'][idx + 1] == '0'
        assert s['env']['MNPBEM_GPU'] == '0'

    # output sub-dirs unique per case
    sim_names = []
    for s in seen:
        idx = s['cmd'].index('--simulation-name')
        sim_names.append(s['cmd'][idx + 1])
    assert sim_names[0] != sim_names[1]


def test_sweep_more_workers_than_cases_caps_correctly(tmp_path, monkeypatch):
    from pymnpbem_simulation.dispatch import sweep as sweep_mod

    str0 = tmp_path / 's0.py'
    sim = tmp_path / 'sim.py'
    _write_py(str0, {'foo': 1})
    _write_py(sim, {'foo': 1})

    sweep_yaml = tmp_path / 'sweep.yaml'
    _write_yaml(sweep_yaml, {
            'sim_conf': 'sim.py',
            'str_confs': ['s0.py'],
            'n_workers': 4,  # more than n_cases (1)
            'gpus_per_worker': 0,
            'output_dir': str(tmp_path / 'out')})

    worker_calls = []

    def fake_run(cmd, env = None, **kwargs):
        # Only count worker cmds (skip nvidia-smi auto-detection probes).
        if any('pymnpbem_simulation.cli' in str(t) for t in cmd):
            worker_calls.append(cmd)
        return _StubProc(0)

    monkeypatch.setenv('MNPBEM_SWEEP_INLINE', '1')
    monkeypatch.setattr(sweep_mod.subprocess, 'run', fake_run)

    from pymnpbem_simulation.dispatch.sweep import dispatch_sweep
    rc = dispatch_sweep(str(sweep_yaml))
    assert rc == 0
    assert len(worker_calls) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
