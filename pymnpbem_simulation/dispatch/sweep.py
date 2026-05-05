"""Sweep dispatch — fan out N independent (str_conf, sim_conf) runs across
worker processes, each pinned to a single GPU via CUDA_VISIBLE_DEVICES.

This is a thin orchestrator: each worker invokes the regular single-node
CLI path (`run_simulation.py --str-conf ... --sim-conf ...`) inside a
subprocess. We do NOT re-implement dispatch here — we just multiplex the
existing dispatch_single_node entry over a worker pool.

Use case: comparing multiple structures on a node with G GPUs. With
G workers (one per GPU), throughput is ~G x compared to running them
serially with VRAM-share over all G GPUs.
"""

import os
import sys
import copy
import time
import subprocess
import multiprocessing as mp
import itertools

from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..util import print_info, print_error, ensure_dir


def dispatch_sweep(sweep_conf_path: str,
        extra_overrides: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        n_wavelengths_override: Optional[int] = None) -> int:
    """Entry point invoked by cli.main() when --sweep-conf is given.

    Parameters
    ----------
    sweep_conf_path : str
        Path to sweep YAML.
    extra_overrides : dict, optional
        CLI-level overrides (--n-workers, --n-threads, ...) that should
        propagate to the sweep plan. Keys: 'compute', 'output'.
    verbose : bool
        Forwards --verbose flag to each worker.
    n_wavelengths_override : int, optional
        Forwards --n-wavelengths to each worker (debug).

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on partial failure.
    """
    print_info('sweep: loading <{}>'.format(sweep_conf_path))

    sweep_cfg = _load_sweep_yaml(sweep_conf_path)
    sweep_cfg = _apply_cli_overrides(sweep_cfg, extra_overrides)

    cases = _expand_cases(sweep_cfg, sweep_conf_path)

    if len(cases) == 0:
        print_error('sweep: no cases generated from <{}>'.format(sweep_conf_path))
        return 4

    plan = _build_plan(sweep_cfg, n_cases = len(cases))

    print_info('sweep: {} case(s), n_workers={}, gpus_per_worker={}, '
            'threads_per_worker={}'.format(
            len(cases), plan['n_workers'], plan['gpus_per_worker'],
            plan['threads_per_worker']))

    for i, case in enumerate(cases):
        print_info('  [{}] str=<{}> sim=<{}> name=<{}>'.format(
                i, _short(case['str_conf']), _short(case['sim_conf']),
                case['name']))

    t0 = time.time()
    results = _run_pool(cases, plan,
            verbose = verbose,
            n_wavelengths_override = n_wavelengths_override)
    wall_s = time.time() - t0

    n_ok = sum(1 for r in results if r['returncode'] == 0)
    n_fail = len(results) - n_ok

    print_info('sweep: done in {:.2f} min — {} ok, {} fail'.format(
            wall_s / 60.0, n_ok, n_fail))

    for r in results:
        status = 'OK' if r['returncode'] == 0 else 'FAIL ({})'.format(r['returncode'])
        print_info('  [{}] gpu={} wall={:.1f}s name=<{}> {}'.format(
                r['idx'], r['gpu_id'], r['wall_s'], r['name'], status))

    return 0 if n_fail == 0 else 5


def _load_sweep_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
                '[error] sweep config not found: <{}>'.format(path))

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
                '[error] sweep YAML root must be a mapping in <{}>'.format(path))

    return raw


def _apply_cli_overrides(sweep_cfg: Dict[str, Any],
        overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return sweep_cfg

    out = copy.deepcopy(sweep_cfg)

    compute = overrides.get('compute', dict()) or dict()

    if compute.get('n_workers') is not None:
        out['n_workers'] = int(compute['n_workers'])

    if compute.get('n_threads') is not None:
        out['threads_per_worker'] = int(compute['n_threads'])

    if compute.get('n_gpus_per_worker') is not None:
        out['gpus_per_worker'] = int(compute['n_gpus_per_worker'])

    output = overrides.get('output', dict()) or dict()

    if output.get('dir') is not None:
        out['output_dir'] = output['dir']

    return out


def _expand_cases(sweep_cfg: Dict[str, Any],
        sweep_conf_path: str) -> List[Dict[str, Any]]:
    """Resolve sweep YAML into a flat list of (str_conf, sim_conf, name)
    cases.

    Supports two formats:

    Format A (explicit list)::

        sim_conf: configs/jk/sim_default.py
        str_confs:
          - configs/jk/.../auag_g0.6.py
          - configs/jk/.../auag_g1.0.py
        # OR a list of dicts:
        cases:
          - str_conf: ...
            sim_conf: ...
            name: foo

    Format B (parameter grid — auto-generated str_conf .py files written
    to a tmp dir)::

        base_str_conf: configs/jk/dimer_auag_4nm_r0.2/auag_base.py
        sim_conf: configs/jk/sim_default.py
        overrides:
          gap: [0.6, 1.0, 2.0, 3.0]
        n_workers: 4
    """
    base_dir = os.path.dirname(os.path.abspath(sweep_conf_path))
    cases: List[Dict[str, Any]] = []

    # Format A.1 — explicit cases list
    if 'cases' in sweep_cfg:
        cases_block = sweep_cfg['cases']

        if not isinstance(cases_block, list) or not cases_block:
            raise ValueError('[error] sweep YAML <cases> must be a non-empty list')

        for i, c in enumerate(cases_block):

            if not isinstance(c, dict):
                raise ValueError(
                        '[error] sweep cases[{}] must be a mapping'.format(i))

            if 'str_conf' not in c or 'sim_conf' not in c:
                raise ValueError(
                        '[error] sweep cases[{}] missing str_conf or sim_conf'.format(i))

            str_path = _resolve_path(c['str_conf'], base_dir)
            sim_path = _resolve_path(c['sim_conf'], base_dir)
            name = c.get('name', None) or _name_from_paths(str_path, sim_path, i)

            cases.append({
                    'idx': i,
                    'str_conf': str_path,
                    'sim_conf': sim_path,
                    'name': name})

        return cases

    # Format A.2 — single sim_conf + list of str_confs
    if 'str_confs' in sweep_cfg:
        sim_conf = sweep_cfg.get('sim_conf')

        if not sim_conf:
            raise ValueError(
                    '[error] sweep YAML with <str_confs> must define <sim_conf>')

        sim_path = _resolve_path(sim_conf, base_dir)
        str_list = sweep_cfg['str_confs']

        if not isinstance(str_list, list) or not str_list:
            raise ValueError('[error] sweep <str_confs> must be a non-empty list')

        for i, s in enumerate(str_list):
            str_path = _resolve_path(s, base_dir)
            name = _name_from_paths(str_path, sim_path, i)
            cases.append({
                    'idx': i,
                    'str_conf': str_path,
                    'sim_conf': sim_path,
                    'name': name})

        return cases

    # Format B — parameter grid, auto-generate str_conf .py files
    if 'base_str_conf' in sweep_cfg:
        return _expand_grid(sweep_cfg, base_dir)

    raise ValueError(
            '[error] sweep YAML must define one of: <cases>, <str_confs>, <base_str_conf>')


def _expand_grid(sweep_cfg: Dict[str, Any],
        base_dir: str) -> List[Dict[str, Any]]:
    from ..config import load_py_config

    base_str_conf = _resolve_path(sweep_cfg['base_str_conf'], base_dir)
    sim_conf = sweep_cfg.get('sim_conf')

    if not sim_conf:
        raise ValueError(
                '[error] sweep grid mode requires <sim_conf>')

    sim_path = _resolve_path(sim_conf, base_dir)

    overrides = sweep_cfg.get('overrides', dict())

    if not isinstance(overrides, dict) or not overrides:
        raise ValueError(
                '[error] sweep grid mode requires <overrides> dict (param: [values])')

    # Build the cartesian product of overrides. Lists become axes.
    keys = []
    value_lists = []

    for k, v in overrides.items():

        if not isinstance(v, list):
            v = [v]

        keys.append(k)
        value_lists.append(v)

    base_args = load_py_config(base_str_conf)

    out_dir = sweep_cfg.get('grid_workdir',
            os.path.join(base_dir, '_sweep_grid'))
    ensure_dir(out_dir)

    cases: List[Dict[str, Any]] = []
    idx = 0

    for combo in itertools.product(*value_lists):
        args = copy.deepcopy(base_args)
        suffix_parts = []

        for k, v in zip(keys, combo):
            args[k] = v
            suffix_parts.append('{}={}'.format(k, _slug(v)))

        suffix = '_'.join(suffix_parts) or 'case{}'.format(idx)
        gen_str_path = os.path.join(out_dir, '{}.py'.format(suffix))

        with open(gen_str_path, 'w') as f:
            f.write('# auto-generated by sweep grid mode\n')
            f.write('args = {}\n'.format(repr(args)))

        cases.append({
                'idx': idx,
                'str_conf': gen_str_path,
                'sim_conf': sim_path,
                'name': suffix})
        idx += 1

    return cases


def _build_plan(sweep_cfg: Dict[str, Any],
        n_cases: int) -> Dict[str, Any]:
    """Resolve worker count, GPU pinning, and thread budget."""
    auto = bool(sweep_cfg.get('auto', False))

    # n_workers — explicit > auto > n_cases-capped default
    n_workers = sweep_cfg.get('n_workers', None)

    if n_workers is None:
        if auto:
            n_workers = _detect_visible_gpus() or os.cpu_count() or 1
        else:
            n_workers = n_cases

    n_workers = max(1, int(n_workers))
    n_workers = min(n_workers, n_cases)

    gpus_per_worker = int(sweep_cfg.get('gpus_per_worker', 1))

    # threads_per_worker — explicit > floor(cpus / n_workers) > 1
    threads_per_worker = sweep_cfg.get('threads_per_worker', None)

    if threads_per_worker is None:
        cpus = os.cpu_count() or 1
        threads_per_worker = max(1, cpus // n_workers)

    threads_per_worker = max(1, int(threads_per_worker))

    # GPU id assignment — round-robin from CUDA_VISIBLE_DEVICES (or 0..G-1)
    gpu_ids = _resolve_gpu_ids(sweep_cfg, n_workers, gpus_per_worker)

    output_dir = sweep_cfg.get('output_dir', None)
    output_subdir_pattern = sweep_cfg.get('output_subdir_pattern',
            '{idx:02d}_{name}')

    return {
            'n_workers': n_workers,
            'gpus_per_worker': gpus_per_worker,
            'threads_per_worker': threads_per_worker,
            'gpu_ids': gpu_ids,
            'output_dir': output_dir,
            'output_subdir_pattern': output_subdir_pattern,
            'auto': auto}


def _detect_visible_gpus() -> int:
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    if cvd is not None:
        if cvd == '':
            return 0
        return len([s for s in cvd.split(',') if s.strip()])

    # Try nvidia-smi as a fallback.
    try:
        out = subprocess.check_output(
                ['nvidia-smi', '-L'], stderr = subprocess.DEVNULL,
                timeout = 5).decode('utf-8', 'ignore')
        return len([line for line in out.splitlines() if line.strip()])
    except Exception:
        return 0


def _resolve_gpu_ids(sweep_cfg: Dict[str, Any],
        n_workers: int,
        gpus_per_worker: int) -> List[List[int]]:
    """Assign physical GPU id(s) to each worker.

    Returns
    -------
    list of list of int
        gpu_ids[w] = list of GPU ids visible to worker w (1 entry per
        gpus_per_worker; CUDA_VISIBLE_DEVICES is set to ','.join(...)).
    """
    explicit = sweep_cfg.get('gpu_ids', None)

    if explicit is not None:
        # accept either a flat list of ints (1 GPU per worker) or list of lists
        if isinstance(explicit, list) and explicit and isinstance(explicit[0], list):
            return [[int(g) for g in row] for row in explicit][:n_workers]
        flat = [int(g) for g in explicit]
        return [[flat[w % len(flat)]] for w in range(n_workers)]

    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    if cvd is not None:
        # CUDA_VISIBLE_DEVICES is set (possibly to '') — respect it.
        # Empty string means "no GPUs visible".
        if cvd == '':
            return [[] for _ in range(n_workers)]
        avail = [int(s) for s in cvd.split(',') if s.strip()]
    else:
        # detect from nvidia-smi
        n_detected = _detect_visible_gpus()
        avail = list(range(n_detected)) if n_detected > 0 else []

    if not avail:
        # CPU-only sweep — no GPU pinning
        return [[] for _ in range(n_workers)]

    out: List[List[int]] = []
    cursor = 0

    for w in range(n_workers):
        ids = []

        for _ in range(gpus_per_worker):
            ids.append(avail[cursor % len(avail)])
            cursor += 1

        out.append(ids)

    return out


def _run_pool(cases: List[Dict[str, Any]],
        plan: Dict[str, Any],
        verbose: bool,
        n_wavelengths_override: Optional[int]) -> List[Dict[str, Any]]:
    """Spawn workers and aggregate results.

    Set ``MNPBEM_SWEEP_INLINE=1`` to run all cases sequentially in the
    *current* process (each one still gets its own subprocess.run call,
    but no multiprocessing fork). This mode exists to make the dispatch
    flow unit-testable without depending on the heavyweight mnpbem kernel
    package.
    """
    if os.environ.get('MNPBEM_SWEEP_INLINE') == '1':
        return _run_pool_inline(cases, plan, verbose, n_wavelengths_override)

    n_workers = plan['n_workers']

    ctx = mp.get_context('spawn')
    queue = ctx.Queue()

    pending = list(range(len(cases)))
    results: Dict[int, Dict[str, Any]] = dict()
    in_flight: Dict[int, Tuple[mp.Process, int]] = dict()  # case_idx -> (proc, worker_slot)
    free_slots = list(range(n_workers))

    while pending or in_flight:

        while pending and free_slots:
            slot = free_slots.pop(0)
            case_idx = pending.pop(0)
            case = cases[case_idx]
            gpu_ids = plan['gpu_ids'][slot] if plan['gpu_ids'] else []

            proc = ctx.Process(
                    target = _run_single_case,
                    args = (case, slot, gpu_ids, plan, verbose,
                            n_wavelengths_override, queue))
            proc.start()
            in_flight[case_idx] = (proc, slot)

        # wait for any completion
        msg = queue.get()
        case_idx = msg['idx']
        results[case_idx] = msg
        proc, slot = in_flight.pop(case_idx)
        proc.join(timeout = 30)
        free_slots.append(slot)

    return [results[i] for i in sorted(results.keys())]


def _run_pool_inline(cases: List[Dict[str, Any]],
        plan: Dict[str, Any],
        verbose: bool,
        n_wavelengths_override: Optional[int]) -> List[Dict[str, Any]]:
    """Test-mode pool: run cases sequentially in the calling process so
    that monkeypatched subprocess.run is honoured.
    """
    out: List[Dict[str, Any]] = []

    for case in cases:
        slot = case['idx'] % plan['n_workers']
        gpu_ids = plan['gpu_ids'][slot] if plan['gpu_ids'] else []

        env = _build_worker_env(gpu_ids, plan['threads_per_worker'])
        cmd = _build_worker_cmd(case, plan,
                verbose = verbose,
                n_wavelengths_override = n_wavelengths_override)

        t0 = time.time()
        proc = subprocess.run(cmd, env = env,
                stdout = subprocess.PIPE, stderr = subprocess.STDOUT,
                check = False)
        wall = time.time() - t0

        out.append({
                'idx': case['idx'],
                'name': case['name'],
                'gpu_id': gpu_ids,
                'worker_slot': slot,
                'returncode': int(proc.returncode),
                'wall_s': wall})

    return out


def _run_single_case(case: Dict[str, Any],
        worker_slot: int,
        gpu_ids: List[int],
        plan: Dict[str, Any],
        verbose: bool,
        n_wavelengths_override: Optional[int],
        queue) -> None:
    """Worker entry — invokes run_simulation.py as a subprocess with its
    own CUDA_VISIBLE_DEVICES and thread limits.
    """
    import traceback

    try:
        env = _build_worker_env(gpu_ids, plan['threads_per_worker'])
        cmd = _build_worker_cmd(case, plan,
                verbose = verbose,
                n_wavelengths_override = n_wavelengths_override)

        t0 = time.time()
        proc = subprocess.run(
                cmd,
                env = env,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                check = False)
        wall = time.time() - t0

        # mirror child stdout (one block per case — good enough for sweep logs)
        out = proc.stdout.decode('utf-8', 'replace') if proc.stdout else ''
        sys.stdout.write('\n=== sweep case [{}] {} (slot {}, gpu {}) ===\n'.format(
                case['idx'], case['name'], worker_slot, gpu_ids))
        sys.stdout.write(out)
        sys.stdout.flush()

        queue.put({
                'idx': case['idx'],
                'name': case['name'],
                'gpu_id': gpu_ids,
                'worker_slot': worker_slot,
                'returncode': int(proc.returncode),
                'wall_s': wall})

    except Exception as exc:
        queue.put({
                'idx': case['idx'],
                'name': case.get('name', '?'),
                'gpu_id': gpu_ids,
                'worker_slot': worker_slot,
                'returncode': -1,
                'wall_s': 0.0,
                'error': repr(exc),
                'traceback': traceback.format_exc()})


def _build_worker_env(gpu_ids: List[int],
        threads_per_worker: int) -> Dict[str, str]:
    env = os.environ.copy()

    if gpu_ids:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
        env['MNPBEM_GPU'] = '1'
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
        env['MNPBEM_GPU'] = '0'

    threads_str = str(threads_per_worker)
    env['OMP_NUM_THREADS'] = threads_str
    env['MKL_NUM_THREADS'] = threads_str
    env['OPENBLAS_NUM_THREADS'] = threads_str
    env['NUMEXPR_NUM_THREADS'] = threads_str
    env['NUMBA_NUM_THREADS'] = threads_str

    # Each worker uses 1 GPU on its own — never inherit VRAM-share state.
    env.pop('MNPBEM_VRAM_SHARE', None)
    env.pop('MNPBEM_VRAM_SHARE_GPUS', None)
    env.pop('MNPBEM_VRAM_SHARE_BACKEND', None)

    return env


def _build_worker_cmd(case: Dict[str, Any],
        plan: Dict[str, Any],
        verbose: bool,
        n_wavelengths_override: Optional[int]) -> List[str]:
    cmd = [
            sys.executable, '-m', 'pymnpbem_simulation.cli',
            '--str-conf', case['str_conf'],
            '--sim-conf', case['sim_conf'],
            '--n-workers', '1',
            '--n-threads', str(plan['threads_per_worker']),
            '--n-gpus-per-worker', str(plan['gpus_per_worker'])]

    # Output naming — each case gets a unique sub-dir to avoid collisions.
    sub_name = plan['output_subdir_pattern'].format(
            idx = case['idx'], name = case['name'])
    cmd.extend(['--simulation-name', sub_name])

    if plan['output_dir']:
        cmd.extend(['--output-dir', plan['output_dir']])

    if verbose:
        cmd.append('--verbose')

    if n_wavelengths_override is not None:
        cmd.extend(['--n-wavelengths', str(int(n_wavelengths_override))])

    return cmd


def _resolve_path(path: str,
        base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _name_from_paths(str_path: str,
        sim_path: str,
        idx: int) -> str:
    base = os.path.splitext(os.path.basename(str_path))[0]

    if base:
        return base

    return 'case{}'.format(idx)


def _short(path: str) -> str:
    parts = path.split(os.sep)

    if len(parts) > 3:
        return os.sep.join(['...'] + parts[-2:])

    return path


def _slug(value: Any) -> str:
    s = str(value)
    return s.replace('.', 'p').replace('-', 'm').replace(' ', '_')
