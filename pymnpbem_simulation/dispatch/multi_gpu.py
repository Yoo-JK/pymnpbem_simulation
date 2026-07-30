import copy
import os
import time

from typing import Any, Dict, List

import numpy as np

from ..util import print_info, print_error


def dispatch_multi_gpu(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    n_workers = int(cfg['compute']['n_workers'])
    n_gpus_per_worker = int(cfg['compute']['n_gpus_per_worker'])

    # Ensure compute.vram_share is materialised into a dict-shaped sub-cfg
    # so downstream paths (single worker + multi-worker pool) can read it
    # uniformly. Honours legacy ``vram_share_backend`` flat key.
    _ensure_vram_share_cfg(cfg)

    # Multi-worker VRAM-share pool: 2+ workers each owning a distinct
    # n_gpus_per_worker GPU partition. Example: 4 GPU + 2 worker + 2 GPU
    # per worker -> worker 0 = GPUs (0, 1), worker 1 = GPUs (2, 3).
    # Each worker runs an independent BEM case under its own VRAM-share
    # cuSolverMg/Magma/NCCL distributed LU.
    if n_workers >= 2 and n_gpus_per_worker >= 2:
        return _dispatch_vram_share_pool(cfg, p, epstab, enei)

    if n_gpus_per_worker > 1:
        return _dispatch_vram_share(cfg, p, epstab, enei)

    return _dispatch_wavelength_split(cfg, p, epstab, enei)


def _dispatch_wavelength_split(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    from mnpbem.utils.multi_gpu import solve_spectrum_multi_gpu

    n_workers = int(cfg['compute']['n_workers'])
    n_wl = len(enei)

    if n_workers > n_wl:
        print_info('multi_gpu: n_workers={} > n_wl={}, capping to n_wl'.format(
                n_workers, n_wl))
        n_workers = n_wl

    pol = cfg['simulation'].get('polarizations',
            [[1, 0, 0], [0, 1, 0]])
    prop = cfg['simulation'].get('propagation_dirs',
            [[0, 0, 1], [0, 0, 1]])

    cfg_pickle = _strip_unpicklable(cfg)
    factory = _make_particle_factory(cfg_pickle)

    bem_kwargs = _build_bem_kwargs(cfg)

    # v1.5.1 Bug 4 follow-up: forward simulation.type to mnpbem so the
    # wavelength-split path picks the right BEM class. Previously this
    # path was hard-coded to BEMRet (dense) in mnpbem.utils.multi_gpu,
    # which OOM'd on 12k+ face Au@Ag dimer when the user requested iter.
    bem_class_name = _resolve_bem_class_name(cfg)

    # Forward iter options (hmatrix, tol, maxit, etc.) to the worker BEM
    # constructor when using an iterative solver.
    if bem_class_name in ('BEMRetIter', 'BEMRetLayerIter'):
        iter_cfg = cfg['simulation'].get('iter', dict()) or dict()
        for key in ('solver', 'tol', 'maxit', 'restart', 'precond',
                    'output', 'hmatrix', 'htol', 'cleaf', 'kmax'):
            if key in iter_cfg:
                bem_kwargs[key] = iter_cfg[key]

    print_info('multi_gpu wavelength-split: n_gpus={}, n_wl={}, n_pol={}, bem_class={}'.format(
            n_workers, n_wl, len(pol), bem_class_name or 'BEMRet'))

    t0 = time.time()
    raw = solve_spectrum_multi_gpu(
            particle_factory = factory,
            enei = enei,
            pol_dirs = pol,
            prop_dirs = prop,
            n_gpus = n_workers,
            bem_kwargs = bem_kwargs,
            bem_class = bem_class_name)
    wall_s = time.time() - t0

    ext = np.asarray(raw['ext'])
    sca = np.asarray(raw['sca'])
    abs_ = ext - sca

    n_pol = ext.shape[1]
    peak_idx = int(np.argmax(ext[:, 0]))
    peak_wl = float(enei[peak_idx])
    peak_ext_x = float(ext[peak_idx, 0])

    print_info('multi_gpu: peak ext_x = {:.3f} at {:.2f} nm'.format(
            peak_ext_x, peak_wl))
    print_info('multi_gpu: total wall = {:.2f} min'.format(wall_s / 60.0))

    return {
            'wavelength': enei,
            'ext': ext,
            'sca': sca,
            'abs': abs_,
            'wall_s': wall_s,
            'warmup_s': 0.0,
            'peak_idx': peak_idx,
            'peak_wl_nm': peak_wl,
            'peak_ext_x': peak_ext_x,
            'n_pol': n_pol,
            'per_gpu_s': raw.get('per_gpu_s', []),
            'n_gpus': raw.get('n_gpus', n_workers)}


def _dispatch_vram_share(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:
    """v1.2.0: VRAM share — pool multi-GPU memory for one large LU factorisation.

    Activated when ``compute.n_gpus_per_worker > 1`` (single worker
    owning all GPUs). The VRAM-share configuration is read from
    ``cfg['compute']['vram_share']`` (preferred) and falls back to the
    legacy flat ``cfg['compute']['vram_share_backend']`` + the
    ``n_gpus_per_worker`` count. Env vars (``MNPBEM_VRAM_SHARE``,
    ``MNPBEM_VRAM_SHARE_GPUS``, ``MNPBEM_VRAM_SHARE_BACKEND``,
    ``MNPBEM_VRAM_SHARE_DEVICE_IDS``) are also set as a transition
    bridge because the installed mnpbem port's ``lu_factor_dispatch``
    still consults env vars at the call site. Both env vars and the
    cfg sub-block are populated so the worker process — which inherits
    a cfg pickle — does not depend on env propagation alone.

    If the installed mnpbem port has not yet adopted the VRAM-share env
    vars (Agent β still in flight), the env vars are simply ignored and
    the run degrades to a single-GPU LU on the default device.
    """
    vs_cfg = _ensure_vram_share_cfg(cfg)

    if not vs_cfg.get('enabled', True):
        print_info('VRAM share dispatch: cfg disabled — falling back to single GPU')
        from .single_node import _dispatch_single_gpu
        return _dispatch_single_gpu(cfg, p, epstab, enei)

    n_gpus = int(vs_cfg['n_gpus'])
    backend = str(vs_cfg.get('backend', 'cusolvermg'))
    device_ids = vs_cfg.get('device_ids', None)

    print_info(
            'VRAM share dispatch: n_gpus={}, backend={}, device_ids={}'.format(
                    n_gpus, backend, device_ids))

    # Bridge: mnpbem's ``lu_factor_dispatch`` still inspects env vars at
    # call time. Setting them here covers in-process single-worker runs
    # and gives a clean fall-back when cfg-pickle propagation is missing.
    saved_env = _apply_vram_share_env(vs_cfg)

    # Single worker handles the full computation; the LU dispatch picks
    # up the env vars internally and partitions across the requested GPUs.
    from .single_node import _dispatch_single_gpu
    try:
        return _dispatch_single_gpu(cfg, p, epstab, enei)
    finally:
        _restore_env(saved_env)


def _ensure_vram_share_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Materialise ``cfg['compute']['vram_share']`` as a dict.

    Reads from the new nested form when present, otherwise composes a
    dict from ``n_gpus_per_worker`` + legacy ``vram_share_backend``.
    Mutates ``cfg`` in place and returns the resulting sub-dict so
    callers can read fields with normal ``.get`` semantics.

    Schema::

        compute:
          vram_share:
            enabled: bool (default True when n_gpus_per_worker > 1)
            n_gpus:  int  (default n_gpus_per_worker)
            backend: str  ('cusolvermg' | 'magma' | 'nccl')
            device_ids: list[int] | None
            distributed: bool (default False) — gate B-3 distributed
                          build (bem_ret.py etc.) via
                          MNPBEM_VRAM_SHARE_DISTRIBUTED env var
    """
    compute = cfg.setdefault('compute', dict())
    existing = compute.get('vram_share', None)

    if isinstance(existing, dict):
        vs_cfg = dict(existing)
    elif isinstance(existing, bool):
        # `vram_share: true` shorthand.
        vs_cfg = {'enabled': bool(existing)}
    else:
        vs_cfg = dict()

    n_gpus_per_worker = int(compute.get('n_gpus_per_worker', 1))

    vs_cfg.setdefault('enabled', n_gpus_per_worker > 1)
    vs_cfg.setdefault('n_gpus', n_gpus_per_worker)

    # Legacy flat key takes precedence only when nested backend is unset
    legacy_backend = compute.get('vram_share_backend', None)
    default_backend = legacy_backend if legacy_backend else 'cusolvermg'
    vs_cfg.setdefault('backend', default_backend)

    vs_cfg.setdefault('device_ids', None)

    # B-3 distributed build gate. Default False so legacy yaml (no
    # ``distributed`` key) keeps the dense in-process build path.
    vs_cfg.setdefault('distributed', False)

    compute['vram_share'] = vs_cfg
    return vs_cfg


def _apply_vram_share_env(vs_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Set ``MNPBEM_VRAM_SHARE_*`` env vars from a vram_share cfg dict.

    Returns a snapshot of the previous values so the caller can restore
    them in a ``finally``. Keys absent from the snapshot are deleted on
    restore.
    """
    keys = ('MNPBEM_VRAM_SHARE',
            'MNPBEM_VRAM_SHARE_GPUS',
            'MNPBEM_VRAM_SHARE_BACKEND',
            'MNPBEM_VRAM_SHARE_DEVICE_IDS',
            'MNPBEM_VRAM_SHARE_DISTRIBUTED')
    saved = {k: os.environ[k] for k in keys if k in os.environ}

    if not vs_cfg.get('enabled', True):
        os.environ['MNPBEM_VRAM_SHARE'] = '0'
        # Clear the rest so the bridge is unambiguously off.
        for k in keys[1:]:
            os.environ.pop(k, None)
        return saved

    os.environ['MNPBEM_VRAM_SHARE'] = '1'
    os.environ['MNPBEM_VRAM_SHARE_GPUS'] = str(int(vs_cfg['n_gpus']))
    os.environ['MNPBEM_VRAM_SHARE_BACKEND'] = str(vs_cfg.get('backend', 'cusolvermg'))

    device_ids = vs_cfg.get('device_ids', None)
    if device_ids:
        os.environ['MNPBEM_VRAM_SHARE_DEVICE_IDS'] = ','.join(
                str(int(d)) for d in device_ids)
    else:
        os.environ.pop('MNPBEM_VRAM_SHARE_DEVICE_IDS', None)

    # B-3 distributed build gate. Only emit the env var when explicitly
    # requested so legacy runs (cfg without ``distributed`` key) keep the
    # in-process dense build path.
    if vs_cfg.get('distributed', False):
        os.environ['MNPBEM_VRAM_SHARE_DISTRIBUTED'] = '1'
    else:
        os.environ.pop('MNPBEM_VRAM_SHARE_DISTRIBUTED', None)

    return saved


def _restore_env(saved: Dict[str, str]) -> None:
    """Restore env keys captured by ``_apply_vram_share_env``.

    Any of the four MNPBEM_VRAM_SHARE_* keys missing from ``saved`` is
    removed so the process returns to its pre-call state.
    """
    keys = ('MNPBEM_VRAM_SHARE',
            'MNPBEM_VRAM_SHARE_GPUS',
            'MNPBEM_VRAM_SHARE_BACKEND',
            'MNPBEM_VRAM_SHARE_DEVICE_IDS',
            'MNPBEM_VRAM_SHARE_DISTRIBUTED')

    for k in keys:
        if k in saved:
            os.environ[k] = saved[k]
        else:
            os.environ.pop(k, None)


def _dispatch_vram_share_pool(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:
    """Multi-worker VRAM-share pool.

    Active when ``compute.n_workers >= 2`` AND
    ``compute.n_gpus_per_worker >= 2``. Each worker owns a disjoint
    block of ``n_gpus_per_worker`` GPUs and runs the BEM dispatch on its
    own GPU partition. Wavelengths are split round-robin across workers
    so each worker handles a sub-spectrum on its local GPU set, while
    the LU factorisation inside each worker is itself partitioned across
    the worker's GPUs via the cuSolverMg / Magma / NCCL backend.

    The per-worker cfg (pickled into the subprocess) carries the
    appropriate ``vram_share`` block including ``device_ids`` so the
    worker's ``lu_factor_dispatch`` selects the correct GPUs. We also
    set ``CUDA_VISIBLE_DEVICES`` for the worker so cupy / cuda
    libraries see only that worker's GPUs (and the matching local index
    space).
    """
    import multiprocessing as mp
    import time

    n_workers = int(cfg['compute']['n_workers'])
    n_gpus_per_worker = int(cfg['compute']['n_gpus_per_worker'])
    n_wl = len(enei)

    if n_workers > n_wl:
        print_info('vram_share_pool: n_workers={} > n_wl={}, capping to n_wl'.format(
                n_workers, n_wl))
        n_workers = n_wl

    total_gpus_needed = n_workers * n_gpus_per_worker
    available_gpus = _list_available_gpu_ids()

    if available_gpus and len(available_gpus) < total_gpus_needed:
        print_error('vram_share_pool: need {} GPUs ({} workers x {} GPUs each) '
                'but only {} visible — partitioning may overlap.'.format(
                        total_gpus_needed,
                        n_workers,
                        n_gpus_per_worker,
                        len(available_gpus)))

    if not available_gpus:
        available_gpus = list(range(total_gpus_needed))

    partitions = _partition_gpus(available_gpus, n_workers, n_gpus_per_worker)

    print_info('vram_share_pool: {} workers x {} GPUs each = {} GPUs total'.format(
            n_workers, n_gpus_per_worker, total_gpus_needed))
    for w, part in enumerate(partitions):
        print_info('vram_share_pool: worker {} -> GPUs {}'.format(w, part))

    chunks = _split_indices(n_wl, n_workers)
    cfg_base = _strip_unpicklable(cfg)
    _ensure_vram_share_cfg(cfg_base)

    ctx = mp.get_context('spawn')
    results_q = ctx.Queue()

    procs = []
    t0 = time.time()

    for w in range(n_workers):
        wl_indices = chunks[w]

        if not wl_indices:
            continue

        # Worker-specific cfg snapshot — embeds the device ids for this
        # worker's VRAM-share partition.
        worker_cfg = copy.deepcopy(cfg_base)
        worker_cfg['compute']['n_workers'] = 1
        # The worker will see only its own GPUs through CUDA_VISIBLE_DEVICES,
        # so its local device index space is 0..n_gpus_per_worker-1.
        worker_cfg['compute']['vram_share']['device_ids'] = list(
                range(n_gpus_per_worker))

        enei_chunk = enei[wl_indices].tolist()
        gpu_ids = partitions[w]

        proc = ctx.Process(
                target = _vram_share_pool_worker,
                args = (w, wl_indices, enei_chunk, worker_cfg, gpu_ids,
                        results_q))
        proc.start()
        procs.append(proc)

    n_pol = _infer_n_pol(cfg)
    ext = np.zeros((n_wl, n_pol))
    sca = np.zeros((n_wl, n_pol))
    abs_ = np.zeros((n_wl, n_pol))
    warm_s = 0.0
    errors = []

    for _ in procs:
        r = results_q.get()

        if not r.get('ok', False):
            errors.append(r)
            continue

        idxs = r['wl_indices']
        ext[idxs] = r['ext']
        sca[idxs] = r['sca']
        abs_[idxs] = r['abs']

        if r.get('warmup_s', 0.0) > warm_s:
            warm_s = r['warmup_s']

    for proc in procs:
        proc.join(timeout = 30)

    wall_s = time.time() - t0

    if errors:
        for err in errors:
            print_error('vram_share_pool worker {} failed: {}'.format(
                    err.get('worker_idx', '?'), err.get('error', 'unknown')))

            if 'traceback' in err:
                print_error(err['traceback'])

        raise RuntimeError(
                '[error] vram_share_pool: {} of {} workers failed'.format(
                        len(errors), len(procs)))

    peak_idx = int(np.argmax(ext[:, 0]))
    peak_wl = float(enei[peak_idx])
    peak_ext_x = float(ext[peak_idx, 0])

    print_info('vram_share_pool: peak ext_x = {:.3f} at {:.2f} nm'.format(
            peak_ext_x, peak_wl))
    print_info('vram_share_pool: total wall = {:.2f} min'.format(wall_s / 60.0))

    return {
            'wavelength': enei,
            'ext': ext,
            'sca': sca,
            'abs': abs_,
            'wall_s': wall_s,
            'warmup_s': warm_s,
            'peak_idx': peak_idx,
            'peak_wl_nm': peak_wl,
            'peak_ext_x': peak_ext_x,
            'n_pol': n_pol,
            'n_gpus': total_gpus_needed,
            'n_workers': n_workers}


def _vram_share_pool_worker(worker_idx: int,
        wl_indices: List[int],
        enei_chunk: List[float],
        cfg: Dict[str, Any],
        gpu_ids: List[int],
        queue) -> None:
    """Worker entry-point for the VRAM-share pool.

    Pins the worker to its assigned GPU partition via
    CUDA_VISIBLE_DEVICES, materialises the env-var bridge from the
    cfg-driven vram_share block, then runs the BEM simulation on the
    worker's wavelength slice.
    """
    import time
    import traceback

    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
        os.environ['MNPBEM_GPU'] = '1'

        n_threads = int(cfg['compute'].get('n_threads', 1))
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        os.environ['MKL_NUM_THREADS'] = str(n_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
        os.environ['NUMBA_NUM_THREADS'] = str(n_threads)

        # Re-establish env-var bridge in this subprocess — the parent's
        # env mutation does not propagate through ``spawn``. The cfg
        # pickle is the source of truth.
        vs_cfg = _ensure_vram_share_cfg(cfg)
        _apply_vram_share_env(vs_cfg)

        from ..structures import build_structure
        from ..simulation.base import run_simulation

        cfg_struct = cfg['structure']
        cfg_materials = cfg.get('materials', dict())

        p, epstab, _ = build_structure(cfg_struct, cfg_materials)

        enei_arr = np.asarray(enei_chunk, dtype = float)
        t0 = time.time()
        local = run_simulation(cfg, p, epstab, enei_arr)
        wall = time.time() - t0

        queue.put({
                'ok': True,
                'worker_idx': worker_idx,
                'wl_indices': list(wl_indices),
                'ext': np.asarray(local['ext']),
                'sca': np.asarray(local['sca']),
                'abs': np.asarray(local['abs']),
                'wall_s': wall,
                'warmup_s': float(local.get('warmup_s', 0.0))})

    except Exception as exc:
        queue.put({
                'ok': False,
                'worker_idx': worker_idx,
                'wl_indices': list(wl_indices),
                'error': repr(exc),
                'traceback': traceback.format_exc()})


def _list_available_gpu_ids() -> List[int]:
    """Best-effort discovery of GPU ids visible to this process.

    Honours an existing ``CUDA_VISIBLE_DEVICES`` (interpreting its
    entries as the physical id space the parent already restricted to).
    Falls back to ``nvidia-smi -L`` count when no env restriction is
    present, then to an empty list.
    """
    raw = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    if raw is not None and raw.strip() != '':
        try:
            return [int(x) for x in raw.split(',') if x.strip()]
        except ValueError:
            return []

    try:
        import subprocess
        out = subprocess.check_output(['nvidia-smi', '-L'],
                stderr = subprocess.DEVNULL).decode('utf-8', errors = 'ignore')
        return list(range(out.count('GPU ')))
    except Exception:
        return []


def _partition_gpus(gpu_ids: List[int],
        n_workers: int,
        n_gpus_per_worker: int) -> List[List[int]]:
    """Partition ``gpu_ids`` into ``n_workers`` consecutive blocks of
    ``n_gpus_per_worker``.

    If the pool is smaller than required, the blocks wrap (which can
    lead to oversubscription; the caller prints a warning in that
    case).
    """
    partitions: List[List[int]] = []

    if not gpu_ids:
        return [list(range(w * n_gpus_per_worker,
                (w + 1) * n_gpus_per_worker))
                for w in range(n_workers)]

    pool = list(gpu_ids)

    for w in range(n_workers):
        block = []
        for k in range(n_gpus_per_worker):
            block.append(pool[(w * n_gpus_per_worker + k) % len(pool)])
        partitions.append(block)

    return partitions


def _split_indices(n_total: int,
        n_workers: int) -> List[List[int]]:
    chunks: List[List[int]] = [[] for _ in range(n_workers)]
    for i in range(n_total):
        chunks[i % n_workers].append(i)
    return chunks


def _infer_n_pol(cfg: Dict[str, Any]) -> int:
    pol = cfg['simulation'].get('polarizations',
            [[1, 0, 0], [0, 1, 0]])
    return len(pol)


def _make_particle_factory(cfg: Dict[str, Any]):
    import functools
    cfg_struct = cfg['structure']
    cfg_materials = cfg.get('materials', dict())

    return functools.partial(_particle_factory_top, cfg_struct, cfg_materials)


def _particle_factory_top(cfg_struct: Dict[str, Any],
        cfg_materials: Dict[str, Any]):
    from pymnpbem_simulation.structures import build_structure
    p, _, _ = build_structure(cfg_struct, cfg_materials)
    return p


def _build_bem_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    bem_kwargs = dict()

    hmode = cfg['compute'].get('hmode', 'dense')

    if hmode != 'dense':
        bem_kwargs['hmode'] = hmode

    # v1.5.1 — propagate iter-only kwargs to the worker BEM class so the
    # wavelength-split path actually uses the requested H-matrix +
    # preconditioner combination (otherwise mnpbem's worker silently runs
    # plain BEMRetIter dense which still OOMs on 12k face).
    iter_cfg = cfg['compute'].get('iter', dict()) or dict()
    sim_type = cfg['simulation'].get('type', 'ret')
    is_iter_path = sim_type.endswith('_iter') or bool(cfg['compute'].get('iterative', False))

    if is_iter_path:
        if 'hmatrix' in iter_cfg:
            v = iter_cfg['hmatrix']
            if isinstance(v, str) and v.lower() == 'auto':
                bem_kwargs['hmatrix'] = True
            else:
                bem_kwargs['hmatrix'] = bool(v)
        else:
            # default ON for large mesh — matches docs/PERFORMANCE.md
            bem_kwargs['hmatrix'] = True

        if 'preconditioner' in iter_cfg:
            bem_kwargs['preconditioner'] = iter_cfg['preconditioner']
        else:
            bem_kwargs['preconditioner'] = 'auto'

        if 'schur' in iter_cfg:
            v = iter_cfg['schur']
            if isinstance(v, str) and v.lower() == 'auto':
                pass  # let mnpbem decide
            else:
                bem_kwargs['schur'] = bool(v)

        if 'htol' in iter_cfg:
            bem_kwargs['htol'] = float(iter_cfg['htol'])

        if 'tol' in iter_cfg:
            bem_kwargs['tol'] = float(iter_cfg['tol'])

        if 'maxit' in iter_cfg:
            bem_kwargs['maxit'] = int(iter_cfg['maxit'])

    return bem_kwargs


def _resolve_bem_class_name(cfg: Dict[str, Any]) -> str:
    """Return the mnpbem BEM class name implied by simulation.type.

    Mirrors the runner registry used by single-node dispatch.
    """
    sim_type = cfg.get('simulation', {}).get('type', 'ret')

    table = {
            'ret':                      'BEMRet',
            'ret_iter':                 'BEMRetIter',
            'ret_layer':                'BEMRetLayer',
            'ret_layer_iter':           'BEMRetLayerIter',
            'planewave_ret':            'BEMRet',
            'planewave_ret_iter':       'BEMRetIter',
            'planewave_ret_layer':      'BEMRetLayer',
            'planewave_ret_layer_iter': 'BEMRetLayerIter'}

    return table.get(sim_type, 'BEMRet')


def _strip_unpicklable(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import copy
    out = copy.deepcopy(cfg)
    from pymnpbem_simulation.util import assert_no_callables
    assert_no_callables(out)
    return out